"""Local-only demonstration using actual cached VLM features and trained heads."""
from pathlib import Path
from functools import lru_cache
import io,json
import numpy as np
import torch
from fastapi import FastAPI,HTTPException,Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field
from .io import ROOT,load_json,read_jsonl
from .images import compose_view
from .rules import verify,parse_facts,graph_json,active_checks
from .experiments import load_classifier

app=FastAPI(title='NSL-CoT Traffic Lab',version='1.0.0')

@lru_cache
def manifest():return load_json(ROOT/'data/processed/manifest.json')
@lru_cache
def records():return {r['id']:r for r in manifest()}
@lru_cache
def labels():return load_json(ROOT/'data/processed/labels.json')
def facts():return {r['frame_id']:r['facts'] for r in read_jsonl(ROOT/'results/inference/facts.jsonl')}
@lru_cache
def selection():return load_json(ROOT/'results/training/selection.json')
@lru_cache
def checkpoint(name):
    if name not in ['feedforward','latent_gru','matched_feedforward','gru_1','gru_5']:raise HTTPException(400,'Unknown model')
    return load_classifier(ROOT/selection()['selected'][name]['fits'][0]['checkpoint'])

def lookup(record_id):
    try:return records()[record_id]
    except KeyError:raise HTTPException(404,'Unknown question identifier')

@app.get('/',response_class=HTMLResponse)
def home():return (ROOT/'demo/index.html').read_text()

@app.get('/api/status')
def status():
    p=ROOT/'results/evaluation/metrics.json'
    stats=load_json(ROOT/'data/processed/statistics.json');nf=len(list((ROOT/'results/inference/features').glob('*.npy')));nframes=sum(s['frames'] for s in stats['selected'].values());nq=sum(s['questions'] for s in stats['selected'].values());ff=len(read_jsonl(ROOT/'results/inference/facts.jsonl'));baselines={mode:len(read_jsonl(ROOT/'results/inference'/(mode+'.jsonl'))) for mode in ['direct','cot','graph']};testn=stats['selected'].get('test',{}).get('questions',0)
    phase='Experiment complete' if p.exists() else f'Encoding features {nf} / {nq}' if nf<nq else f'Extracting scene facts {ff} / {nframes}' if ff<nframes else f'Prompting baselines {sum(baselines.values())} / {3*testn}' if sum(baselines.values())<3*testn else 'Computing held-out metrics'
    return {'ready':p.exists(),'features':nf,'facts':ff,'baselines':baselines,'phase':phase,'dataset':stats,'images_ready':all((ROOT/'data/images'/p).exists() for r in manifest() for p in r['images'].values()),'runtime':load_json(ROOT/'results/runtime.json') if (ROOT/'results/runtime.json').exists() else None}

@app.get('/api/metrics')
def metrics():
    p=ROOT/'results/evaluation/metrics.json'
    if not p.exists():raise HTTPException(503,'Evaluation is still running')
    return load_json(p)

@app.get('/api/questions')
def questions(split:str='test'):
    if split not in ['train','validation','test']:raise HTTPException(400,'Invalid split')
    fm=facts();trace_path=ROOT/'results/evaluation/rule_traces.json';changed={t['id'] for t in load_json(trace_path).get('latent_gru/seed18',[]) if t['changed']} if trace_path.exists() else set()
    return [{'id':r['id'],'question':r['question'],'category':r['category'],'intent':r['intent'],'scene_id':r['scene_id'],'rule_covered':bool(active_checks(r['question'],fm.get(r['frame_id'],parse_facts(None)))),'gru_changed':r['id'] in changed} for r in manifest() if r['split']==split]

@app.get('/api/image/{record_id}')
def image(record_id:str):
    r=lookup(record_id)
    if not all((ROOT/'data/images'/p).exists() for p in r['images'].values()):
        raise HTTPException(409,'Source images are downloaded separately. Run python scripts/prepare_demo_images.py --archive /path/to/drivelm_images.zip')
    im=compose_view(r['images'],ROOT/'data/images',[rr['question'] for rr in manifest() if rr['frame_id']==r['frame_id']],all_cameras=True);buf=io.BytesIO();im.save(buf,format='JPEG',quality=92)
    return Response(buf.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'public, max-age=3600'})

class PredictRequest(BaseModel):
    record_id:str
    model:str='ensemble'
    automatic_correction:bool=False
    # Optional what-if facts affect this request only and are never written to the benchmark.
    facts_override:dict|None=None

@app.post('/api/predict')
def predict(req:PredictRequest):
    r=lookup(req.record_id)
    h=np.load(ROOT/'results/inference/features'/(r['id']+'.npy'),allow_pickle=False)
    if req.model=='ensemble':
        from .prediction import ensemble_probabilities
        prob=ensemble_probabilities(h,ROOT)
    else:
        model,ckpt=checkpoint('gru_5' if req.model=='adaptive_gru' else req.model)
        from .prediction import standardized
        xx=standardized(h,ckpt)
        with torch.inference_mode():
            if req.model=='adaptive_gru':
                from .demo_analysis import adaptive_evolution
                threshold=load_json(ROOT/'results/adaptive/selection.json')['selected']['threshold']
                yes=adaptive_evolution(model,xx,threshold)[-1]['yes'];prob=[1-yes,yes]
            else:prob=torch.softmax(model(xx),-1)[0].tolist()
    fact_map=facts();pending=r['frame_id'] not in fact_map;f=fact_map.get(r['frame_id'],parse_facts(None));whatif=req.facts_override is not None
    if whatif:f=parse_facts(req.facts_override)
    v=verify(r['question'],int(prob[1]>prob[0]),f)
    v['proposed_final']=v['final'];v['proposed_changed']=v['changed'];v['checker_status']=v['status']
    v['mode']='automatic' if req.automatic_correction else 'advisory'
    if not req.automatic_correction:
        v['final']=v['original'];v['changed']=False;v['violations_after']=list(v['violations_before'])
        if v['proposed_changed']:v['status']='advisory_contradiction'

    return {'record':r,'reference_answer':'yes' if labels()[r['id']] else 'no','probabilities':{'no':prob[0],'yes':prob[1]},'verification':v,'facts':f,'graph':graph_json(f),'what_if':whatif,'facts_pending':pending,'seed':None if req.model=='ensemble' else 18,'seeds':[18,42,572] if req.model=='ensemble' else [18],'model':req.model,'feature_source':'Cached real Qwen2.5-VL-3B-Instruct final prompt-token representation','notice':'What-if changes are illustrative and excluded from benchmark results.' if whatif else 'Scene facts are still processing for this frame. Neural-only result shown.' if pending else 'Offline research example; not a vehicle-control decision.'}

@app.get('/api/raw/{record_id}')
def raw(record_id:str):
    r=lookup(record_id);outputs={}
    for mode in ['direct','cot','graph','cot_two_pass']:
        row=next((v for v in read_jsonl(ROOT/'results/inference'/(mode+'.jsonl')) if v['id']==record_id),None)
        if row:outputs[mode]=row
    ff=next((v for v in read_jsonl(ROOT/'results/inference/facts.jsonl') if v['frame_id']==r['frame_id']),None)
    return {'baselines':outputs,'fact_extraction':ff}

@app.get('/api/compare/{record_id}')
def compare_models(record_id:str):
    lookup(record_id)
    from .demo_analysis import compare
    return compare(record_id)

@app.get('/api/gallery')
def gallery():
    traces=load_json(ROOT/'results/evaluation/rule_traces.json')['latent_gru/seed18']
    truth=labels();examples={}
    for t in sorted(traces,key=lambda x:x['id']):
        pred=t['original'];final=t['final'];correct=pred==truth[t['id']]
        kind='beneficial' if t['changed'] and final==truth[t['id']] else 'harmful' if t['changed'] else 'unknown' if not t['checks'] else 'neural_correct' if correct else 'neural_failure'
        if kind not in examples:examples[kind]={'id':t['id'],'kind':kind,'question':lookup(t['id'])['question'],'status':t['status'],'model':'latent_gru','automatic_correction':kind in ['beneficial','harmful']}
    ensemble=load_json(ROOT/'results/ensemble/metrics.json')['predictions']
    predictions=load_json(ROOT/'results/evaluation/predictions.json')
    ff={r['id']:r['prediction'] for r in predictions['feedforward/seed18']}
    gru={r['id']:r['prediction'] for r in predictions['latent_gru/seed18']}
    direct={r['id']:r['prediction'] for r in read_jsonl(ROOT/'results/inference/direct.jsonl')}
    normal=next((r for r in ensemble if r['prediction']==truth[r['id']] and ff[r['id']]==truth[r['id']] and gru[r['id']]==truth[r['id']] and direct.get(r['id'])==truth[r['id']]),None)
    if normal:
        examples={'normal':{'id':normal['id'],'kind':'normal','question':lookup(normal['id'])['question'],'model':'ensemble','automatic_correction':False},**examples}
    return {'examples':list(examples.values()),'notice':'Examples selected to demonstrate behavior, not estimate prevalence. Actual pilot safety-action correction is unavailable; no fabricated success is shown.'}

@app.get('/api/final-demo')
def final_demo_results():
    dest=ROOT/'results/final_demo'
    return {'ready':(dest/'metrics.json').exists(),'completed':len(read_jsonl(dest/'outputs.jsonl')),'protocol':load_json(dest/'protocol.json') if (dest/'protocol.json').exists() else None,'metrics':load_json(dest/'metrics.json') if (dest/'metrics.json').exists() else None,'human_audit':load_json(dest/'human_audit/summary.json') if (dest/'human_audit/summary.json').exists() else {'completed':False},'ensemble':load_json(ROOT/'results/ensemble/metrics.json'),'adaptive':load_json(ROOT/'results/adaptive/metrics.json')}

@app.get('/api/targeted-cases')
def targeted_cases():
    path=ROOT/'results/final_demo/manifest.json'
    if not path.exists():return []
    done={r['id'] for r in read_jsonl(ROOT/'results/final_demo/outputs.jsonl')}
    return [{'id':r['id'],'question':r['question']} for r in load_json(path) if r['id'] in done]

def targeted_record(record_id):
    p=ROOT/'results/final_demo/manifest.json'
    row=next((r for r in load_json(p) if r['id']==record_id),None) if p.exists() else None
    if row is None:raise HTTPException(404,'Unknown diagnostic record')
    return row

@app.get('/api/targeted-case/{record_id}')
def targeted_case(record_id:str):
    r=targeted_record(record_id);dest=ROOT/'results/final_demo'
    o=next((r for r in read_jsonl(dest/'outputs.jsonl') if r['id']==record_id),None)
    if o is None:raise HTTPException(503,'This example is still processing')
    traces=load_json(dest/'traces.json') if (dest/'traces.json').exists() else []
    return {'record':r,'extraction':o,'evaluation':next((t for t in traces if t['id']==record_id),None),'notice':'Targeted diagnostic; inference excluded reference labels. Human fact audit pending.'}

@app.get('/api/targeted-image/{record_id}')
def targeted_image(record_id:str):
    from .targeted import relevant_view
    r=targeted_record(record_id)
    if not all((ROOT/'data/images'/p).exists() for p in r['images'].values()):
        raise HTTPException(409,'Download the source image archive and run scripts/prepare_demo_images.py')
    im,_=relevant_view(r);buf=io.BytesIO();im.save(buf,format='JPEG',quality=90)
    return Response(buf.getvalue(),media_type='image/jpeg')
