"""Separate, wording-selected diagnostic on scenes excluded from the pilot.

No reference answers enter prompts. Selection precedes inference and scoring.
"""
import json,re,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image,ImageDraw
from .io import ROOT,load_json,save_json,append_jsonl,read_jsonl,digest
from .data import binary_label,question_intent,image_relpath,TAG,CAMERAS
from .images import compose_view,cameras_for_question
from .rules import parse_facts,verify
from .experiments import load_classifier

DEST=ROOT/'results/final_demo'
VALUES={'yes','no','unknown','not_applicable'}
KINDS={'vehicle','pedestrian','cyclist','traffic_sign','traffic_light','barrier','other','unknown','not_applicable'}
FIELDS=['object_type','crossing','in_ego_path','obstacle_blocks_path','front_traffic_element']
PATTERN=re.compile(r'proceed|accelerat|stop|yield|cross|obstruct|block|traffic (?:light|sign|element)|road barrier|consider|turn|merg',re.I)

def parse_targeted(raw):
    try:
        obj=json.loads(str(raw).strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
        if set(obj)!=set(FIELDS):raise ValueError()
        for field in FIELDS:
            cell=obj[field]
            if not isinstance(cell,dict) or set(cell)!={'value','status'}:raise ValueError()
            v=cell['value'];status=cell['status'];allowed=KINDS if field=='object_type' else VALUES
            if v not in allowed or status not in {'known','unknown','not_applicable'}:raise ValueError()
            if status!=('unknown' if v=='unknown' else 'not_applicable' if v=='not_applicable' else 'known'):raise ValueError()
        return {'valid':True,'fields':obj}
    except (ValueError,TypeError,KeyError):return {'valid':False,'fields':{}}

def relevant_view(row):
    context=compose_view(row['images'],ROOT/'data/images',[row['question']],all_cameras=True)
    tags=TAG.findall(row['question']);cam=tags[0][1] if tags else cameras_for_question(row['question'])[0]
    cam=cam if cam in row['images'] else 'CAM_FRONT'
    with Image.open(ROOT/'data/images'/row['images'][cam]) as im:
        im=im.convert('RGB');w,h=im.size
        if tags:
            px,py=map(float,tags[0][2:]);px=min(w,max(0,px));py=min(h,max(0,py));rad=max(96,int(min(w,h)*.2))
            box=(max(0,int(px-rad)),max(0,int(py-rad)),min(w,int(px+rad)),min(h,int(py+rad)))
            crop=im.crop(box)
        else:box=(0,0,w,h);crop=im.copy()
        crop.thumbnail((672,420))
    panel=Image.new('RGB',(context.width,context.height+450),'white');panel.paste(context,(0,0));panel.paste(crop,((context.width-crop.width)//2,context.height+28));ImageDraw.Draw(panel).text((12,context.height+6),'Question-local detail: '+cam,fill='black')
    return panel,{'camera':cam,'box':box,'query_only':True}

def legacy(row,parsed):
    if not parsed['valid']:return parse_facts(None)
    f=parsed['fields'];refs=TAG.findall(row['question']);cid=refs[0][0] if len(refs)==1 else None
    def tri(key):return {'yes':True,'no':False}.get(f.get(key,{}).get('value'))
    obj={cid:{'kind':f['object_type']['value'] if f.get('object_type',{}).get('value') in KINDS-{'not_applicable'} else 'unknown','in_ego_path':tri('in_ego_path')}} if cid else {}
    crossing=tri('crossing');path=tri('in_ego_path')
    # Pedestrian hazard requires BOTH crossing and path intersection to be known true.
    hazard=True if f.get('object_type',{}).get('value')=='pedestrian' and crossing is True and path is True else None
    return parse_facts({'objects':obj,'pedestrian_crossing_path':hazard,'obstacle_blocks_path':tri('obstacle_blocks_path'),'front_traffic_element':tri('front_traffic_element')})

def prepare(n=100):
    if (DEST/'manifest.json').exists():return load_json(DEST/'manifest.json')
    pilot=load_json(ROOT/'data/processed/manifest.json');used={r['scene_id'] for r in pilot};raw=load_json(ROOT/'data/raw/v1_0_train_nus.json');candidates=[];labels={}
    for sid,scene in raw.items():
        if sid in used:continue
        for fid,frame in scene.get('key_frames',{}).items():
            images={c:image_relpath(p) for c,p in frame.get('image_paths',{}).items() if c in CAMERAS}
            if len(images)!=6:continue
            for cat,qas in frame.get('QA',{}).items():
                for qa in qas:
                    q=qa.get('Q','');y=binary_label(q,qa.get('A',''))
                    if y is None or not PATTERN.search(q):continue
                    rid=digest([sid,fid,cat,q])[:24];intent=question_intent(q)
                    row={'id':rid,'scene_id':sid,'frame_id':fid,'category':cat,'question':q,'images':images,'intent':intent,'split':'targeted_diagnostic'}
                    candidates.append(row);labels[rid]=y
    # Round robin over wording-defined intents, with seeded hash ordering; one frame each.
    groups={k:[] for k in ['proceed','stop','consider','front_traffic','traffic_object','other']}
    for r in sorted(candidates,key=lambda r:digest(['final-demo-v1',r['id']])):groups[r['intent']].append(r)
    selected=[];frames=set()
    while len(selected)<n and any(groups.values()):
        for key in groups:
            while groups[key] and groups[key][-1]['frame_id'] in frames:groups[key].pop()
            if groups[key] and len(selected)<n:
                r=groups[key].pop();selected.append(r);frames.add(r['frame_id'])
    save_json(DEST/'manifest.json',selected);save_json(DEST/'labels.json',{r['id']:labels[r['id']] for r in selected})
    save_json(DEST/'protocol.json',{'n':len(selected),'eligible_questions':len(candidates),'excluded_pilot_scenes':len(used),'scene_overlap':0,'intent_counts':dict(Counter(r['intent'] for r in selected)),'selection':'Wording only; round robin intent, deterministic hash, one question per frame; labels separate','notice':'Targeted diagnostic, not representative benchmark. No tuning on these outcomes.'})
    needed={v for r in selected for v in r['images'].values()}
    with zipfile.ZipFile(ROOT/'data/raw/drivelm_images.zip') as z:
        for info in z.infolist():
            parts=Path(info.filename).parts
            if 'samples' not in parts:continue
            rel=Path(*parts[parts.index('samples'):])
            if str(rel) not in needed:continue
            p=ROOT/'data/images'/rel
            if not p.resolve().is_relative_to((ROOT/'data/images').resolve()):raise ValueError('Unsafe image path')
            p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():p.write_bytes(z.read(info))
    return selected

PROMPT='''Inspect the six-camera context and the enlarged query-local detail. Extract only facts relevant to this question: {question}
Return one JSON object with exactly these keys: object_type, crossing, in_ego_path, obstacle_blocks_path, front_traffic_element.
Each value must be an object with exactly value and status keys.
object_type.value must be vehicle, pedestrian, cyclist, traffic_sign, traffic_light, barrier, other, unknown, or not_applicable. All other .value fields must be yes, no, unknown, or not_applicable.
status must be known for an observed value, unknown for unknown, or not_applicable for not_applicable. A red marker is a location, never a category.
Only assert in_ego_path if the ego path is visually identifiable. Crossing alone does not imply path intersection. obstacle_blocks_path means a visible obstruction blocks that path. front_traffic_element means a visible sign, light, or barrier in CAM_FRONT.
Use unknown for ambiguous evidence. Use not_applicable for facts unrelated to the question. Do not answer the question. No markdown or extra keys.'''

def run():
    from .backend import QwenBackend
    rows=prepare();cfg=load_json(ROOT/'configs/pilot.json');cfg={**cfg,'max_pixels':401408}
    # Isolate runtime provenance from the frozen pilot.
    before=(ROOT/'results/runtime.json').read_bytes()
    backend=QwenBackend(cfg);save_json(DEST/'runtime.json',backend.provenance);(ROOT/'results/runtime.json').write_bytes(before)
    done={r['id'] for r in read_jsonl(DEST/'outputs.jsonl')}
    for i,r in enumerate(rows):
        if r['id'] in done:continue
        context=compose_view(r['images'],ROOT/'data/images',[r['question']],all_cameras=True)
        # Original pixel budget preserved for the classifier features.
        oldmax=backend.processor.image_processor.max_pixels;backend.processor.image_processor.max_pixels=200704
        h,ft=backend.feature(context,r['question']);backend.processor.image_processor.max_pixels=oldmax
        (DEST/'features').mkdir(exist_ok=True);np.save(DEST/'features'/(r['id']+'.npy'),h)
        panel,region=relevant_view(r);raw,t=backend.generate(panel,PROMPT.format(question=r['question']),240);parsed=parse_targeted(raw)
        oldfacts,oldraw,oldt=backend.facts(context,[r['question']])
        append_jsonl(DEST/'outputs.jsonl',{'id':r['id'],'targeted':parsed,'facts':legacy(r,parsed),'raw':raw,'original_facts':oldfacts,'original_raw':oldraw,'region':region,'timing':t,'original_timing':oldt,'feature_timing':ft})
        print('TARGETED',i+1,'/',len(rows),'schema valid',parsed['valid'],flush=True)
    score()

def score():
    rows=load_json(DEST/'manifest.json');ys=load_json(DEST/'labels.json');outputs={r['id']:r for r in read_jsonl(DEST/'outputs.jsonl')};s=load_json(ROOT/'results/training/selection.json')
    # Depth choice uses ONLY the original validation scores, frozen before diagnostic scoring.
    depth=min(['gru_1','latent_gru','gru_5'],key=lambda k:-s['candidates'][k][str(s['selected'][k]['learning_rate'])]['mean_validation_macro_f1'])
    save_json(DEST/'depth_selection.json',{'selected':depth,'policy':'Highest original validation macro-F1 across depths; no diagnostic labels used.'})
    model,ckpt=load_classifier(ROOT/s['selected'][depth]['fits'][0]['checkpoint']);torch.set_num_threads(4);traces=[]
    for r in rows:
        o=outputs[r['id']];h=np.load(DEST/'features'/(r['id']+'.npy'));x=torch.tensor((h-np.array(ckpt['mean'],dtype=np.float32))/np.array(ckpt['std'],dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():pred=int(model(x).argmax(-1)[0])
        trace={'id':r['id'],'reference':ys[r['id']],'neural':pred}
        for name,key in [('original','original_facts'),('targeted','facts')]:trace[name]=verify(r['question'],pred,o[key])
        traces.append(trace)
    from .metrics import classification_metrics
    metrics={'n':len(rows),'selected_depth':depth,'neural':classification_metrics([t['reference'] for t in traces],[t['neural'] for t in traces]),'valid_schema_rate':sum(o['targeted']['valid'] for o in outputs.values())/len(rows)}
    for name in ['original','targeted']:
        m=classification_metrics([t['reference'] for t in traces],[t[name]['final'] for t in traces]);m.update({'covered':sum(bool(t[name]['checks']) for t in traces),'beneficial':sum(t['neural']!=t['reference'] and t[name]['final']==t['reference'] for t in traces),'harmful':sum(t['neural']==t['reference'] and t[name]['final']!=t['reference'] for t in traces),'unresolved':sum(t[name]['status'].startswith('unresolved') for t in traces)})
        metrics[name]=m
    save_json(DEST/'metrics.json',metrics);save_json(DEST/'traces.json',traces);print(json.dumps(metrics),flush=True)

if __name__=='__main__':run()
