"""Additional two-pass CoT format control over frozen cached reasoning outputs.

The original single-pass condition is retained. This additional condition was
introduced after observing output-format failures, without altering its raw
responses, parser or scores. It never reads reference labels.
"""
from .io import ROOT,load_json,read_jsonl,append_jsonl,save_json,digest,file_hash
from .backend import QwenBackend,parse_answer
from .images import compose_view
from collections import defaultdict


def run():
    cfg=load_json(ROOT/'configs/pilot.json');manifest=load_json(ROOT/'data/processed/manifest.json');test=[r for r in manifest if r['split']=='test'];byframe=defaultdict(list)
    for r in manifest:byframe[r['frame_id']].append(r)
    out=ROOT/'results/inference';reason={r['id']:r for r in read_jsonl(out/'cot.jsonl')}
    if not all(r['id'] in reason for r in test):raise ValueError('Original reasoning condition must be complete')
    provenance={'condition':'Additional two-pass CoT output-format control','manifest_sha256':digest(manifest),'config_sha256':digest(cfg),'reasoning_file_sha256':file_hash(out/'cot.jsonl'),'timing':'Original reasoning pass plus finalization pass; shared image composition measured by evaluator.','status':'Introduced after initial strict-format scoring; original single-pass scores and raw responses retained. No labels used in this module.'}
    save_json(out/'cot_two_pass_provenance.json',provenance)
    backend=QwenBackend(cfg);done={r['id'] for r in read_jsonl(out/'cot_two_pass.jsonl')}
    first=test[0];warm=compose_view(first['images'],ROOT/'data/images',[q['question'] for q in byframe[first['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height']);backend.feature(warm,first['question'])
    for i,r in enumerate(test):
        if r['id'] in done:continue
        im=compose_view(r['images'],ROOT/'data/images',[q['question'] for q in byframe[r['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
        prompt='Traffic-scene question: '+r['question']+'\nEarlier model reasoning, which may contain errors:\n'+reason[r['id']]['raw_response']+'\nReview the images and question. Give the final answer using exactly one word: Yes or No. Do not explain.'
        raw,t=backend.generate(im,prompt,cfg['direct_max_new_tokens']);timing=dict(t);timing['finalization_seconds']=t['total_seconds'];timing['reasoning_seconds']=reason[r['id']]['timing']['total_seconds'];timing['total_seconds']+=timing['reasoning_seconds']
        append_jsonl(out/'cot_two_pass.jsonl',{'id':r['id'],'prediction':parse_answer(raw,'direct'),'raw_response':raw,'earlier_reasoning':reason[r['id']]['raw_response'],'timing':timing})
        if i%10==0:print('COT FINALIZATION',i+1,'/',len(test),flush=True)
    print('COT FORMAT CONTROL COMPLETE',flush=True)

if __name__=='__main__':run()
