"""Recompute published predictions and checker behavior without loading Qwen."""
import json
import numpy as np
import torch
from fastapi.testclient import TestClient
from nslcot.io import ROOT,load_json,read_jsonl,digest,file_hash,save_json
from nslcot.experiments import load_classifier
from nslcot.prediction import ensemble_probabilities
from nslcot.rules import verify
from nslcot.api import app

def main():
 torch.set_num_threads(4)
 rows=load_json(ROOT/'data/processed/manifest.json');test=[r for r in rows if r['split']=='test'];selection=load_json(ROOT/'results/training/selection.json')
 assert digest(rows)==selection['inputs']['manifest_sha256']
 y=load_json(ROOT/'data/processed/labels.json');expected=load_json(ROOT/'results/evaluation/predictions.json')
 h=np.stack([np.load(ROOT/'results/inference/features'/(r['id']+'.npy'),allow_pickle=False) for r in test]);checked=0
 for name,choice in selection['selected'].items():
  for seed,fit in zip([18,42,572],choice['fits']):
   model,c=load_classifier(ROOT/fit['checkpoint']);x=torch.tensor((h-np.array(c['mean'],dtype=np.float32))/np.array(c['std'],dtype=np.float32))
   with torch.inference_mode():pred=model(x).argmax(-1).tolist()
   saved={r['id']:r['prediction'] for r in expected[f'{name}/seed{seed}']}
   assert pred==[saved[r['id']] for r in test];checked+=len(test)
 ensemble=load_json(ROOT/'results/ensemble/metrics.json');saved={r['id']:r['prediction'] for r in ensemble['predictions']};pred=[]
 for r,feature in zip(test,h):
  p=ensemble_probabilities(feature);v=int(p[1]>p[0]);assert v==saved[r['id']];pred.append(v)
 accuracy=sum(v==y[r['id']] for v,r in zip(pred,test))/len(test);assert abs(accuracy-ensemble['test']['accuracy'])<1e-12
 diag=load_json(ROOT/'results/final_demo/manifest.json');assert {r['scene_id'] for r in rows}.isdisjoint({r['scene_id'] for r in diag})
 sources={r['id']:r for r in read_jsonl(ROOT/'results/final_demo/outputs.jsonl')};traces=load_json(ROOT/'results/final_demo/traces.json');diagrows={r['id']:r for r in diag}
 for t in traces:
  for condition,key in [('original','original_facts'),('targeted','facts')]:
   assert verify(diagrows[t['id']]['question'],t['neural'],sources[t['id']][key])==t[condition]
 metrics=load_json(ROOT/'results/final_demo/metrics.json');assert sum(t['neural']==t['reference'] for t in traces)==62;assert sum(t['targeted']['final']==t['reference'] for t in traces)==50
 client=TestClient(app);before=file_hash(ROOT/'results/inference/facts.jsonl')
 for r in test:
  a=client.post('/api/predict',json={'record_id':r['id']});assert a.status_code==200;v=a.json();assert v['model']=='ensemble' and v['verification']['mode']=='advisory';assert v['verification']['original']==saved[r['id']]==v['verification']['final']
  b=client.post('/api/predict',json={'record_id':r['id'],'automatic_correction':True}).json();assert b['verification']['final']==b['verification']['proposed_final']
 assert file_hash(ROOT/'results/inference/facts.jsonl')==before
 result={'checkpoint_predictions_matched':checked,'ensemble_predictions_matched':len(test),'ensemble_accuracy':accuracy,'advisory_default_api_cases':len(test),'automatic_api_cases':len(test),'diagnostic_rule_traces_recomputed':len(traces)*2,'diagnostic_questions':len(diag),'diagnostic_scenes':len({r['scene_id'] for r in diag}),'human_audit_completed':load_json(ROOT/'results/final_demo/human_audit/summary.json')['completed'],'all_checks_passed':True}
 save_json(ROOT/'results/release_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
