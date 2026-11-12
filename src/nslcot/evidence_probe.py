"""Fixed image-permutation diagnostic; not a new labeled benchmark.

Inference never reads labels. Scoring uses the original reference answers only
as a sensitivity diagnostic: a shuffled image does not inherit those labels as
new ground truth. We make no causal-faithfulness claim from this intervention.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,load_json,save_json,append_jsonl,digest
from .images import compose_view
from .backend import QwenBackend
from .experiments import load_classifier
from .metrics import classification_metrics,paired_scene_bootstrap


def run_probe():
    cfg=load_json(ROOT/'configs/pilot.json');all_rows=load_json(ROOT/'data/processed/manifest.json');rows=[r for r in all_rows if r['split']=='test'];byframe=defaultdict(list)
    for r in rows:byframe[r['frame_id']].append(r)
    frames=sorted(byframe,key=lambda fid:digest([572,'image-permutation',fid]))
    mapping=None
    for offset in range(1,len(frames)):
        trial={fid:frames[(i+offset)%len(frames)] for i,fid in enumerate(frames)}
        if all(byframe[a][0]['scene_id']!=byframe[b][0]['scene_id'] for a,b in trial.items()):mapping=trial;break
    if mapping is None:raise ValueError('No different-scene permutation found')
    out=ROOT/'results/evidence_probe';(out/'features').mkdir(parents=True,exist_ok=True)
    provenance={'manifest_sha256':digest(all_rows),'config_sha256':digest(cfg),'donor_mapping':mapping,'intervention':'Replace all six camera photographs with another held-out scene; keep recipient questions and query-derived marker coordinates. Fixed cyclic derangement, no labels used.'}
    if (out/'provenance.json').exists() and load_json(out/'provenance.json')!=provenance:raise ValueError('Probe cache provenance mismatch')
    save_json(out/'provenance.json',provenance)
    backend=QwenBackend(cfg)
    first=rows[0];donor=byframe[mapping[first['frame_id']]][0]
    im=compose_view(donor['images'],ROOT/'data/images',[r['question'] for r in byframe[first['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height']);backend.feature(im,first['question'])
    for i,r in enumerate(rows):
        path=out/'features'/(r['id']+'.npy')
        if path.exists():continue
        donor=byframe[mapping[r['frame_id']]][0]
        im=compose_view(donor['images'],ROOT/'data/images',[q['question'] for q in byframe[r['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
        h,t=backend.feature(im,r['question']);tmp=path.with_suffix('.tmp.npy');np.save(tmp,h);tmp.replace(path)
        append_jsonl(out/'timing.jsonl',dict(id=r['id'],donor_frame_id=donor['frame_id'],**t))
        if i%10==0:print('IMAGE PERMUTATION',i+1,'/',len(rows),flush=True)
    del backend
    if torch.backends.mps.is_available():torch.mps.empty_cache()
    # Evaluation is separate from the label-free inference loop above.
    labels=load_json(ROOT/'data/processed/labels.json');selection=load_json(ROOT/'results/training/selection.json');original=load_json(ROOT/'results/evaluation/predictions.json');features=np.stack([np.load(out/'features'/(r['id']+'.npy'),allow_pickle=False) for r in rows]);truth=[labels[r['id']] for r in rows];results={};predictions={}
    torch.set_num_threads(4)
    for name in ['feedforward','matched_feedforward','latent_gru']:
        mm=[]
        for seed,fit in zip(cfg['train_seeds'],selection['selected'][name]['fits']):
            model,ckpt=load_classifier(ROOT/fit['checkpoint']);x=torch.tensor((features-np.asarray(ckpt['mean'],dtype=np.float32))/np.asarray(ckpt['std'],dtype=np.float32))
            with torch.inference_mode():pred=model(x).argmax(-1).tolist()
            old={r['id']:r['prediction'] for r in original[f'{name}/seed{seed}']};before=[old[r['id']] for r in rows];m=classification_metrics(truth,pred);m['prediction_agreement_with_original']=float(np.mean([a==b for a,b in zip(before,pred)]));m['original_accuracy']=classification_metrics(truth,before)['accuracy'];m['accuracy_change']=m['accuracy']-m['original_accuracy'];m['paired_interval']=paired_scene_bootstrap(rows,truth,before,pred);results[f'{name}/seed{seed}']=m;predictions[f'{name}/seed{seed}']=[{'id':r['id'],'prediction':p} for r,p in zip(rows,pred)];mm.append(m)
        results[name+'_mean']={k:float(np.mean([m[k] for m in mm])) for k in ['accuracy','macro_f1','prediction_agreement_with_original','original_accuracy','accuracy_change']}
    save_json(out/'metrics.json',{'conditions':results,'n':len(rows),'notice':'Scores are agreement with ORIGINAL reference answers after images are deliberately mismatched. They measure sensitivity and shortcuts, not accuracy on newly labeled scenes or faithful internal reasoning.'});save_json(out/'predictions.json',predictions)
    print('EVIDENCE PROBE COMPLETE',results['latent_gru_mean'],flush=True)

if __name__=='__main__':run_probe()
