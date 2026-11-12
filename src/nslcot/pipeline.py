from pathlib import Path
from collections import defaultdict
import json,time,os,zipfile
import numpy as np
from .io import ROOT,load_json,save_json,read_jsonl,append_jsonl,digest
from .backend import QwenBackend
from .images import compose_view


def extract_images(archive=None):
    archive=archive or ROOT/'data/raw/drivelm_images.zip';dest=ROOT/'data/images';dest.mkdir(parents=True,exist_ok=True)
    manifest=load_json(ROOT/'data/processed/manifest.json');needed={rel for r in manifest for rel in r['images'].values()};found=set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts=Path(info.filename).parts
            if 'samples' not in parts:continue
            rel=str(Path(*parts[parts.index('samples'):]))
            if rel not in needed:continue
            path=dest/rel
            if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('Archive member escaped image root')
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():path.write_bytes(z.read(info))
            found.add(rel)
    missing=needed-found
    if missing:raise FileNotFoundError(f'{len(missing)} selected images absent from archive; first: {sorted(missing)[:3]}')
    save_json(ROOT/'data/processed/image_validation.json',{'required':len(needed),'found':len(found),'missing':0})
    return len(found)


def run_inference(config_path,phase='all',limit=None):
    cfg=load_json(config_path);manifest=load_json(ROOT/'data/processed/manifest.json');stats=load_json(ROOT/'data/processed/statistics.json')
    if digest(manifest)!=stats['manifest_sha256']:raise ValueError('Manifest changed after preparation')
    if digest(cfg)!=stats['config_sha256']:raise ValueError('Config changed after manifest creation')
    out=ROOT/'results/inference';out.mkdir(parents=True,exist_ok=True)
    provenance={'manifest_sha256':digest(manifest),'config_sha256':digest(cfg)}
    if (out/'provenance.json').exists() and load_json(out/'provenance.json')!=provenance:raise ValueError('Existing inference cache belongs to another experiment')
    save_json(out/'provenance.json',provenance)
    backend=QwenBackend(cfg);print('MODEL READY',backend.device,flush=True)
    byframe=defaultdict(list)
    for r in manifest:byframe[r['frame_id']].append(r)
    image_root=ROOT/'data/images'
    # Warmup is explicit and excluded from aggregate efficiency statistics.
    first=manifest[0];im=compose_view(first['images'],image_root,[rr['question'] for rr in byframe[first['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
    _,warm=backend.feature(im,first['question']);save_json(out/'warmup.json',warm)
    if phase in ['all','features']:
        done={p.stem for p in (out/'features').glob('*.npy')} if (out/'features').exists() else set();(out/'features').mkdir(exist_ok=True)
        tasks=[r for r in manifest if r['id'] not in done]
        for i,r in enumerate(tasks[:limit] if limit else tasks):
            im=compose_view(r['images'],image_root,[rr['question'] for rr in byframe[r['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
            h,timing=backend.feature(im,r['question']);tmp=out/'features'/(r['id']+'.tmp.npy');np.save(tmp,h);os.replace(tmp,out/'features'/(r['id']+'.npy'))
            append_jsonl(out/'feature_timing.jsonl',dict(id=r['id'],**timing))
            if i%10==0:print('FEATURE',i+1,'/',len(tasks),'seconds',round(timing['total_seconds'],2),flush=True)
    if phase in ['all','facts']:
        done={r['frame_id'] for r in read_jsonl(out/'facts.jsonl')};tasks=[(fid,rr) for fid,rr in sorted(byframe.items()) if fid not in done]
        for i,(fid,rr) in enumerate(tasks[:limit] if limit else tasks):
            im=compose_view(rr[0]['images'],image_root,[r['question'] for r in rr],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
            facts,raw,timing=backend.facts(im,[r['question'] for r in rr]);append_jsonl(out/'facts.jsonl',{'frame_id':fid,'facts':facts,'raw_response':raw,'timing':timing})
            if i%5==0:print('FACTS',i+1,'/',len(tasks),'valid',facts['valid'],'seconds',round(timing['total_seconds'],2),flush=True)
    if phase in ['all','baselines']:
        facts={r['frame_id']:r['facts'] for r in read_jsonl(out/'facts.jsonl')}
        for mode in ['direct','cot','graph']:
            done={r['id'] for r in read_jsonl(out/(mode+'.jsonl'))};tasks=[r for r in manifest if r['split']=='test' and r['id'] not in done]
            for i,r in enumerate(tasks[:limit] if limit else tasks):
                if mode=='graph' and r['frame_id'] not in facts:raise ValueError('Run fact extraction before graph-context baseline')
                im=compose_view(r['images'],image_root,[rr['question'] for rr in byframe[r['frame_id']]],all_cameras=True,width=cfg['view_width'],height=cfg['view_height'])
                pred,raw,timing=backend.answer(im,r['question'],mode,facts.get(r['frame_id']))
                append_jsonl(out/(mode+'.jsonl'),{'id':r['id'],'prediction':pred,'raw_response':raw,'timing':timing})
                if i%5==0:print(mode.upper(),i+1,'/',len(tasks),'seconds',round(timing['total_seconds'],2),flush=True)
    print('INFERENCE PHASE COMPLETE',phase,flush=True)
