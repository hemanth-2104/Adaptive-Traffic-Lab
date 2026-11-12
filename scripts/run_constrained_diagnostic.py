import numpy as np
from nslcot.io import ROOT,load_json,save_json,read_jsonl,append_jsonl,digest,file_hash
from nslcot.targeted import DEST,prepare,relevant_view,legacy,score
from nslcot.backend import QwenBackend
from nslcot.images import compose_view
from nslcot.constrained_facts import extract

def main():
 rows=prepare();cfg={**load_json(ROOT/'configs/pilot.json'),'max_pixels':401408}
 provenance={'manifest_sha256':digest(rows),'config_sha256':digest(cfg),'extractor_sha256':file_hash(ROOT/'src/nslcot/constrained_facts.py'),'control':'same constrained schema and question on full context; treatment adds query-local detail; same fact-pass pixel budget'}
 if (DEST/'constrained_provenance.json').exists() and load_json(DEST/'constrained_provenance.json')!=provenance:raise ValueError('Diagnostic cache provenance mismatch')
 save_json(DEST/'constrained_provenance.json',provenance)
 pilot_runtime=(ROOT/'results/runtime.json').read_bytes();b=QwenBackend(cfg);save_json(DEST/'runtime.json',b.provenance);(ROOT/'results/runtime.json').write_bytes(pilot_runtime)
 done={r['id'] for r in read_jsonl(DEST/'outputs.jsonl')}
 for i,r in enumerate(rows):
  if r['id'] in done:continue
  context=compose_view(r['images'],ROOT/'data/images',[r['question']],all_cameras=True)
  path=DEST/'features'/(r['id']+'.npy');path.parent.mkdir(exist_ok=True)
  ft={'reused_label_free_feature':True}
  if not path.exists():
   old=b.processor.image_processor.max_pixels;b.processor.image_processor.max_pixels=200704
   h,ft=b.feature(context,r['question']);np.save(path,h);b.processor.image_processor.max_pixels=old
  panel,region=relevant_view(r)
  full,fullraw,fullt=extract(b,context,r);target,raw,t=extract(b,panel,r)
  append_jsonl(DEST/'outputs.jsonl',{'id':r['id'],'targeted':target,'facts':legacy(r,target),'raw':raw,'original_facts':legacy(r,full),'original_raw':fullraw,'original_extraction':full,'region':region,'timing':t,'original_timing':fullt,'feature_timing':ft,'method':'constrained categorical selection; full context versus context plus local detail'})
  print('CONSTRAINED',i+1,'/',len(rows),flush=True)
 score()
if __name__=='__main__':main()
