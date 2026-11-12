"""Audit selected image duplicates and query/template distributions."""
import sys,json,hashlib,re
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nslcot.io import load_json,save_json,file_hash
from nslcot.data import TAG
rows=load_json(ROOT/'data/processed/manifest.json');images=defaultdict(set);file_splits=defaultdict(set)
for r in rows:
 for rel in r['images'].values():file_splits[rel].add(r['split'])
for rel,splits in file_splits.items():images[file_hash(ROOT/'data/images'/rel)].update(splits)
leaks=[h for h,s in images.items() if len(s)>1]
templates=defaultdict(Counter)
for r in rows:templates[r['split']][TAG.sub('<object>',r['question'])]+=1
report={'selected_image_files':len(file_splits),'unique_image_content_hashes':len(images),'cross_split_identical_image_hashes':leaks,'scene_overlap':{},'template_counts':{s:dict(c) for s,c in templates.items()},'train_test_template_overlap':len(set(templates['train'])&set(templates['test'])),'notes':'Template overlap is measured and expected; question-only baseline diagnoses language shortcuts. Exact image-content overlap across partitions is disallowed.'}
for a,b in [('train','validation'),('train','test'),('validation','test')]:report['scene_overlap'][a+'/'+b]=sorted({r['scene_id'] for r in rows if r['split']==a}&{r['scene_id'] for r in rows if r['split']==b})
save_json(ROOT/'data/processed/leakage_audit.json',report)
assert not leaks,'Identical image content crosses a partition boundary'
assert not any(report['scene_overlap'].values()),'Scene overlap'
print('Leakage audit passed:',len(file_splits),'images;',len(images),'unique contents; 0 scene overlaps; 0 cross-split identical images.')
