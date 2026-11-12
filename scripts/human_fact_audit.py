"""Export two independent review sheets; summarize only genuine completed reviews."""
import csv,json
from collections import Counter
from nslcot.io import ROOT,load_json,read_jsonl,save_json,digest
DEST=ROOT/'results/final_demo';VALID={'correct','incorrect','unknown_but_visible','unknown_and_unclear','not_applicable'}

def main():
 rows=load_json(DEST/'manifest.json');rows=sorted(rows,key=lambda r:digest(['human-audit-v1',r['id']]))[:50]
 out=DEST/'human_audit';out.mkdir(exist_ok=True)
 columns=['id','question','image_url','object_type','crossing','in_ego_path','obstacle_blocks_path','front_traffic_element','notes']
 for reviewer in ['A','B']:
  path=out/f'reviewer_{reviewer}.csv'
  if path.exists():continue
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
   for r in rows:w.writerow({'id':r['id'],'question':r['question'],'image_url':'http://127.0.0.1:8573/api/targeted-image/'+r['id']})
 sheets=[]
 for reviewer in ['A','B']:
  with (out/f'reviewer_{reviewer}.csv').open() as f:sheet=list(csv.DictReader(f))
  if len({r['id'] for r in sheet})!=len(sheet):raise ValueError('Duplicate review identifiers')
  if {r['id'] for r in sheet}!={r['id'] for r in rows}:raise ValueError('Review identifiers differ from frozen packet')
  for r in sheet:
   if any(r[k] and r[k] not in VALID for k in columns[3:-1]):raise ValueError('Unsupported review category')
  sheets.append({r['id']:r for r in sheet})
 summary={}
 for field in columns[3:-1]:
  paired=[(sheets[0][r['id']][field],sheets[1][r['id']][field]) for r in rows];complete=[(a,b) for a,b in paired if a and b]
  summary[field]={'paired_completed':len(complete),'agreement':sum(a==b for a,b in complete)/len(complete) if complete else None,'reviewer_A':dict(Counter(a for a,b in paired if a)),'reviewer_B':dict(Counter(b for a,b in paired if b))}
 save_json(out/'summary.json',{'requested_examples':50,'completed':all(v['paired_completed']==50 for v in summary.values()),'fields':summary,'notice':'Empty cells are pending human review. No automated or simulated reviewer counts as a human.'})
 (out/'README.txt').write_text('Two group members should review independently before comparing their sheets. For each predicted field, use exactly: correct, incorrect, unknown_but_visible, unknown_and_unclear, not_applicable. View image_url and the corresponding extraction in the demo Targeted diagnostic tab. Do not use dataset labels to infer a visual fact. Preserve IDs. Re-run scripts/human_fact_audit.py to aggregate paired reviews; existing sheets are never overwritten. Agreement concerns review judgments, not driving safety.\n')
 print('Audit packet: 50 examples; human review remains pending unless both sheets are complete.')
if __name__=='__main__':main()
