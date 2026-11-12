"""Additional diagnostic evidence; never changes original benchmark outputs."""
from collections import Counter
import numpy as np
import torch
from nslcot.io import ROOT,load_json,save_json,read_jsonl
from nslcot.experiments import load_classifier
from nslcot.metrics import classification_metrics,paired_scene_bootstrap
from nslcot.targeted import DEST,FIELDS

def main():
 rows=load_json(DEST/'manifest.json');outputs={r['id']:r for r in read_jsonl(DEST/'outputs.jsonl')}
 if len(outputs)!=len(rows):raise RuntimeError('Wait for all frozen diagnostic examples to finish before scoring')
 traces=load_json(DEST/'traces.json');y=[t['reference'] for t in traces];pred=[t['neural'] for t in traces]
 s=load_json(ROOT/'results/training/selection.json');torch.set_num_threads(4)
 features=np.stack([np.load(DEST/'features'/(r['id']+'.npy')) for r in rows]);controls={}
 for name in ['feedforward','matched_feedforward','gru_1','latent_gru','gru_5']:
  fits=[]
  for fit in s['selected'][name]['fits']:
   model,c=load_classifier(ROOT/fit['checkpoint']);x=torch.tensor((features-np.array(c['mean'],dtype=np.float32))/np.array(c['std'],dtype=np.float32))
   with torch.inference_mode():p=model(x).argmax(-1).tolist()
   fits.append(classification_metrics(y,p))
  controls[name]={'seeds':fits,'mean_accuracy':float(np.mean([m['accuracy'] for m in fits]))}
 from sklearn.pipeline import make_pipeline
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.linear_model import LogisticRegression
 from nslcot.data import TAG
 pilot=load_json(ROOT/'data/processed/manifest.json');pilot_y=load_json(ROOT/'data/processed/labels.json');train=[r for r in pilot if r['split']=='train'];c=load_json(ROOT/'results/evaluation/metrics.json')['methods']['question_only']['selected_C'];cfg=load_json(ROOT/'configs/pilot.json')
 normalize=lambda r:TAG.sub(lambda m:'object in '+m.group(2).lower(),r['question'])
 text_model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True),LogisticRegression(C=c,max_iter=1000,random_state=cfg['split_seed']))
 text_model.fit([normalize(r) for r in train],[pilot_y[r['id']] for r in train]);controls['question_only']=classification_metrics(y,text_model.predict([normalize(r) for r in rows]).tolist())
 majority=int(sum(pilot_y[r['id']] for r in train)>=len(train)/2);controls['training_majority']=classification_metrics(y,[majority]*len(rows))
 metrics=load_json(DEST/'metrics.json');metrics['controls']=controls;metrics['seed']=18;metrics['fact_status_counts']={f:dict(Counter(outputs[r['id']]['targeted']['fields'].get(f,{}).get('status','schema_invalid') for r in rows)) for f in FIELDS}
 metrics['condition_labels']={'original':'Constrained full context','targeted':'Constrained context plus query-local detail'};metrics['scenes']=len({r['scene_id'] for r in rows});metrics['intent_counts']=dict(Counter(r['intent'] for r in rows))
 for method in ['original','targeted']:
  v=[t[method]['final'] for t in traces];metrics[method]['paired_interval']=paired_scene_bootstrap(rows,y,pred,v);metrics[method]['net_accuracy_change']=metrics[method]['accuracy']-metrics['neural']['accuracy'];metrics[method]['activation_rate']=metrics[method]['covered']/len(rows)
  metrics[method]['rule_counts']=dict(Counter(c['rule_id'] for t in traces for c in t[method]['checks']))
 metrics['notice']='Wording-selected diagnostic on scenes excluded from the original pilot, not a representative benchmark or independent human fact audit. Both conditions use the same constrained categorical extractor and fact-pass pixel budget; treatment adds a question-local detail panel and redistributes pixels. The generative JSON smoke attempt was stopped for schema failure before diagnostic scoring.'
 save_json(DEST/'metrics.json',metrics);print('Saved diagnostic controls, known rates, correction intervals.')
if __name__=='__main__':main()
