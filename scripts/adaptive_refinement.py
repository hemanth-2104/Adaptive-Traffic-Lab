"""Exploratory early exit from the existing five-step head; no retraining."""
import numpy as np
import torch
from nslcot.experiments import tensors,load_classifier
from nslcot.io import ROOT,load_json,save_json
from nslcot.metrics import classification_metrics

def stopped(prob,threshold):
 pred=[];steps=[]
 for row in prob:
  depth=len(row)
  for t in range(1,len(row)):
   if abs(row[t]-row[t-1])<threshold:depth=t+1;break
  pred.append(int(row[depth-1]>.5));steps.append(depth)
 return pred,steps

def main():
 torch.set_num_threads(4);rows,x,y,ix,cfg,mean,std=tensors();s=load_json(ROOT/'results/training/selection.json');m,_=load_classifier(ROOT/s['selected']['gru_5']['fits'][0]['checkpoint'])
 with torch.inference_mode():
  base=m.projection(x);z=base;states=[]
  for _ in range(5):z=m.refine(base,z);states.append(m.head(z).softmax(-1)[:,1].numpy())
 probs=np.stack(states,axis=1);candidates=[]
 for threshold in [0,.01,.025,.05]:
  p,steps=stopped(probs[ix['validation']],threshold);metric=classification_metrics(y[ix['validation']].tolist(),p)
  candidates.append({'threshold':threshold,'validation_macro_f1':metric['macro_f1'],'validation_mean_steps':float(np.mean(steps))})
 best=min(candidates,key=lambda r:(-r['validation_macro_f1'],r['validation_mean_steps'],r['threshold']))
 dest=ROOT/'results/adaptive';save_json(dest/'selection.json',{'selected':best,'candidates':candidates,'notice':'Exploratory; intermediate heads were not individually supervised. Rule compares consecutive P(yes), minimum two steps, maximum five.'})
 p,steps=stopped(probs[ix['test']],best['threshold']);save_json(dest/'metrics.json',{'test':classification_metrics(y[ix['test']].tolist(),p),'mean_steps':float(np.mean(steps)),'steps_histogram':{str(k):steps.count(k) for k in sorted(set(steps))},'notice':'Previously examined pilot holdout; no end-to-end latency improvement claimed. VLM encoding remains the dominant cost.'})
 print(best)
if __name__=='__main__':main()
