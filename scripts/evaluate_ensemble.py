"""Exploratory ensemble; original benchmark remains unchanged."""
import torch
from nslcot.experiments import tensors,load_classifier
from nslcot.io import ROOT,load_json,save_json,digest
from nslcot.metrics import classification_metrics

def main():
 torch.set_num_threads(4)
 rows,x,y,ix,cfg,mean,std=tensors();s=load_json(ROOT/'results/training/selection.json')
 assert s['inputs']['manifest_sha256']==digest(rows)
 probs={}
 with torch.inference_mode():
  for name in ['feedforward','latent_gru']:
   outputs=[]
   for fit in s['selected'][name]['fits']:
    model,_=load_classifier(ROOT/fit['checkpoint']);outputs.append(model(x).softmax(-1))
   probs[name]=torch.stack(outputs).mean(0)
 probs['combined']=(probs['feedforward']+probs['latent_gru'])/2
 val={k:classification_metrics(y[ix['validation']].tolist(),v[ix['validation']].argmax(-1).tolist()) for k,v in probs.items()}
 chosen=sorted(probs,key=lambda k:(-val[k]['macro_f1'],k))[0]
 dest=ROOT/'results/ensemble';dest.mkdir(exist_ok=True)
 save_json(dest/'selection.json',{'status':'exploratory after original benchmark','selected':chosen,'selection':'highest validation macro-F1; alphabetical tie','validation':val,'manifest_sha256':digest(rows)})
 # Persist the choice before computing held-out metrics.
 pred=probs[chosen][ix['test']].argmax(-1).tolist()
 result={'selected':chosen,'test':classification_metrics(y[ix['test']].tolist(),pred),'predictions':[{'id':rows[i]['id'],'prediction':p} for i,p in zip(ix['test'],pred)],'limitation':'Previously examined pilot holdout; fresh scenes needed for confirmation. No verification applied.'}
 save_json(dest/'metrics.json',result);print(chosen,'validation F1',val[chosen]['macro_f1'],'test accuracy',result['test']['accuracy'])
if __name__=='__main__':main()
