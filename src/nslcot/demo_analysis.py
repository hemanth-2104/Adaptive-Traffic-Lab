"""Inspectable latent states. Intermediate outputs are diagnostics, not rationales."""
import numpy as np
import torch
from .io import ROOT,load_json
from .experiments import load_classifier

def probability_evolution(model,x):
    with torch.inference_mode():
        z=model.projection(x);out=[]
        if model.kind=='gru':
            for step in range(1,model.steps+1):
                z=model.refine(model.projection(x),z)
                out.append({'step':step,'yes':float(model.head(z).softmax(-1)[0,1])})
        else:out=[{'step':1,'yes':float(model(x).softmax(-1)[0,1])}]
    return out

def compare(record_id):
    h=np.load(ROOT/'results/inference/features'/(record_id+'.npy'),allow_pickle=False)
    selection=load_json(ROOT/'results/training/selection.json');out={}
    for name in ['feedforward','latent_gru','matched_feedforward','gru_1','gru_5']:
        model,ckpt=load_classifier(ROOT/selection['selected'][name]['fits'][0]['checkpoint'])
        x=torch.tensor((h-np.array(ckpt['mean'],dtype=np.float32))/np.array(ckpt['std'],dtype=np.float32)).unsqueeze(0)
        evolution=probability_evolution(model,x)
        out[name]={'yes':evolution[-1]['yes'],'prediction':int(evolution[-1]['yes']>.5),'evolution':evolution}
    model,ckpt=load_classifier(ROOT/selection['selected']['gru_5']['fits'][0]['checkpoint'])
    x=torch.tensor((h-np.array(ckpt['mean'],dtype=np.float32))/np.array(ckpt['std'],dtype=np.float32)).unsqueeze(0)
    threshold=load_json(ROOT/'results/adaptive/selection.json')['selected']['threshold']
    evolution=adaptive_evolution(model,x,threshold)
    out['adaptive_gru']={'yes':evolution[-1]['yes'],'prediction':int(evolution[-1]['yes']>.5),'evolution':evolution,'threshold':threshold,'notice':'Exploratory validation-selected early exit; intermediate readouts are not individually supervised.'}
    path=ROOT/'results/evidence_probe/features'/(record_id+'.npy')
    if path.exists():
        model,ckpt=load_classifier(ROOT/selection['selected']['latent_gru']['fits'][0]['checkpoint'])
        x=torch.tensor((np.load(path)-np.array(ckpt['mean'],dtype=np.float32))/np.array(ckpt['std'],dtype=np.float32)).unsqueeze(0)
        out['shuffled_gru']={'yes':probability_evolution(model,x)[-1]['yes'],'notice':'Photographs replaced by another scene. Original reference is not a label for this intervention.'}
    from .prediction import ensemble_probabilities
    prob=ensemble_probabilities(h,ROOT)
    out['ensemble']={'yes':prob[1],'prediction':int(prob[1]>prob[0]),'evolution':[],'notice':'Mean probabilities across validation-selected seed ensemble; exploratory on the original holdout.'}
    return out

def adaptive_evolution(model,x,threshold):
    """Actually stop computation, minimum two updates; one example at a time."""
    if model.kind!='gru' or x.shape[0]!=1:raise ValueError('Adaptive demo expects one GRU example')
    out=[]
    with torch.inference_mode():
        base=model.projection(x);z=base
        for step in range(1,model.steps+1):
            z=model.refine(base,z);yes=float(model.head(z).softmax(-1)[0,1]);out.append({'step':step,'yes':yes})
            if len(out)>=2 and abs(out[-1]['yes']-out[-2]['yes'])<threshold:break
    return out
