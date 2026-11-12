"""Cached-feature classifiers and the validation-selected three-seed ensemble."""
from functools import lru_cache
import numpy as np
import torch
from .io import ROOT, load_json
from .experiments import load_classifier

@lru_cache(maxsize=24)
def loaded(path):
    return load_classifier(path)

def standardized(feature, checkpoint):
    feature=np.asarray(feature,dtype=np.float32)
    mean=np.asarray(checkpoint['mean'],dtype=np.float32)
    std=np.asarray(checkpoint['std'],dtype=np.float32)
    if feature.shape!=mean.shape or not np.isfinite(feature).all() or np.any(std<=0):
        raise ValueError('Invalid feature vector or checkpoint normalization')
    return torch.from_numpy((feature-mean)/std).unsqueeze(0)

def ensemble_probabilities(feature, root=None):
    root=root or ROOT
    policy=load_json(root/'results/ensemble/selection.json')
    selection=load_json(root/'results/training/selection.json')
    chosen=policy['selected']
    names=['feedforward','latent_gru'] if chosen=='combined' else [chosen]
    if any(name not in ['feedforward','latent_gru'] for name in names):
        raise ValueError('Unsupported validation-selected ensemble')
    families=[]
    with torch.inference_mode():
        for name in names:
            members=[]
            for fit in selection['selected'][name]['fits']:
                model,ckpt=loaded(str(root/fit['checkpoint']))
                members.append(model(standardized(feature,ckpt)).softmax(-1)[0])
            if len(members)!=3:raise ValueError('Expected the three recorded training seeds')
            families.append(torch.stack(members).mean(0))
    return torch.stack(families).mean(0).tolist()
