from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
from .rules import verify


def classification_metrics(labels,predictions):
    y=np.asarray(labels,dtype=int);pred=np.asarray([-1 if p is None else int(p) for p in predictions])
    if len(y)!=len(pred) or len(y)==0:raise ValueError('Nonempty aligned labels/predictions required')
    if not set(y)<= {0,1}:raise ValueError('Labels must be binary')
    return {'n':len(y),'accuracy':float(np.mean(y==pred)),'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),'invalid_rate':float(np.mean(pred==-1)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,-1]).tolist(),'confusion_matrix_order':['no','yes','invalid']}


def verification_metrics(records,predictions,labels,facts_by_frame):
    traces=[];covered=0;total_checks=0;sat_before=0;sat_after=0;changes=Counter();unresolved=0
    for row,pred,y in zip(records,predictions,labels):
        facts=facts_by_frame.get(row['frame_id'],{'objects':{}});v=verify(row['question'],pred,facts)
        covered+=bool(v['checks']);total_checks+=len(v['checks']);sat_before+=len(v['checks'])-len(v['violations_before']);sat_after+=len(v['checks'])-len(v['violations_after'])
        if v['status'].startswith('unresolved'):unresolved+=1
        if v['changed']:changes['incorrect_to_correct' if pred!=y and v['final']==y else 'correct_to_incorrect' if pred==y and v['final']!=y else 'incorrect_to_incorrect']+=1
        else:changes['unchanged']+=1
        traces.append(dict(v,id=row['id']))
    n=len(records)
    return {'coverage':covered/n,'covered_questions':covered,'active_checks':total_checks,'rule_satisfaction_before':sat_before/total_checks if total_checks else None,'rule_satisfaction_after':sat_after/total_checks if total_checks else None,'unresolved':unresolved,'changes':dict(changes),'net_accuracy_change':(changes['incorrect_to_correct']-changes['correct_to_incorrect'])/n},traces


def scene_bootstrap(records,labels,predictions,iterations=1000,seed=572):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['scene_id']].append(i)
    scenes=sorted(groups);rng=np.random.default_rng(seed);scores=[]
    for _ in range(iterations):
        selected=rng.choice(scenes,len(scenes),replace=True);idx=[i for s in selected for i in groups[s]]
        scores.append(np.mean([labels[i]==predictions[i] for i in idx]))
    return {'accuracy_ci95':[float(v) for v in np.quantile(scores,[.025,.975])],'unit':'scene','iterations':iterations}


def paired_scene_bootstrap(records,labels,before,after,iterations=1000,seed=572):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['scene_id']].append(i)
    scenes=sorted(groups);rng=np.random.default_rng(seed);deltas=[]
    for _ in range(iterations):
        ix=[i for s in rng.choice(scenes,len(scenes),replace=True) for i in groups[s]]
        deltas.append(np.mean([(after[i]==labels[i])-(before[i]==labels[i]) for i in ix]))
    return {'accuracy_delta_ci95':[float(v) for v in np.quantile(deltas,[.025,.975])],'unit':'scene','iterations':iterations}
