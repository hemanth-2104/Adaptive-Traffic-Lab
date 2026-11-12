"""Evaluation-only audit of predicted object kinds against held-out annotations.

This module is never imported by the VLM backend, training, or verifier. The
coarse mapping cannot validate path intersection, hazard facts, or faithfulness.
"""
from collections import Counter,defaultdict
from .io import ROOT,load_json,read_jsonl,save_json
from .data import TAG

COARSE={'vehicle':'Vehicle','pedestrian':'Pedestrian','cyclist':'Cycle','traffic_sign':'Traffic element','traffic_light':'Traffic element','barrier':'Traffic element','other':'Others'}


def audit_fact_kinds():
    manifest=load_json(ROOT/'data/processed/manifest.json');raw=load_json(ROOT/'data/raw/v1_0_train_nus.json');facts={r['frame_id']:r['facts'] for r in read_jsonl(ROOT/'results/inference/facts.jsonl')}
    references=defaultdict(set);metadata={}
    for r in manifest:
        if r['split']!='test':continue
        metadata[r['frame_id']]=r
        references[r['frame_id']].update(m[0] for m in TAG.findall(r['question']))
    rows=[];gold_counts=Counter();known=0;correct=0;bycat=defaultdict(lambda:{'n':0,'known':0,'correct':0})
    for fid,ids in sorted(references.items()):
        r=metadata[fid];objects=raw[r['scene_id']]['key_frames'][fid].get('key_object_infos',{});gold={}
        for tag,obj in objects.items():
            matches=TAG.findall(tag)
            if len(matches)==1:gold[matches[0][0]]=obj.get('Category')
        for cid in sorted(ids):
            target=gold.get(cid)
            if target not in set(COARSE.values()):continue
            kind=facts.get(fid,{}).get('objects',{}).get(cid,{}).get('kind','unknown');pred=COARSE.get(kind);is_known=pred is not None;is_correct=pred==target
            gold_counts[target]+=1;known+=is_known;correct+=is_correct;bycat[target]['n']+=1;bycat[target]['known']+=is_known;bycat[target]['correct']+=is_correct
            rows.append({'frame_id':fid,'object_id':cid,'annotation_category':target,'predicted_kind':kind,'mapped_prediction':pred,'known':is_known,'correct':is_correct})
    n=len(rows)
    result={'split':'test','unique_frame_object_pairs':n,'known_kind_predictions':known,'kind_coverage':known/n if n else None,'known_kind_accuracy':correct/known if known else None,'all_kind_accuracy_unknown_as_error':correct/n if n else None,'annotation_majority_accuracy':max(gold_counts.values())/n if n else None,'gold_categories':dict(gold_counts),'by_annotation_category':dict(bycat),'mapping':COARSE,'limitations':['Coarse taxonomy agreement only; this does not validate hazard or path facts.','The annotation Cycle category includes motorcycles, while model kind cyclist is narrower; category-mapping mismatch is a limitation.','Annotations are read only after inference for this diagnostic, never supplied to model inputs or rules.'],'objects':rows}
    save_json(ROOT/'results/evaluation/fact_kind_audit.json',result)
    return result
