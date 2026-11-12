"""Audit original DriveLM records and build a deterministic scene-disjoint manifest.

Ground-truth scene descriptions, object categories, states and other QA answers
never enter the inference manifest. Labels live in a separate evaluation file.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,re,random,math
from .io import ROOT,load_json,save_json,digest,file_hash

YESNO_QUESTION=re.compile(r'^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|had|may|must)\b',re.I)
LABEL=re.compile(r'^(yes|no)(?=$|[\s,.!;:])',re.I)
AMBIGUOUS=re.compile(r'\b(?:maybe|perhaps|not sure|uncertain|depends|cannot determine|can.t tell|if|unless)\b',re.I)
TAG=re.compile(r'<(c\d+),(CAM_[A-Z_]+),([\d.]+),([\d.]+)>')
CAMERAS=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']


def binary_label(question,answer):
    if not isinstance(question,str) or not isinstance(answer,str):return None
    if not YESNO_QUESTION.match(question.strip()) or AMBIGUOUS.search(answer):return None
    m=LABEL.match(answer.strip())
    if not m:return None
    # Reject explicit contradictory yes/no answers, never map 'No entry' to no.
    remainder=answer[m.end():]
    if re.search(r'\b(?:yes|no)(?=[,.!;:]|$)',remainder,re.I):return None
    return int(m.group(1).lower()=='yes')


def question_intent(question):
    q=question.lower()
    if re.search(r"\bnot\b|n.t\b",q):return 'other'
    if re.search(r'\b(?:should|must)\b.*\b(?:stop|wait|yield)\b',q):return 'stop'
    if re.search(r'\b(?:can|could|may|is it safe)\b',q) and re.search(r'\b(?:accelerate|proceed|go straight|move forward|drive forward)\b',q):return 'proceed'
    if re.search(r'\b(?:consider|into account)\b',q):return 'consider'
    if 'traffic sign or a road barrier' in q:return 'traffic_object'
    if 'traffic element in the front view' in q:return 'front_traffic'
    return 'other'


def image_relpath(original):
    path=Path(str(original))
    if 'samples' not in path.parts:raise ValueError(f'Unsupported image path {original!r}')
    start=path.parts.index('samples');return str(Path(*path.parts[start:]))


def inference_input(record):
    allowed=['id','scene_id','frame_id','category','question','images','intent','split']
    return {k:record[k] for k in allowed if k in record}


def prepare(config_path,raw_path=None):
    cfg=load_json(config_path);raw_path=Path(raw_path or ROOT/'data/raw/v1_0_train_nus.json')
    data=load_json(raw_path);excluded=Counter();source_categories=Counter();eligible_classes=Counter();frames=[]
    eligible=0;total_frames=0
    for sid,scene in sorted(data.items()):
        for fid,frame in sorted(scene.get('key_frames',{}).items()):
            total_frames+=1;bycat=defaultdict(list);seen={};conflicted=set()
            try:images={cam:image_relpath(path) for cam,path in frame['image_paths'].items() if cam in CAMERAS}
            except (KeyError,ValueError):excluded['missing_or_invalid_images']+=1;continue
            if len(images)!=6:excluded['incomplete_camera_set']+=1;continue
            for cat,qas in frame.get('QA',{}).items():
                for qa in qas:
                    source_categories[cat]+=1;q=qa.get('Q');a=qa.get('A');y=binary_label(q,a)
                    if y is None:excluded['nonbinary_or_ambiguous']+=1;continue
                    if cat not in cfg['categories']:excluded['category_not_selected']+=1;continue
                    key=re.sub(r'\s+',' ',q.strip()).lower()
                    if key in seen:
                        if seen[key]!=y:conflicted.add(key);excluded['contradictory_duplicate']+=1
                        else:excluded['duplicate_question']+=1
                        continue
                    seen[key]=y
                    row={'id':digest([sid,fid,cat,q])[:24],'scene_id':sid,'frame_id':fid,'category':cat,'question':q.strip(),'images':images,'intent':question_intent(q),'label':y,'dedup_key':key}
                    bycat[cat].append(row)
            candidates=[]
            for cat,rows in bycat.items():
                rows=[r for r in rows if r['dedup_key'] not in conflicted]
                eligible+=len(rows)
                for r in rows:eligible_classes[(cat,'yes' if r['label'] else 'no')]+=1
                # Uses question intent and a fixed hash, never the reference label.
                rows.sort(key=lambda r:(0 if r['intent'] in ['proceed','stop'] else 1,digest([cfg['split_seed'],r['id']])))
                candidates.extend(rows[:cfg['questions_per_category_per_frame']])
            if candidates:frames.append((sid,fid,candidates))
    frames.sort(key=lambda f:digest([cfg['split_seed'],f[0],f[1]]))
    selected=frames[:cfg['max_frames']]
    scenes=sorted({sid for sid,_,_ in selected});random.Random(cfg['split_seed']).shuffle(scenes)
    n=len(scenes)
    if n<6:raise ValueError('At least six distinct scenes are required for a meaningful three-way split.')
    ntrain=max(1,int(n*cfg['split_ratios'][0]));nval=max(1,int(n*cfg['split_ratios'][1]))
    assignments={s:('train' if i<ntrain else 'validation' if i<ntrain+nval else 'test') for i,s in enumerate(scenes)}
    rows=[dict(r,split=assignments[sid]) for sid,fid,rr in selected for r in rr]
    rows.sort(key=lambda r:r['id'])
    inference=[inference_input(r) for r in rows]
    labels={r['id']:r['label'] for r in rows}
    stats={'source':{'scenes':len(data),'frames':total_frames,'qa_by_category':dict(source_categories),'eligible_binary_questions':eligible,'eligible_binary_by_category_and_class':{f'{c}/{y}':v for (c,y),v in sorted(eligible_classes.items())}},'excluded':dict(excluded),'selected':{},'split_seed':cfg['split_seed'],'sampling':'Deterministic hash sample of eligible frames; at most one yes/no question per category per frame; supported stop/proceed intents prioritized within planning independent of answer value.','raw_sha256':file_hash(raw_path),'config_sha256':digest(cfg),'manifest_sha256':digest(inference)}
    for split in ['train','validation','test']:
        rr=[r for r in rows if r['split']==split]
        stats['selected'][split]={'scenes':len({r['scene_id'] for r in rr}),'frames':len({r['frame_id'] for r in rr}),'questions':len(rr),'yes':sum(r['label'] for r in rr),'no':sum(1-r['label'] for r in rr),'categories':dict(Counter(r['category'] for r in rr)),'intents':dict(Counter(r['intent'] for r in rr))}
    assert len({r['id'] for r in rows})==len(rows)
    for split in ['train','validation','test']:
        own={r['scene_id'] for r in rows if r['split']==split};other={r['scene_id'] for r in rows if r['split']!=split};assert own.isdisjoint(other)
    dest=ROOT/'data/processed';save_json(dest/'manifest.json',inference);save_json(dest/'labels.json',labels);save_json(dest/'statistics.json',stats);save_json(dest/'config.json',cfg)
    return stats
