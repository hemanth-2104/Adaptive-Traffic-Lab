"""Validation-only model selection, three-seed fits, and paired held-out evaluation."""
from pathlib import Path
from collections import defaultdict
import copy,time,json,random,platform,inspect
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .io import ROOT,load_json,save_json,read_jsonl,file_hash,digest
from .models import FeatureClassifier
from .data import TAG
from .metrics import classification_metrics,verification_metrics,scene_bootstrap,paired_scene_bootstrap

SPECS=[('feedforward','mlp',1),('matched_feedforward','matched_mlp',1),('gru_1','gru',1),('latent_gru','gru',3),('gru_5','gru',5)]


def tensors():
    rows=load_json(ROOT/'data/processed/manifest.json');labels=load_json(ROOT/'data/processed/labels.json');cfg=load_json(ROOT/'data/processed/config.json')
    cache=ROOT/'results/inference/features';missing=[r['id'] for r in rows if not (cache/(r['id']+'.npy')).exists()]
    if missing:raise FileNotFoundError(f'{len(missing)} features are still missing')
    x=np.stack([np.load(cache/(r['id']+'.npy'),allow_pickle=False) for r in rows]);y=np.array([labels[r['id']] for r in rows],dtype=np.int64)
    if not np.isfinite(x).all():raise ValueError('Non-finite feature cache')
    ix={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','validation','test']}
    mean=x[ix['train']].mean(0);std=x[ix['train']].std(0).clip(min=1e-4)
    return rows,torch.tensor((x-mean)/std),torch.tensor(y),ix,cfg,mean,std


def fit_one(x,y,train_ix,val_ix,cfg,kind,steps,lr,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=FeatureClassifier(x.shape[1],cfg['hidden_dim'],kind,steps)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
    criterion=torch.nn.CrossEntropyLoss();best_score=-1.;best_state=None;best_epoch=0;history=[];start=time.perf_counter()
    gen=torch.Generator().manual_seed(seed)
    for epoch in range(cfg['max_epochs']):
        model.train();order=torch.as_tensor(train_ix)[torch.randperm(len(train_ix),generator=gen)];losses=[]
        for batch in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True);loss=criterion(model(x[batch]),y[batch]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():pred=model(x[val_ix]).argmax(-1).tolist()
        m=classification_metrics(y[val_ix].tolist(),pred);score=m['macro_f1'];history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'validation_accuracy':m['accuracy'],'validation_macro_f1':score})
        if score>best_score+1e-9:best_score=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=cfg['patience']:break
    model.load_state_dict(best_state);model.eval()
    return model,{'seed':seed,'lr':lr,'best_epoch':best_epoch,'validation_macro_f1':best_score,'training_seconds':time.perf_counter()-start,'parameters':model.trainable_parameters,'history':history}


def train_models():
    torch.set_num_threads(4)
    rows,x,y,ix,cfg,mean,std=tensors();dest=ROOT/'results/training';dest.mkdir(parents=True,exist_ok=True)
    inputs={'manifest_sha256':digest(rows),'config_sha256':digest(cfg),'model_code_sha256':file_hash(ROOT/'src/nslcot/models.py'),'training_code_sha256':file_hash(Path(__file__)),'training_kernel_sha256':digest({'fit_one':inspect.getsource(fit_one),'specs':SPECS,'selection_policy':'mean validation macro-F1; lower lr tie'})}
    if (dest/'selection.json').exists():
        old=load_json(dest/'selection.json')
        if {k:v for k,v in old['inputs'].items() if k!='training_code_sha256'}!={k:v for k,v in inputs.items() if k!='training_code_sha256'}:raise ValueError('Training inputs or fit kernel changed; use a separate experiment directory')
        print('Training already complete',flush=True);return old
    candidates=defaultdict(dict)
    for name,kind,steps in SPECS:
        for lr in cfg['learning_rates']:
            entries=[]
            for seed in cfg['train_seeds']:
                key=f'{name}_lr{lr}_seed{seed}';metapath=dest/(key+'.json');ckptpath=dest/(key+'.pt')
                if metapath.exists() and ckptpath.exists():meta=load_json(metapath)
                else:
                    model,meta=fit_one(x,y,ix['train'],ix['validation'],cfg,kind,steps,lr,seed)
                    checkpoint={'state_dict':model.state_dict(),'kind':kind,'steps':steps,'input_dim':x.shape[1],'hidden_dim':cfg['hidden_dim'],'mean':mean.tolist(),'std':std.tolist(),'inputs':inputs}
                    torch.save(checkpoint,ckptpath);save_json(metapath,meta)
                print('FIT',key,'val macro-F1',round(meta['validation_macro_f1'],4),'epoch',meta['best_epoch'],flush=True)
                entries.append({'checkpoint':str(ckptpath.relative_to(ROOT)),'metadata':str(metapath.relative_to(ROOT)),'validation_macro_f1':meta['validation_macro_f1'],'parameters':meta['parameters']})
            candidates[name][str(lr)]={'mean_validation_macro_f1':float(np.mean([e['validation_macro_f1'] for e in entries])),'fits':entries}
    selected={}
    for name,options in candidates.items():
        lr=min(options,key=lambda k:(-options[k]['mean_validation_macro_f1'],float(k)))
        selected[name]={'learning_rate':float(lr),**options[lr]}
    selection={'inputs':inputs,'selection_policy':'Maximize mean validation macro-F1 across the three seeds; tie goes to smaller learning rate. Checkpoint within each fit maximizes validation macro-F1, with early stopping. No test metrics used.','candidates':dict(candidates),'selected':selected,'normalization':'Per-dimension training-only mean and standard deviation, std floor 1e-4.'}
    save_json(dest/'selection.json',selection);print('MODEL SELECTION FROZEN',flush=True)
    return selection


def load_classifier(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    model=FeatureClassifier(ckpt['input_dim'],ckpt['hidden_dim'],ckpt['kind'],ckpt['steps']);model.load_state_dict(ckpt['state_dict']);model.eval()
    return model,ckpt


def classifier_timing(model,sample,repeats=50):
    with torch.inference_mode():
        for _ in range(5):model(sample)
        times=[]
        for _ in range(repeats):
            start=time.perf_counter();model(sample);times.append(time.perf_counter()-start)
    return {'cached_head_mean_seconds':float(np.mean(times)),'cached_head_p95_seconds':float(np.quantile(times,.95)),'repeats':repeats,'warmup_calls':5,'device':'cpu','batch_size':1}


def evaluate():
    torch.set_num_threads(4);rows,x,y,ix,cfg,mean,std=tensors();selection=load_json(ROOT/'results/training/selection.json')
    if selection['inputs']['manifest_sha256']!=digest(rows):raise ValueError('Training manifest mismatch')
    test=[rows[i] for i in ix['test']];truth=y[ix['test']].tolist();inf=ROOT/'results/inference';dest=ROOT/'results/evaluation';dest.mkdir(parents=True,exist_ok=True)
    fact_rows=read_jsonl(inf/'facts.jsonl');facts={r['frame_id']:r['facts'] for r in fact_rows}
    if not all(r['frame_id'] in facts for r in test):raise ValueError('Missing test scene facts')
    methods={};row_outputs={};traces_out={}
    from .images import compose_view
    from .rules import verify
    # Warm local image preparation and graph/check costs are measured separately.
    byframe=defaultdict(list)
    for r in rows:byframe[r['frame_id']].append(r)
    prepare_times={};verify_times={}
    for fid in sorted({r['frame_id'] for r in test}):
        frame_rows=byframe[fid];tt=[]
        for _ in range(3):
            start=time.perf_counter();compose_view(frame_rows[0]['images'],ROOT/'data/images',[r['question'] for r in frame_rows],all_cameras=True,width=cfg['view_width'],height=cfg['view_height']);tt.append(time.perf_counter()-start)
        prepare_times[fid]=float(np.mean(tt))
    for r in test:
        start=time.perf_counter()
        for _ in range(20):verify(r['question'],0,facts[r['frame_id']])
        verify_times[r['id']]=(time.perf_counter()-start)/20
    save_json(dest/'pipeline_timing.json',{'image_preparation_by_frame':prepare_times,'graph_and_rule_by_question':verify_times,'definition':'Warm local image reads/composition averaged over three calls. Graph construction and checking averaged over 20 calls. One shared composite per query. Model loading and dataset download excluded.'})
    feature_times={r['id']:r['total_seconds'] for r in read_jsonl(inf/'feature_timing.jsonl')};fact_times={r['frame_id']:r['timing']['total_seconds'] for r in fact_rows}
    def add(name,pred,seed=None,extra=None,verify_variant=False):
        key=name if seed is None else f'{name}/seed{seed}'
        m=classification_metrics(truth,pred);m.update(scene_bootstrap(test,truth,pred));m.update(extra or {})
        vm,traces=verification_metrics(test,pred,truth,facts)
        m['verification']=vm;m['by_category']={cat:classification_metrics([truth[i] for i,r in enumerate(test) if r['category']==cat],[pred[i] for i,r in enumerate(test) if r['category']==cat]) for cat in cfg['categories']}
        methods[key]=m;row_outputs[key]=[{'id':r['id'],'scene_id':r['scene_id'],'category':r['category'],'label':t,'prediction':p} for r,t,p in zip(test,truth,pred)];traces_out[key]=traces
        if verify_variant:
            corrected=[v['final'] for v in traces];add(name+'_verified',corrected,seed,{**(extra or {}),'correction':vm,'paired_accuracy_change':paired_scene_bootstrap(test,truth,pred,corrected)},False)
        return m
    # Prompting baselines are evaluated only after all fixed test outputs exist.
    for mode in ['direct','cot','graph']+(['cot_two_pass'] if (inf/'cot_two_pass.jsonl').exists() else []):
        rr={r['id']:r for r in read_jsonl(inf/(mode+'.jsonl'))}
        if not all(r['id'] in rr for r in test):raise ValueError(f'Incomplete {mode} baseline')
        pred=[rr[r['id']]['prediction'] for r in test];tt=[rr[r['id']]['timing']['total_seconds']+prepare_times[r['frame_id']]+(fact_times[r['frame_id']] if mode=='graph' else 0) for r in test]
        add(mode,pred,extra={'end_to_end_mean_seconds':float(np.mean(tt)),'end_to_end_p95_seconds':float(np.quantile(tt,.95))},verify_variant=False)
    # Baselines that expose dataset and language-template shortcuts.
    majority=int(y[ix['train']].sum()>=len(ix['train'])/2);add('majority',[majority]*len(test))
    def normalized_question(r):return TAG.sub(lambda m:'object in '+m.group(2).lower(),r['question'])
    qtrain=[normalized_question(rows[i]) for i in ix['train']];qval=[normalized_question(rows[i]) for i in ix['validation']];qtest=[normalized_question(r) for r in test]
    question_options=[]
    for c in [.1,1.,10.]:
        pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True),LogisticRegression(C=c,max_iter=1000,random_state=cfg['split_seed']))
        pipe.fit(qtrain,y[ix['train']].tolist());score=classification_metrics(y[ix['validation']].tolist(),pipe.predict(qval).tolist())['macro_f1'];question_options.append((score,c,pipe))
    score,c,pipe=max(question_options,key=lambda v:(v[0],-v[1]));add('question_only',pipe.predict(qtest).tolist(),extra={'selected_C':c,'validation_macro_f1':score,'visual_input':False,'normalization':'Object IDs and coordinates removed; camera identity retained.'})
    # A stronger literal-text control retains every query token, including object identifiers.
    raw_options=[]
    for c in [.1,1.,10.]:
        raw_pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True),LogisticRegression(C=c,max_iter=1000,random_state=cfg['split_seed']))
        raw_pipe.fit([rows[i]['question'] for i in ix['train']],y[ix['train']].tolist())
        raw_score=classification_metrics(y[ix['validation']].tolist(),raw_pipe.predict([rows[i]['question'] for i in ix['validation']]).tolist())['macro_f1']
        raw_options.append((raw_score,c,raw_pipe))
    raw_score,raw_c,raw_pipe=max(raw_options,key=lambda v:(v[0],-v[1]))
    add('question_only_raw',raw_pipe.predict([r['question'] for r in test]).tolist(),extra={'selected_C':raw_c,'validation_macro_f1':raw_score,'visual_input':False,'normalization':'None; includes all literal question text, object identifiers and coordinates.'})
    for name,choice in selection['selected'].items():
        for seed,fit in zip(cfg['train_seeds'],choice['fits']):
            model,ckpt=load_classifier(ROOT/fit['checkpoint'])
            with torch.inference_mode():pred=model(x[ix['test']]).argmax(-1).tolist()
            timing=classifier_timing(model,x[ix['test'][:1]])
            base=np.mean([feature_times[r['id']]+prepare_times[r['frame_id']] for r in test]);facts_mean=np.mean([fact_times[r['frame_id']] for r in test])
            extra={**timing,'parameters':model.trainable_parameters,'selected_learning_rate':choice['learning_rate'],'validation_macro_f1':fit['validation_macro_f1'],'end_to_end_mean_seconds':float(base+timing['cached_head_mean_seconds']),'fact_extraction_mean_seconds':float(facts_mean),'graph_and_rule_mean_seconds':float(np.mean(list(verify_times.values())))}
            add(name,pred,seed,extra,verify_variant=True)
            methods[f'{name}_verified/seed{seed}']['end_to_end_mean_seconds']+=float(facts_mean)+float(np.mean(list(verify_times.values())))
    aggregate={}
    for name in list(selection['selected'])+[n+'_verified' for n in selection['selected']]:
        mm=[methods[f'{name}/seed{s}'] for s in cfg['train_seeds']]
        aggregate[name]={metric:{'mean':float(np.mean([m[metric] for m in mm])),'std':float(np.std([m[metric] for m in mm],ddof=1))} for metric in ['accuracy','macro_f1','end_to_end_mean_seconds']}
        aggregate[name]['seeds']=cfg['train_seeds']
    evaluation={'test_questions':len(test),'test_scenes':len({r['scene_id'] for r in test}),'manifest_sha256':digest(rows),'methods':methods,'aggregate':aggregate,'facts_valid_frames':sum(r['facts']['valid'] for r in fact_rows),'facts_total_frames':len(fact_rows),'notes':['Rule satisfaction is measured against predicted facts, not an external safety oracle.','Pilot includes only one supported proceed question in test; most covered questions concern object taxonomy or relevance.','Verified end-to-end latency includes a separate, unamortized fact extraction pass per query.','GPU memory is the maximum observed allocation at recorded call boundaries, not a profiler-certified peak.','Question-only control normalizes object tags to prevent memorization of coordinates or object identifiers.']}
    timing=[r for r in read_jsonl(inf/'feature_timing.jsonl')]+[r['timing'] for r in fact_rows]+[r['timing'] for mode in ['direct','cot','graph','cot_two_pass'] for r in read_jsonl(inf/(mode+'.jsonl'))]
    evaluation['max_observed_gpu_allocated_mb']=max((r['gpu_allocated_mb'] or 0) for r in timing)
    evaluation['max_observed_gpu_driver_mb']=max((r.get('gpu_driver_mb') or 0) for r in timing)
    save_json(dest/'metrics.json',evaluation);save_json(dest/'predictions.json',row_outputs);save_json(dest/'rule_traces.json',traces_out)
    print('EVALUATION COMPLETE',len(methods),'method/seed conditions',flush=True)
    for name in ['direct','cot','graph','majority','question_only','question_only_raw']:print(name,methods[name]['accuracy'],methods[name]['macro_f1'],flush=True)
    for name,m in aggregate.items():print(name,m['accuracy'],m['macro_f1'],flush=True)
    return evaluation
