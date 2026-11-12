"""Generate empirical figures, tables, and diagnostic artifacts from saved results."""
import csv,io,json,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from .io import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import ROOT,load_json,save_json,digest
from .metrics import classification_metrics,paired_scene_bootstrap
from .rules import verify,parse_facts

PRIMARY=[('direct','Direct VLM'),('cot','CoT single pass'),('cot_two_pass','CoT + finalization'),('graph','Graph context'),('feedforward','Feedforward'),('latent_gru','Latent GRU'),('feedforward_verified','Feedforward + rules'),('latent_gru_verified','NSL-CoT')]


def analyze():
    ev=load_json(ROOT/'results/evaluation/metrics.json');preds=load_json(ROOT/'results/evaluation/predictions.json');traces=load_json(ROOT/'results/evaluation/rule_traces.json');stats=load_json(ROOT/'data/processed/statistics.json');cfg=load_json(ROOT/'configs/pilot.json')
    dest=ROOT/'results/figures';dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
    def value(name,metric):
        if name in ev['aggregate']:return ev['aggregate'][name][metric]['mean'],ev['aggregate'][name][metric]['std']
        return ev['methods'][name][metric],0.
    table=[]
    for name,label in PRIMARY:
        a,astd=value(name,'accuracy');f,fstd=value(name,'macro_f1');t,_=value(name,'end_to_end_mean_seconds')
        table.append({'method':name,'display':label,'accuracy':a,'accuracy_seed_std':astd,'macro_f1':f,'macro_f1_seed_std':fstd,'end_to_end_seconds':t})
    with (ROOT/'results/evaluation/comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    colors=['#a9b7ad']*len(PRIMARY);colors[-1]='#176b49'
    fig,ax=plt.subplots(figsize=(7.2,4.2));y=np.arange(len(table));ax.barh(y,[r['accuracy']*100 for r in table],xerr=[r['accuracy_seed_std']*100 for r in table],color=colors,capsize=3)
    ax.set_yticks(y,[r['display'] for r in table]);ax.invert_yaxis();ax.set_xlabel('Held-out accuracy (%)');ax.set_xlim(0,100);ax.set_title(f'DriveLM pilot · {ev["test_questions"]} questions / {ev["test_scenes"]} scenes',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for i,r in enumerate(table):ax.text((r['accuracy']+r['accuracy_seed_std'])*100+1.5,i,f'{r["accuracy"]*100:.1f}',va='center',fontsize=9)
    fig.text(.02,-.015,'Classifier error bars: standard deviation over three training seeds. Prompting: greedy decoding.',fontsize=8,color='#5d685f');fig.savefig(dest/'accuracy_comparison.png');fig.savefig(dest/'accuracy_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.7));matrix=[]
    for name,label in PRIMARY:
        mm=[ev['methods'][f'{name}/seed{s}'] for s in cfg['train_seeds']] if name in ev['aggregate'] else [ev['methods'][name]]
        matrix.append([np.mean([m['by_category'][c]['accuracy'] for m in mm]) for c in cfg['categories']])
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Greens',aspect='auto');ax.set_yticks(range(len(PRIMARY)),[n for _,n in PRIMARY]);ax.set_xticks(range(3),[c.title() for c in cfg['categories']]);ax.set_title('Accuracy by question category',loc='left',fontweight='bold')
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,f'{100*v:.1f}%',ha='center',va='center',color='white' if v>.65 else '#15221b')
    fig.colorbar(im,ax=ax,label='Accuracy');fig.savefig(dest/'category_accuracy.png');plt.close(fig)
    # Fixed-seed paired diagnostics complement the across-seed standard deviations.
    records={r['id']:r for r in load_json(ROOT/'data/processed/manifest.json')}
    pairs={}
    for before,after in [('feedforward/seed18','latent_gru/seed18'),('matched_feedforward/seed18','latent_gru/seed18'),('feedforward_verified/seed18','latent_gru_verified/seed18'),('question_only_raw','latent_gru/seed18')]:
        a=preds[before];b={r['id']:r for r in preds[after]};rr=[records[r['id']] for r in a]
        pairs[before+' -> '+after]=paired_scene_bootstrap(rr,[r['label'] for r in a],[r['prediction'] for r in a],[b[r['id']]['prediction'] for r in a])
    save_json(ROOT/'results/evaluation/paired_architecture_comparisons.json',{'fixed_seed':18,'comparisons':pairs,'note':'Exploratory scene-grouped paired intervals for one prespecified seed, distinct from across-seed variability; not multiplicity-adjusted hypothesis tests.'})
    corrections={}
    for name in ['feedforward','latent_gru']:
        cc=[ev['methods'][f'{name}/seed{s}']['verification'] for s in cfg['train_seeds']]
        corrections[name]={k:float(np.mean([m['changes'].get(k,0) for m in cc])) for k in ['incorrect_to_correct','correct_to_incorrect','unchanged']}
    fig,ax=plt.subplots(figsize=(5.6,3.3));xx=np.arange(2);w=.32
    ax.bar(xx-w/2,[corrections[k]['incorrect_to_correct'] for k in corrections],w,label='Incorrect → correct',color='#176b49');ax.bar(xx+w/2,[corrections[k]['correct_to_incorrect'] for k in corrections],w,label='Correct → incorrect',color='#bd6d42');ax.set_xticks(xx,['Feedforward','Latent GRU']);ax.set_ylabel('Changed questions · mean over seeds');ax.set_ylim(0,2.4);ax.set_title('Verification can help or harm',loc='left',fontweight='bold');ax.legend(fontsize=8,frameon=False);fig.savefig(dest/'corrections.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.7));ax.barh([r['display'] for r in table],[r['end_to_end_seconds'] for r in table],color=colors);ax.invert_yaxis();ax.set_xlabel('Mean end-to-end seconds per question');ax.set_title('Inference cost includes the fact extraction pass',loc='left',fontweight='bold');fig.savefig(dest/'latency.png');plt.close(fig)
    # Unknown-fact robustness: a fixed structural stress check, not retuning.
    from .io import read_jsonl
    fact_records=read_jsonl(ROOT/'results/inference/facts.jsonl')
    scene_fact_counts={key:dict(Counter(str(r['facts'].get(key)) for r in fact_records)) for key in ['pedestrian_crossing_path','obstacle_blocks_path','front_traffic_element']}
    object_path_counts=dict(Counter(str(o.get('in_ego_path')) for r in fact_records for o in r['facts'].get('objects',{}).values()))
    activation=Counter(c['rule_id'] for t in traces['latent_gru/seed18'] for c in t['checks'])
    save_json(ROOT/'results/evaluation/fact_coverage_details.json',{'scene_predicate_counts':scene_fact_counts,'object_path_counts':object_path_counts,'test_active_rule_counts_seed18':dict(activation),'note':'Unknown facts disable their checks. Rule coverage is empirical and may be limited to taxonomy; this is not a safety guarantee.'})
    manifest={r['id']:r for r in load_json(ROOT/'data/processed/manifest.json')};facts={r['frame_id']:r['facts'] for r in read_jsonl(ROOT/'results/inference/facts.jsonl')}
    sensitivity=[]
    for rate in [0.,.25,.5,.75,1.]:
        values=[]
        for seed in cfg['train_seeds']:
            rr=preds[f'latent_gru/seed{seed}'];out=[];covered=0
            for r in rr:
                row=manifest[r['id']];f=facts[row['frame_id']]
                # Deterministic frame-level masking independent of labels and predictions.
                masked=int(digest([572,row['frame_id']])[:8],16)/2**32<rate
                v=verify(row['question'],r['prediction'],parse_facts(None) if masked else f);out.append(v['final']);covered+=bool(v['checks'])
            values.append({'accuracy':classification_metrics([r['label'] for r in rr],out)['accuracy'],'coverage':covered/len(rr)})
        sensitivity.append({'masked_frame_fraction':rate,'mean_accuracy':float(np.mean([v['accuracy'] for v in values])),'mean_coverage':float(np.mean([v['coverage'] for v in values]))})
    save_json(ROOT/'results/evaluation/unknown_fact_sensitivity.json',sensitivity)
    # A deterministic 50-case audit packet; independent human decisions are not fabricated.
    cases=sorted(preds['latent_gru/seed18'],key=lambda r:digest([18,r['id']]))[:50]
    audit=[]
    for r in cases:
        row=manifest[r['id']];v=next(t for t in traces['latent_gru/seed18'] if t['id']==r['id'])
        audit.append({'id':r['id'],'scene_id':row['scene_id'],'question':row['question'],'category':row['category'],'reference':r['label'],'neural_prediction':r['prediction'],'verified_prediction':v['final'],'rules':';'.join(c['rule_id'] for c in v['checks']),'visual_evidence_review':'','reviewer_1':'','reviewer_2':'','resolution':''})
    audit_path=ROOT/'results/evaluation/qualitative_audit_50.csv'
    if not audit_path.exists():
        with audit_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    save_json(ROOT/'results/evaluation/analysis_summary.json',{'primary_table':table,'corrections':corrections,'fact_sensitivity':sensitivity,'audit_packet_cases':len(audit),'independent_human_audit_status':'Not performed; CSV is a review packet, not claimed human evaluation.'})
    from .fact_audit import audit_fact_kinds
    audit_fact_kinds()
    print('Created figures, CSV tables, sensitivity check, held-out kind audit, and 50-case audit packet.')
