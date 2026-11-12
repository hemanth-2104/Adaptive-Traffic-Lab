from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
selection=json.loads((ROOT/'results/training/selection.json').read_text());out=ROOT/'results/figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
for ax,name,title in zip(axes,['latent_gru','matched_feedforward'],['Three-step GRU','Parameter-matched feedforward']):
 for seed,entry,color in zip([18,42,572],selection['selected'][name]['fits'],['#176b49','#638c72','#ba8a52']):
  meta=json.loads((ROOT/entry['metadata']).read_text());hist=meta['history'];ax.plot([r['epoch'] for r in hist],[r['validation_macro_f1'] for r in hist],label=f'Seed {seed}',color=color,linewidth=1.6);ax.scatter(meta['best_epoch'],meta['validation_macro_f1'],color=color,s=18,zorder=4)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Training epoch');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel('Validation macro-F1');fig.suptitle('Checkpoint selection uses validation scores only',x=.09,ha='left',fontsize=12,fontweight='bold');fig.tight_layout();fig.savefig(out/'validation_learning_curves.png',dpi=180,bbox_inches='tight');print('Saved validation learning curves')
