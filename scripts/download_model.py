from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi,snapshot_download
repo='Qwen/Qwen2.5-VL-3B-Instruct'
# Exact revision used by the completed experiment, also recorded in model_provenance.json.
PINNED_REVISION='66285546d2b821cf421d4f5eb2576359d3770cd3'
info=HfApi().model_info(repo,revision=PINNED_REVISION)
print('Downloading',repo,'revision',info.sha,flush=True)
p=snapshot_download(repo,revision=info.sha,local_dir=ROOT/'models/qwen2.5-vl-3b',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE','README.md'],max_workers=4)
(ROOT/'models/model_provenance.json').write_text(json.dumps({'model_id':repo,'revision':info.sha,'local_dir':str(p)},indent=2))
print('DONE',p,flush=True)
