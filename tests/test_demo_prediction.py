"""Exercise real trained-head inference and what-if isolation on isolated fixtures."""
import json
import numpy as np
import torch
from fastapi.testclient import TestClient
from nslcot.models import FeatureClassifier
from nslcot.data import CAMERAS


def test_demo_predict_and_fact_override_do_not_mutate_benchmark(tmp_path,monkeypatch):
    import nslcot.api as api
    monkeypatch.setattr(api,'ROOT',tmp_path)
    for fn in [api.manifest,api.records,api.labels,api.selection,api.checkpoint]:fn.cache_clear()
    dirs=['data/processed','results/inference/features','results/training']
    for path in dirs:(tmp_path/path).mkdir(parents=True)
    row={'id':'fixture','scene_id':'s','frame_id':'f','category':'planning','question':'Can the vehicle proceed?','images':{},'intent':'proceed','split':'test'}
    (tmp_path/'data/processed/manifest.json').write_text(json.dumps([row]));(tmp_path/'data/processed/labels.json').write_text('{"fixture":0}')
    np.save(tmp_path/'results/inference/features/fixture.npy',np.ones(16,dtype=np.float32))
    torch.manual_seed(18);model=FeatureClassifier(16,8,'gru',3)
    with torch.no_grad():model.head.weight.zero_();model.head.bias.copy_(torch.tensor([0.,2.]))
    torch.save({'state_dict':model.state_dict(),'input_dim':16,'hidden_dim':8,'kind':'gru','steps':3,'mean':[0.]*16,'std':[1.]*16},tmp_path/'results/training/model.pt')
    (tmp_path/'results/training/selection.json').write_text(json.dumps({'selected':{'latent_gru':{'fits':[{'checkpoint':'results/training/model.pt'}]}}}))
    facts_path=tmp_path/'results/inference/facts.jsonl';facts_path.write_text(json.dumps({'frame_id':'f','facts':{'valid':True,'objects':{},'pedestrian_crossing_path':None,'obstacle_blocks_path':None,'front_traffic_element':None}})+'\n')
    before=facts_path.read_bytes();client=TestClient(api.app)
    normal=client.post('/api/predict',json={'record_id':'fixture','model':'latent_gru'}).json()
    assert normal['verification']['original']==1 and normal['verification']['final']==1 and not normal['what_if']
    altered=client.post('/api/predict',json={'record_id':'fixture','model':'latent_gru','automatic_correction':True,'facts_override':{'pedestrian_crossing_path':True}}).json()
    assert altered['what_if'] and altered['verification']['final']==0
    again=client.post('/api/predict',json={'record_id':'fixture','model':'latent_gru'}).json()
    assert again['verification']['final']==1 and facts_path.read_bytes()==before
    assert normal['probabilities']['yes']>normal['probabilities']['no']
    for fn in [api.manifest,api.records,api.labels,api.selection,api.checkpoint]:fn.cache_clear()
