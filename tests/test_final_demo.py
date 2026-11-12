import torch
from nslcot.models import FeatureClassifier
from nslcot.demo_analysis import probability_evolution
from nslcot.targeted import parse_targeted,legacy,FIELDS

def test_latent_readout_matches_final_prediction():
 torch.manual_seed(18)
 for depth in [1,3,5]:
  m=FeatureClassifier(8,4,'gru',depth).eval();x=torch.randn(1,8)
  states=probability_evolution(m,x)
  assert len(states)==depth
  assert abs(states[-1]['yes']-float(m(x).softmax(-1)[0,1].detach()))<1e-7

def test_strict_fact_schema_rejects_inconsistent_status_and_extra_keys():
 import json
 good={k:{'value':'unknown','status':'unknown'} for k in FIELDS}
 assert parse_targeted(json.dumps(good))['valid']
 good['crossing']={'value':'yes','status':'unknown'}
 assert not parse_targeted(json.dumps(good))['valid']
 assert not parse_targeted('{}')['valid']

def test_crossing_without_known_path_does_not_become_hazard():
 fields={k:{'value':'unknown','status':'unknown'} for k in FIELDS}
 fields['object_type']={'value':'pedestrian','status':'known'};fields['crossing']={'value':'yes','status':'known'}
 f=legacy({'question':'Can the ego vehicle proceed?'},{'valid':True,'fields':fields})
 assert f['pedestrian_crossing_path'] is None

def test_final_demo_api_uses_real_cases():
 from fastapi.testclient import TestClient
 from nslcot.api import app
 c=TestClient(app);g=c.get('/api/gallery');assert g.status_code==200
 examples=g.json()['examples'];assert any(x['kind']=='harmful' for x in examples)
 rid=examples[0]['id'];v=c.get('/api/compare/'+rid);assert v.status_code==200
 assert len(v.json()['latent_gru']['evolution'])==3
 assert c.get('/api/compare/not-a-real-id').status_code==404


def test_invalid_extraction_never_activates_rules():
 from nslcot.rules import verify
 parsed={'valid':False,'fields':{}}
 facts=legacy({'question':'Is <c1,CAM_FRONT,10,20> a traffic sign or a road barrier?'},parsed)
 assert not facts['valid']
 result=verify('Is <c1,CAM_FRONT,10,20> a traffic sign or a road barrier?',0,facts)
 assert result['final']==0 and not result['checks']

def test_new_diagnostic_scenes_are_disjoint():
 from nslcot.io import ROOT,load_json
 pilot=load_json(ROOT/'data/processed/manifest.json');diag=load_json(ROOT/'results/final_demo/manifest.json')
 assert len(diag)==100
 assert {r['scene_id'] for r in pilot}.isdisjoint({r['scene_id'] for r in diag})
 assert all('label' not in r and 'answer' not in r for r in diag)

def test_constrained_extractor_returns_only_allowed_schema():
 from types import SimpleNamespace
 from nslcot.constrained_facts import extract
 import json
 class Tokenizer:
  def encode(self,s,**kw):return [ord(s)-65]
 class Backend:
  processor=SimpleNamespace(tokenizer=Tokenizer())
  def inputs(self,image,text):return {}
  def model(self,**kw):
   logits=torch.zeros(1,1,26);logits[0,0,0]=2
   return SimpleNamespace(logits=logits)
  def synchronize(self):pass
 parsed,raw,timing=extract(Backend(),None,{'question':'Is c1 a traffic sign?','intent':'traffic_object'})
 assert parse_targeted(json.dumps(parsed['fields']))['valid']
 assert parsed['fields']['object_type']['value']=='vehicle'
 assert parsed['fields']['crossing']['status']=='not_applicable'
 assert len(raw)==1

def test_adaptive_option_stops_updates_and_matches_readout():
 from nslcot.demo_analysis import adaptive_evolution
 torch.manual_seed(42);m=FeatureClassifier(8,4,'gru',5).eval();x=torch.randn(1,8)
 full=probability_evolution(m,x);early=adaptive_evolution(m,x,1.)
 assert len(early)==2 and early[-1]==full[1]
 assert len(adaptive_evolution(m,x,0.))==5

def test_adaptive_api_matches_its_inspection_panel():
 from fastapi.testclient import TestClient
 from nslcot.api import app
 c=TestClient(app);rid=c.get('/api/gallery').json()['examples'][0]['id']
 result=c.post('/api/predict',json={'record_id':rid,'model':'adaptive_gru'})
 assert result.status_code==200
 panel=c.get('/api/compare/'+rid).json()['adaptive_gru']
 assert abs(result.json()['probabilities']['yes']-panel['yes'])<1e-7
 assert 2<=len(panel['evolution'])<=5
