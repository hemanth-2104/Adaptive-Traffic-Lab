from fastapi.testclient import TestClient
from nslcot.api import app
from nslcot.io import ROOT,load_json,file_hash
from nslcot.prediction import ensemble_probabilities
import numpy as np

def test_default_preserves_ensemble_prediction():
 c=TestClient(app);rid=c.get('/api/questions').json()[0]['id'];r=c.post('/api/predict',json={'record_id':rid}).json()
 assert r['model']=='ensemble' and r['seeds']==[18,42,572]
 assert r['verification']['mode']=='advisory'
 assert r['verification']['final']==r['verification']['original']
 assert not r['verification']['changed']

def test_advisory_flags_real_contradiction_without_applying_it():
 c=TestClient(app);ex=next(e for e in c.get('/api/gallery').json()['examples'] if e['kind']=='harmful')
 before=file_hash(ROOT/'results/inference/facts.jsonl');payload={'record_id':ex['id'],'model':'latent_gru'}
 advisory=c.post('/api/predict',json=payload).json()['verification']
 applied=c.post('/api/predict',json={**payload,'automatic_correction':True}).json()['verification']
 assert advisory['status']=='advisory_contradiction'
 assert advisory['final']==advisory['original']!=advisory['proposed_final']
 assert applied['final']==advisory['proposed_final'] and applied['changed']
 assert advisory['violations_after']==advisory['violations_before']
 assert file_hash(ROOT/'results/inference/facts.jsonl')==before

def test_ensemble_matches_saved_outputs():
 saved=load_json(ROOT/'results/ensemble/metrics.json')['predictions']
 for row in saved:
  h=np.load(ROOT/'results/inference/features'/(row['id']+'.npy'),allow_pickle=False);p=ensemble_probabilities(h)
  assert int(p[1]>p[0])==row['prediction']

def test_normal_gallery_is_ensemble_and_correction_examples_are_explicit():
 g=TestClient(app).get('/api/gallery').json()['examples'];normal=next(x for x in g if x['kind']=='normal')
 assert normal['model']=='ensemble' and not normal['automatic_correction']
 for e in g:
  if e['kind'] in ['beneficial','harmful']:assert e['model']=='latent_gru' and e['automatic_correction']
