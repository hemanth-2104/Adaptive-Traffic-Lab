import pytest
import torch
from nslcot.data import binary_label,question_intent,inference_input
from nslcot.backend import parse_answer
from nslcot.rules import tri,parse_facts,verify,scene_graph
from nslcot.metrics import classification_metrics,verification_metrics
from nslcot.models import FeatureClassifier

@pytest.mark.parametrize('q,a,y',[
 ('Is a pedestrian present?','Yes.',1),('Can the car accelerate?','No, it cannot.',0),
 ('What is the sign?','No entry.',None),('What actions lead to collision?','No such action.',None),
 ('Can it proceed?','Maybe.',None),('Can it proceed?','Yes, if conditions change.',None),
 ('Can it proceed?','Yes. No.',None),('Can it proceed?','Yes, it depends.',None),
 ('Is it moving?','Not sure.',None),('Is it moving?','Yesterday.',None)])
def test_binary_filter(q,a,y):assert binary_label(q,a)==y

@pytest.mark.parametrize('text,mode,result',[
 ('Yes.','direct',1),('No','direct',0),('Yes, because I see it','direct',None),
 ('Evidence.\nAnswer: No','cot',0),('Answer: Yes\nAnswer: No','cot',None),
 ('I think yes','cot',None),('No.\nAnswer: Yes','cot',1),('not yes','direct',None)])
def test_answer_parser(text,mode,result):assert parse_answer(text,mode)==result

def test_unknown_does_not_become_false():
 assert tri('false') is False and tri('unknown') is None and tri(0) is None
 facts=parse_facts('{broken')
 assert not facts['valid']
 assert verify('Can the vehicle proceed?',1,facts)['final']==1

def test_polarity_and_conservative_rules():
 facts=parse_facts({'pedestrian_crossing_path':True})
 assert verify('Can the vehicle proceed?',1,facts)['final']==0
 assert verify('Should the vehicle stop?',0,facts)['final']==1
 assert verify('Should the vehicle not stop?',0,facts)['status']=='uncovered'
 assert verify('Are there cars to the front?',0,facts)['status']=='uncovered'
 assert verify('Can the vehicle proceed?',None,facts)['final'] is None

def test_irrelevance_not_inferred_from_offpath():
 q='Is <c1,CAM_FRONT,5.0,5.0> an object that the ego vehicle should consider in the current scene?'
 assert verify(q,0,parse_facts({'objects':{'c1':{'kind':'vehicle','in_ego_path':False}}}))['status']=='uncovered'
 assert verify(q,0,parse_facts({'objects':{'c1':{'kind':'vehicle','in_ego_path':True}}}))['final']==1

def test_taxonomy_and_graph():
 q='Is <c1,CAM_FRONT,5.0,5.0> a traffic sign or a road barrier?'
 f=parse_facts({'objects':{'c1':{'kind':'traffic_sign','in_ego_path':'unknown'},'c99':{'kind':'vehicle'}}},['c1'])
 assert set(f['objects'])=={'c1'}
 assert verify(q,0,f)['final']==1
 assert scene_graph(f).nodes['c1']['kind']=='traffic_sign'

def test_invalid_predictions_in_denominator():
 m=classification_metrics([0,1,1],[0,1,None])
 assert m['accuracy']==pytest.approx(2/3) and m['invalid_rate']==pytest.approx(1/3)
 assert m['macro_f1']==pytest.approx(5/6)

def test_empty_constraint_set_is_undefined():
 rows=[{'id':'a','frame_id':'f','question':'Can the vehicle proceed?'}]
 m,traces=verification_metrics(rows,[1],[0],{})
 assert m['coverage']==0 and m['rule_satisfaction_after'] is None

def test_reference_information_removed():
 row={'id':'a','question':'Is it moving?','scene_description':'secret','label':1,'key_object_infos':{'kind':'car'}}
 assert inference_input(row)=={'id':'a','question':'Is it moving?'}

@pytest.mark.parametrize('kind,steps',[('mlp',3),('matched_mlp',3),('gru',1),('gru',3),('gru',5)])
def test_classifier_gradients_and_determinism(kind,steps):
 torch.manual_seed(18);model=FeatureClassifier(16,8,kind,steps)
 x=torch.randn(6,16);out=model(x);assert out.shape==(6,2)
 loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,1,0,1,0,1]));loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 assert torch.equal(model(x),model(x))

def test_conflicting_constraints_preserve_prediction(monkeypatch):
 import nslcot.rules as rules
 monkeypatch.setattr(rules,'active_checks',lambda q,f:[{'rule_id':'r1','required_answer':0},{'rule_id':'r2','required_answer':1}])
 result=rules.verify('Can it proceed?',1,{})
 assert result['status']=='unresolved_conflict' and result['final']==1 and not result['changed']

def test_growing_jsonl_does_not_hide_completed_corruption(tmp_path):
 from nslcot.io import read_jsonl
 p=tmp_path/'fixture.jsonl';p.write_text('{"a":1}\n{"a":')
 assert read_jsonl(p)==[{'a':1}]
 p.write_text('{"a":1}\n{broken}\n')
 import json
 with pytest.raises(json.JSONDecodeError):read_jsonl(p)

def test_reversed_agent_roles_do_not_activate_ego_rules():
 from nslcot.rules import parse_facts,verify
 facts=parse_facts({'objects':{'c1':{'kind':'vehicle','in_ego_path':True}},'pedestrian_crossing_path':True})
 q='Would <c1,CAM_FRONT,5.0,5.0> take the ego vehicle into account?'
 assert verify(q,0,facts)['status']=='uncovered'
 assert verify('Should <c1,CAM_FRONT,5.0,5.0> stop?',0,facts)['status']=='uncovered'
 assert verify('Can a pedestrian proceed?',1,facts)['status']=='uncovered'
 assert verify('Can the car accelerate?',1,facts)['final']==0

def test_paired_grouped_interval_is_zero_for_identical_predictions():
 from nslcot.metrics import paired_scene_bootstrap
 rows=[{'scene_id':'s1'},{'scene_id':'s1'},{'scene_id':'s2'}]
 result=paired_scene_bootstrap(rows,[0,1,0],[0,0,0],[0,0,0],iterations=100)
 assert result['accuracy_delta_ci95']==[0.,0.]
