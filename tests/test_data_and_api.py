"""Isolated synthetic fixtures test plumbing only; never benchmark evidence."""
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient
from nslcot import data


def test_manifest_group_disjoint_and_labels_separate(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'ROOT',tmp_path)
    raw={}
    for s in range(12):
        frames={}
        for f in range(2):
            frames[f'frame-{s}-{f}']={'image_paths':{c:f'../nuscenes/samples/{c}/{s}-{f}.jpg' for c in data.CAMERAS},'scene_description':'Must not enter inference','key_object_infos':{'category':'ground truth'},'QA':{'perception':[{'Q':'Are there cars in front?','A':'Yes.'}], 'prediction':[{'Q':'Will the object move?','A':'No.'}], 'planning':[{'Q':'Can the ego vehicle proceed?','A':'No.'}]}}
        raw[f'scene-{s}']={'scene_description':'privileged answer','key_frames':frames}
    path=tmp_path/'fixture.json';path.write_text(json.dumps(raw))
    cfg={'categories':['perception','prediction','planning'],'max_frames':20,'questions_per_category_per_frame':1,'split_seed':572,'split_ratios':[.7,.15,.15]}
    cp=tmp_path/'config.json';cp.write_text(json.dumps(cfg));stats=data.prepare(cp,path)
    rows=json.loads((tmp_path/'data/processed/manifest.json').read_text());labels=json.loads((tmp_path/'data/processed/labels.json').read_text())
    assert len(rows)==60 and set(labels)=={r['id'] for r in rows}
    assert all('label' not in r and 'scene_description' not in r and 'key_object_infos' not in r for r in rows)
    split_scenes={s:{r['scene_id'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert split_scenes['train'].isdisjoint(split_scenes['test'])
    assert split_scenes['validation'].isdisjoint(split_scenes['test'])
    first=(tmp_path/'data/processed/manifest.json').read_bytes();data.prepare(cp,path)
    assert (tmp_path/'data/processed/manifest.json').read_bytes()==first


def test_api_invalid_ids_and_parameter_validation(tmp_path,monkeypatch):
    import nslcot.api as api
    monkeypatch.setattr(api,'ROOT',tmp_path)
    for fn in [api.manifest,api.records,api.labels,api.selection,api.checkpoint]:fn.cache_clear()
    processed=tmp_path/'data/processed';processed.mkdir(parents=True)
    (processed/'manifest.json').write_text('[]')
    (processed/'statistics.json').write_text('{"selected":{}}')
    client=TestClient(api.app)
    assert client.get('/api/questions?split=unknown').status_code==400
    assert client.get('/api/image/../../outside').status_code==404
    assert client.get('/api/image/unknown').status_code==404
    assert client.post('/api/predict',json={'record_id':'unknown'}).status_code==404
    assert client.post('/api/predict',json={}).status_code==422
    assert client.get('/api/metrics').status_code==503
    assert client.get('/api/status').json()['ready'] is False
    for fn in [api.manifest,api.records,api.labels,api.selection,api.checkpoint]:fn.cache_clear()
