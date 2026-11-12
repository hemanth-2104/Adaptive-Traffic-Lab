"""Check the visual-input and label-separation contract without downloading a model."""
import hashlib
import numpy as np
from PIL import Image
from nslcot import pipeline
from nslcot.io import digest,save_json
from nslcot.data import CAMERAS


def test_all_methods_receive_identical_six_view_images(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    images={}
    for i,c in enumerate(CAMERAS):
        rel=f'samples/{c}/fixture.jpg';path=tmp_path/'data/images'/rel;path.parent.mkdir(parents=True);Image.new('RGB',(160,90),(i*30,50,100)).save(path);images[c]=rel
    row={'id':'fixture-only','scene_id':'fixture-scene','frame_id':'fixture-frame','category':'perception','question':'Are there cars to the front?','images':images,'intent':'other','split':'test'}
    cfg={'view_width':112,'view_height':63}
    save_json(tmp_path/'data/processed/manifest.json',[row]);save_json(tmp_path/'data/processed/statistics.json',{'manifest_sha256':digest([row]),'config_sha256':digest(cfg)});save_json(tmp_path/'config.json',cfg)
    # No labels file exists; inference must not need one.
    captured=[]
    def record(image):captured.append((image.size,hashlib.sha256(image.tobytes()).hexdigest()))
    timing={'total_seconds':.01}
    class Backend:
        def __init__(self,cfg):self.device='fixture'
        def feature(self,image,question):record(image);return np.zeros(16,dtype=np.float32),timing
        def facts(self,image,questions):record(image);return {'valid':True,'objects':{}},'{}',timing
        def answer(self,image,question,mode,facts):record(image);return 0,'No',timing
    monkeypatch.setattr(pipeline,'QwenBackend',Backend)
    pipeline.run_inference(tmp_path/'config.json')
    assert len(captured)==6 # warm-up, feature, facts, and three prompting conditions
    assert len(set(captured))==1
    assert captured[0][0][0]==112*3 and captured[0][0][1]==2*(63+26)
