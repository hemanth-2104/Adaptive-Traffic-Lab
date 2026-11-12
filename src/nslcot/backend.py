"""Local frozen Qwen backend, with explicit device, token and timing provenance."""
import os,re,time,json,platform
from pathlib import Path
import numpy as np
import torch,psutil
from .io import ROOT,load_json,save_json,digest
from .data import TAG
from .images import compose_view
from .rules import parse_facts

SYSTEM='You are a visual traffic-scene question-answering assistant. Use the supplied camera images. Object references c1, c2, etc. refer to red markers in the indicated camera. Do not assume an object category from its identifier.'
DIRECT='Answer the following traffic-scene question. Output exactly Yes or No, with no other text.\nQuestion: {question}'
COT='Answer the following traffic-scene question using the images. Briefly reason about visible evidence in at most two sentences, then put the final answer on its own line as Answer: Yes or Answer: No.\nQuestion: {question}'
GRAPH='Predicted scene facts (may be incomplete or mistaken):\n{facts}\nUse these facts together with the camera images to answer the question. Output exactly Yes or No.\nQuestion: {question}'
FACTS='''Extract visible traffic facts from these labeled camera views. Red markers identify objects: {ids}.
Return ONLY valid JSON with this schema:
{{"objects":{{{object_schema}}},"pedestrian_crossing_path":"unknown","obstacle_blocks_path":"unknown","front_traffic_element":"unknown"}}
For kind, choose vehicle, pedestrian, cyclist, traffic_sign, traffic_light, barrier, other, or unknown. For in_ego_path and scene predicates use true, false, or "unknown". A marker identifies a point, not a category. Infer path intersection only if visually supported. Use unknown when uncertain. front_traffic_element means a visible traffic sign, light or barrier in CAM_FRONT. Do not answer a separate question; describe visible facts. Keep JSON compact.'''


def parse_answer(text,mode='direct'):
    text=str(text).strip()
    if mode=='cot':
        matches=re.findall(r'(?:^|\n)\s*Answer\s*:\s*(yes|no)\s*[.!]?\s*(?=\n|$)',text,re.I)
        return int(matches[-1].lower()=='yes') if len(matches)==1 else None
    m=re.fullmatch(r'\s*(yes|no)\s*[.!]?\s*',text,re.I)
    return int(m.group(1).lower()=='yes') if m else None


class QwenBackend:
    def __init__(self,config,device=None):
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.config=config
        self.device=device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype={'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[config.get('precision','bfloat16')] if self.device in ['mps','cuda'] else torch.float32
        self.model_path=ROOT/'models/qwen2.5-vl-3b'
        self.processor=AutoProcessor.from_pretrained(self.model_path,local_files_only=True,use_fast=True,min_pixels=28*28*16,max_pixels=config['max_pixels'])
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path,local_files_only=True,dtype=self.dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).to(self.device).eval()
        self.model.requires_grad_(False);self.calls=0;self.warmup=[]
        self.provenance=load_json(ROOT/'models/model_provenance.json')
        self.provenance.update({'device':self.device,'dtype':str(self.dtype),'torch':torch.__version__,'platform':platform.platform(),'feature_layer':'last decoder hidden layer after final normalization','feature_token':'last actual input prompt token after assistant generation prefix','attn_implementation':'sdpa','image_processor_use_fast':True,'config_sha256':digest(config),'prompts_sha256':digest([SYSTEM,DIRECT,COT,GRAPH,FACTS])})
        save_json(ROOT/'results/runtime.json',self.provenance)

    def synchronize(self):
        if self.device=='mps':torch.mps.synchronize()
        elif self.device=='cuda':torch.cuda.synchronize()

    def inputs(self,image,text):
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[prompt],images=[image],padding=True,return_tensors='pt')
        return inputs.to(self.device)

    def memory(self):
        return {'process_rss_mb':psutil.Process().memory_info().rss/(1024**2),'gpu_allocated_mb':float(torch.mps.current_allocated_memory()/(1024**2)) if self.device=='mps' else float(torch.cuda.max_memory_allocated()/(1024**2)) if self.device=='cuda' else None,'gpu_driver_mb':float(torch.mps.driver_allocated_memory()/(1024**2)) if self.device=='mps' else None}

    @torch.inference_mode()
    def feature(self,image,question):
        self.synchronize();t=time.perf_counter();inputs=self.inputs(image,DIRECT.format(question=question));pre=time.perf_counter()-t
        self.synchronize();t=time.perf_counter()
        out=self.model(**inputs,output_hidden_states=True,use_cache=False,logits_to_keep=1)
        feature=out.hidden_states[-1][0,-1].float().cpu().numpy();self.synchronize();elapsed=time.perf_counter()-t
        assert np.isfinite(feature).all(), 'Non-finite backbone features'
        self.calls+=1
        return feature,{'input_tokens':int(inputs['input_ids'].shape[-1]),'processor_seconds':pre,'model_seconds':elapsed,'total_seconds':pre+elapsed,'call_index':self.calls,**self.memory()}

    @torch.inference_mode()
    def generate(self,image,text,max_new_tokens):
        self.synchronize();t=time.perf_counter();inputs=self.inputs(image,text);pre=time.perf_counter()-t
        self.synchronize();t=time.perf_counter()
        out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
        tokens=out[0,inputs['input_ids'].shape[-1]:];response=self.processor.decode(tokens,skip_special_tokens=True).strip();self.synchronize();elapsed=time.perf_counter()-t;self.calls+=1
        return response,{'input_tokens':int(inputs['input_ids'].shape[-1]),'output_tokens':len(tokens),'processor_seconds':pre,'model_seconds':elapsed,'total_seconds':pre+elapsed,'call_index':self.calls,**self.memory()}

    def facts(self,image,questions):
        ids=sorted({m[0] for q in questions for m in TAG.findall(q)})
        schema=','.join(json.dumps(cid)+':{"kind":"unknown","in_ego_path":"unknown"}' for cid in ids)
        prompt=FACTS.format(ids=', '.join(ids) or 'none',object_schema=schema)
        raw,timing=self.generate(image,prompt,self.config['fact_max_new_tokens'])
        return parse_facts(raw,ids),raw,timing

    def answer(self,image,question,mode='direct',facts=None):
        text=DIRECT.format(question=question) if mode=='direct' else COT.format(question=question) if mode=='cot' else GRAPH.format(question=question,facts=json.dumps(facts,separators=(',',':')))
        raw,timing=self.generate(image,text,self.config['cot_max_new_tokens'] if mode=='cot' else self.config['direct_max_new_tokens'])
        return parse_answer(raw,mode),raw,timing
