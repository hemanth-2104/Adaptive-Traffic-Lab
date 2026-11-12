"""Question-conditioned categorical facts from constrained next-token choices.

Scores normalize over allowed letters only; they are not calibrated confidence.
"""
import time
import torch
from .targeted import FIELDS

OPTIONS={'object_type':['vehicle','pedestrian','cyclist','traffic_sign','traffic_light','barrier','other','unknown'],'crossing':['yes','no','unknown'],'in_ego_path':['yes','no','unknown'],'obstacle_blocks_path':['yes','no','unknown'],'front_traffic_element':['yes','no','unknown']}
DESCRIPTIONS={'object_type':'What is the referenced object category? If no object is referenced, identify a pedestrian relevant to the ego path only if visible; otherwise unknown.','crossing':'Is that referenced pedestrian crossing? Unknown if no relevant pedestrian is identifiable.','in_ego_path':'Does that referenced object intersect the visually identifiable ego path? Unknown if the intended path is not identifiable.','obstacle_blocks_path':'Does a visible obstruction block the visually identifiable ego path? Unknown if path or obstruction is unclear.','front_traffic_element':'Is there a visible traffic sign, traffic light or barrier in CAM_FRONT?'}
RELEVANT={'traffic_object':['object_type'],'consider':['object_type','in_ego_path'],'front_traffic':['front_traffic_element'],'proceed':['object_type','crossing','in_ego_path','obstacle_blocks_path'],'stop':['object_type','crossing','in_ego_path','obstacle_blocks_path']}

@torch.inference_mode()
def extract(backend,image,row):
 fields={f:{'value':'not_applicable','status':'not_applicable'} for f in FIELDS};evidence={};start=time.perf_counter()
 for field in RELEVANT.get(row['intent'],[]):
  values=OPTIONS[field];letters=[chr(65+i) for i in range(len(values))];token_ids=[]
  for letter in letters:
   ids=backend.processor.tokenizer.encode(letter,add_special_tokens=False)
   if len(ids)!=1:raise ValueError('Choice label must be exactly one token')
   token_ids.append(ids[0])
  prompt='Question context: '+row['question']+'\nObserve the supplied visual evidence. '+DESCRIPTIONS[field]+'\nChoose the letter of the best supported value. Unknown is appropriate when evidence is insufficient.\n'+'\n'.join(f'{k}: {v}' for k,v in zip(letters,values))+'\nOutput one letter only.'
  inputs=backend.inputs(image,prompt);output=backend.model(**inputs,use_cache=False,logits_to_keep=1)
  logits=output.logits[0,-1,token_ids].float().cpu();probs=logits.softmax(-1);idx=int(logits.argmax());value=values[idx]
  fields[field]={'value':value,'status':'unknown' if value=='unknown' else 'known'};evidence[field]={'choices':values,'restricted_probabilities':probs.tolist(),'selected_letter':letters[idx]}
 backend.synchronize()
 return {'valid':True,'fields':fields},evidence,{'total_seconds':time.perf_counter()-start,'method':'next-token logits restricted to allowed letter choices; not calibrated confidence'}
