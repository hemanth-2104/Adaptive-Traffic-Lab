"""Conservative verification over predicted facts; no reference labels accepted."""
from dataclasses import dataclass,asdict
import re,json
import networkx as nx
from .data import TAG,question_intent

KINDS={'vehicle','pedestrian','cyclist','traffic_sign','traffic_light','barrier','other','unknown'}
BOOLS={'true':True,'false':False,'unknown':None}


def tri(value):
    if value is True or value is False:return value
    if isinstance(value,str):return BOOLS.get(value.strip().lower())
    return None


def parse_facts(raw,allowed_objects=None):
    """Malformed JSON becomes unknown; never coerce a string to a truthy boolean."""
    if isinstance(raw,dict):obj=raw
    else:
        text=str(raw).strip()
        if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
        try:obj=json.loads(text)
        except (ValueError,TypeError):return {'valid':False,'objects':{},'pedestrian_crossing_path':None,'obstacle_blocks_path':None,'front_traffic_element':None}
    if not isinstance(obj,dict):return parse_facts(None)
    objects={}
    for key,v in (obj.get('objects',{}) or {}).items() if isinstance(obj.get('objects',{}),dict) else []:
        if allowed_objects is not None and key not in allowed_objects:continue
        if not isinstance(v,dict):continue
        kind=v.get('kind','unknown');kind=kind if kind in KINDS else 'unknown'
        objects[key]={'kind':kind,'in_ego_path':tri(v.get('in_ego_path'))}
    return {'valid':True,'objects':objects,**{k:tri(obj.get(k)) for k in ['pedestrian_crossing_path','obstacle_blocks_path','front_traffic_element']}}


def scene_graph(facts):
    g=nx.DiGraph();g.add_node('ego',kind='ego')
    for key,obj in facts.get('objects',{}).items():
        g.add_node(key,kind=obj['kind'])
        if obj['in_ego_path'] is not None:g.add_edge(key,'ego',relation='in_path',value=obj['in_ego_path'])
    for k in ['pedestrian_crossing_path','obstacle_blocks_path','front_traffic_element']:
        g.add_node(k,kind='predicate',value=facts.get(k))
        if facts.get(k) is not None:g.add_edge(k,'ego',relation='scene_fact',value=facts[k])
    return g


def active_checks(question,facts):
    intent=question_intent(question);checks=[];g=scene_graph(facts)
    def add(rule,required,evidence):checks.append({'rule_id':rule,'required_answer':int(required),'evidence':evidence})
    ego_action=not TAG.search(question) and bool(re.match(r'^(?:(?:can|could|may|should|must)\s+(?:the\s+)?(?:ego\s+vehicle|self(?:[- ]driving)?\s+(?:vehicle|car)|own\s+(?:vehicle|car)|vehicle|car|i|we)\b|is it safe (?:for the ego vehicle )?to\b)',question.strip(),re.I))
    if intent in ['proceed','stop'] and ego_action:
        for pred in ['pedestrian_crossing_path','obstacle_blocks_path']:
            if g.nodes[pred].get('value') is True:add(pred+'_'+intent,int(intent=='stop'),{pred:True})
    refs=[m[0] for m in TAG.findall(question)]
    if intent=='traffic_object' and len(set(refs))==1:
        cid=refs[0];kind=g.nodes[cid].get('kind','unknown') if cid in g else 'unknown'
        if kind!='unknown':add('object_taxonomy',kind in ['traffic_sign','barrier'],{cid:{'kind':kind}})
    ego_considers=bool(re.search(r'\bego (?:vehicle|car)\s+(?:(?:should|must|needs? to|to)\s+)?(?:consider|take)\b',question,re.I))
    if intent=='consider' and len(set(refs))==1 and ego_considers:
        cid=refs[0];edge=g.get_edge_data(cid,'ego') or {}
        # One-way implication: unknown/off-path does not mean irrelevant.
        if edge.get('relation')=='in_path' and edge.get('value') is True:add('in_path_requires_consideration',1,{cid:{'in_ego_path':True}})
    if intent=='front_traffic' and g.nodes['front_traffic_element'].get('value') is not None:
        add('front_traffic_consistency',g.nodes['front_traffic_element']['value'],{'front_traffic_element':g.nodes['front_traffic_element']['value']})
    return checks


def verify(question,prediction,facts):
    if prediction not in [0,1,None]:raise ValueError('prediction must be 0, 1, or None')
    checks=active_checks(question,facts);allowed={0,1}
    for c in checks:allowed&={c['required_answer']}
    violations=[c['rule_id'] for c in checks if prediction!=c['required_answer']]
    final=prediction;status='uncovered'
    if checks:
        if not allowed:status='unresolved_conflict'
        elif prediction is None:status='unresolved_invalid_answer'
        elif prediction in allowed:status='consistent'
        elif len(allowed)==1:final=next(iter(allowed));status='corrected'
        else:status='unresolved'
    return {'original':prediction,'final':final,'changed':final!=prediction,'status':status,'checks':checks,'violations_before':violations,'violations_after':[c['rule_id'] for c in checks if final!=c['required_answer']]}


def graph_json(facts):
    return nx.node_link_data(scene_graph(facts),edges='edges')
