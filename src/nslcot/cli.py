import argparse,json
from .io import ROOT,load_json

def main():
    parser=argparse.ArgumentParser(description='Adaptive Traffic Lab research pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--config',default=str(ROOT/'configs/pilot.json'));p.add_argument('--raw')
    sub.add_parser('extract-images')
    p=sub.add_parser('infer');p.add_argument('--config',default=str(ROOT/'configs/pilot.json'));p.add_argument('--phase',choices=['all','features','facts','baselines'],default='all');p.add_argument('--limit',type=int)
    sub.add_parser('train');sub.add_parser('evaluate');sub.add_parser('analyze')
    p=sub.add_parser('serve');p.add_argument('--port',type=int,default=8572)
    p=sub.add_parser('predict');p.add_argument('--image',required=True);p.add_argument('--question',required=True);p.add_argument('--automatic-correction',action='store_true',help='Opt in to applying symbolic corrections; default is advisory')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare
        print(json.dumps(prepare(args.config,args.raw),indent=2))
    elif args.command=='extract-images':
        from .pipeline import extract_images
        print('Images verified:',extract_images())
    elif args.command=='infer':
        from .pipeline import run_inference
        run_inference(args.config,args.phase,args.limit)
    elif args.command=='train':
        from .experiments import train_models
        train_models()
    elif args.command=='evaluate':
        from .experiments import evaluate
        evaluate()
    elif args.command=='analyze':
        from .analysis import analyze
        analyze()
    elif args.command=='serve':
        import uvicorn
        uvicorn.run('nslcot.api:app',host='127.0.0.1',port=args.port)
    elif args.command=='predict':
        from PIL import Image
        import numpy as np,torch
        from .backend import QwenBackend
        from .experiments import load_classifier
        from .rules import verify
        b=QwenBackend(load_json(ROOT/'configs/pilot.json'));image=Image.open(args.image).convert('RGB');h,t=b.feature(image,args.question)
        from .prediction import ensemble_probabilities
        probs=ensemble_probabilities(h)
        f,raw,ft=b.facts(image,[args.question]);v=verify(args.question,int(probs[1]>probs[0]),f)
        v['proposed_final']=v['final'];v['mode']='automatic' if args.automatic_correction else 'advisory'
        if not args.automatic_correction:v['final']=v['original'];v['changed']=False;v['violations_after']=list(v['violations_before'])
        print(json.dumps({'probabilities':dict(zip(['no','yes'],probs)),'verification':v,'facts':f,'feature_timing':t,'fact_timing':ft,'notice':'Research prototype; not a driving controller.'},indent=2))

if __name__=='__main__':main()
