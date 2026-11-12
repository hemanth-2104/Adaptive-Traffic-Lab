"""Run from the repository root after: python -m pip install -e ."""
import argparse
import uvicorn
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8573);args=p.parse_args()
 print(f'Open http://127.0.0.1:{args.port}. Press Ctrl-C to stop.')
 uvicorn.run('nslcot.api:app',host='127.0.0.1',port=args.port)
