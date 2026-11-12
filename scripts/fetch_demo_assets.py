"""Download the published CPU-demo assets and verify their SHA-256 before extraction."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,help='Use an already downloaded release bundle');a=p.parse_args()
 config=json.loads((ROOT/'configs/demo_assets.json').read_text());cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
 archive=a.archive or cache/config['filename']
 if not archive.exists():
  tmp=archive.with_suffix('.download');print('Downloading',config['url']);urllib.request.urlretrieve(config['url'],tmp);tmp.replace(archive)
 h=hashlib.sha256()
 with archive.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=config['sha256']:raise ValueError('Bundle checksum mismatch; refusing to extract')
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   dest=(ROOT/info.filename).resolve()
   if not dest.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe archive path')
   if info.is_dir():continue
   # The release cannot replace source, settings, or an environment.
   if Path(info.filename).parts[0] not in ['data','results','models']:raise ValueError('Unexpected archive member')
   dest.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
 print('Verified demo assets installed. Source images and Qwen weights are separate downloads.')
if __name__=='__main__':main()
