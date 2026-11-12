"""Extract only referenced images from the original author-provided archive."""
from pathlib import Path
import argparse, zipfile
from nslcot.io import ROOT,load_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
 rows=load_json(ROOT/'data/processed/manifest.json')
 target=ROOT/'results/final_demo/manifest.json'
 if target.exists():rows+=load_json(target)
 needed={v for r in rows for v in r['images'].values()};found=set();root=ROOT/'data/images'
 with zipfile.ZipFile(a.archive) as z:
  for info in z.infolist():
   parts=Path(info.filename).parts
   if 'samples' not in parts:continue
   rel=Path(*parts[parts.index('samples'):])
   if str(rel) not in needed:continue
   dest=(root/rel).resolve()
   if not dest.is_relative_to(root.resolve()):raise ValueError('Unsafe archive path')
   dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_bytes(z.read(info))
   found.add(str(rel))
 missing=needed-found
 if missing:raise FileNotFoundError(f'{len(missing)} referenced images are absent from the archive')
 print(f'Ready: {len(found)} referenced images. No model download is needed for the cached demo.')
if __name__=='__main__':main()
