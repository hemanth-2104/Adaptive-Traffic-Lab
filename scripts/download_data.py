"""Download original author-hosted assets. No credentials or gated mirrors used."""
from pathlib import Path
import argparse,json,hashlib,zipfile
import gdown
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_SHA256={'annotations': 'e7348866761a76ca1595646f410615ddd5adffb7affd7e056a203da0bf85a8cd', 'images': '2cb95fafac00ca058c04735c339a183657d60962afee12d782b410c602eab936'}
FILES={'annotations':('1LK7pYHytv64neN1626u6eTQBy1Uf4IQH','v1_0_train_nus.json'),'images':('1DeosPGYeM2gXSChjMODGsQChZyYDmaUz','drivelm_images.zip')}
parser=argparse.ArgumentParser();parser.add_argument('--asset',choices=['all',*FILES],default='all');args=parser.parse_args()
root=ROOT/'data/raw';root.mkdir(parents=True,exist_ok=True);provenance={}
for kind,(fid,name) in FILES.items():
 if args.asset not in ['all',kind]:continue
 path=root/name
 if not path.exists():gdown.download(id=fid,output=str(path),quiet=False)
 if kind=='annotations':json.loads(path.read_text())
 else:
  with zipfile.ZipFile(path) as z:
   if not z.namelist():raise ValueError('Empty image archive')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=EXPECTED_SHA256[kind]:raise ValueError(f'Asset checksum mismatch for {kind}; do not mix a changed dataset into this experiment.')
 provenance[kind]={'author_url':f'https://drive.google.com/file/d/{fid}/view','filename':name,'bytes':path.stat().st_size,'sha256':h.hexdigest()}
(root/'download_provenance.json').write_text(json.dumps(provenance,indent=2))
print('Assets verified:',', '.join(provenance))
