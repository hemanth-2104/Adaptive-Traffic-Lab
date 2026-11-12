from pathlib import Path
import hashlib,json,os,tempfile

ROOT=Path(os.environ.get('TRAFFIC_LAB_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve()

def load_json(path):
    return json.loads(Path(path).read_text())

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(tmp,path)

def read_jsonl(path):
    path=Path(path)
    if not path.exists():return []
    raw=path.read_text();lines=raw.splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not raw.endswith('\n'):continue # concurrent writer has not finished its final row
            raise
    return rows

def append_jsonl(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
