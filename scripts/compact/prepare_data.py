"""Pinned, byte-bounded experimental dataset builder. No held-out inference."""
import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import re
import sys
import tarfile
import threading
import time
from pathlib import Path
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--metadata', default='metadata/clips.parquet', help='pinned dataset clip metadata (parquet)')
parser.add_argument('--videos-only', action='store_true', help='skip the annotation download step')
args = parser.parse_args()
SOURCE = json.loads(Path('tracebrief/data/evaluation/sources.json').read_text())
URL = f"https://huggingface.co/datasets/{SOURCE['dataset_id']}/resolve/{SOURCE['dataset_revision']}/"
CAP = 3_000_000_000
lock = threading.Lock()
reserved = json.loads((BASE/'download_budget.json').read_text())['reserved_response_bytes'] if (BASE/'download_budget.json').exists() else 0
excluded = set(json.loads((BASE/'excluded_runs.json').read_text()))
pattern = re.compile(r'[a-f0-9]{20}_run_\d+_seed_\d+')
clips = pd.read_parquet(args.metadata)
eligible = clips[((clips.scenario=='nearmiss') & (clips.camera_alias=='ceiling_00')) |
                 ((clips.scenario=='box_pickup') & (clips.camera_alias=='cam_00'))]
eligible = {r['clip_key']:r for r in eligible.to_dict(orient='records') if r['run_id'] not in excluded}
records = json.loads((BASE/'download_manifest.json').read_text()) if (BASE/'download_manifest.json').exists() else []
for d in ['annotations','videos','logs']: (BASE/d).mkdir(exist_ok=True)

def save(name, value):
    (BASE/name).write_text(json.dumps(value, indent=2))

class Remote(io.RawIOBase):
    def __init__(self,url): self.url=url; self.pos=0; self.session=requests.Session(); self.cache_start=-1; self.cache=b''
    def tell(self): return self.pos
    def seek(self,n,whence=0):
        if whence==0: self.pos=n
        elif whence==1: self.pos+=n
        else: raise ValueError(whence)
        return self.pos
    def read(self,n=-1):
        global reserved
        if n<0 or n>8*1024*1024: raise ValueError('Unbounded range read')
        if not n: return b''
        start=self.pos
        if self.cache_start<=start and start+n<=self.cache_start+len(self.cache):
            b=self.cache[start-self.cache_start:start-self.cache_start+n]; self.pos+=n; return b
        # Small neighboring tar headers share one bounded read.
        size=max(n,16384) if n<16384 else n
        for attempt in range(4):
            with lock:
                if reserved+size>CAP: raise RuntimeError('3GB aggregate byte budget exceeded')
                reserved+=size
                save('download_budget.json',{'reserved_response_bytes':reserved,'cap_bytes':CAP})
            try:
                with self.session.get(self.url+f'?compact_range={start}',headers={'Range':f'bytes={start}-{start+size-1}'},stream=True,timeout=90) as response:
                    content_range=response.headers.get('Content-Range','')
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',content_range)
                    if response.status_code!=206 or not match:
                        raise RuntimeError(f'Refused HTTP {response.status_code} range {response.headers.get("Content-Range")}')
                    first,last,total=map(int,match.groups())
                    if first!=start or last!=min(start+size-1,total-1) or last-start+1<n:
                        raise RuntimeError(f'Unexpected range {content_range}')
                    actual=last-start+1
                    b=response.raw.read(actual)
                if len(b)!=actual: raise RuntimeError('Truncated range')
                self.cache_start=start;self.cache=b;self.pos+=n;return b[:n]
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)

def annotation_job(job):
    scenario,shard,count=job
    prefix='forklift_human_nearmiss/nearmiss' if scenario=='nearmiss' else 'warehouse_box_pickup/box_pickup'
    url=URL+f'artifacts/{prefix}-artifacts-{shard:05d}.tar'
    result=[r for r in records if r['annotation']['source_url']==url]
    if len(result)>=count:return result
    with tarfile.open(fileobj=Remote(url),mode='r:') as archive:
        for member in archive:
            if not member.name.endswith('.object_detection.jsonl'):continue
            run=pattern.search(member.name)
            if not run:continue
            camera='ceiling_00' if scenario=='nearmiss' else 'cam_00'
            key=run.group()+'.'+camera
            if key not in eligible or not member.name.endswith(key+'.object_detection.jsonl'):continue
            if any(r['clip_key']==key for r in result):continue
            row=eligible[key]; dest=BASE/'annotations'/Path(member.name).name
            if not dest.exists():
                with archive.extractfile(member) as stream,dest.open('wb') as output:
                    while block:=stream.read(4*1024*1024):output.write(block)
            record={'run_id':row['run_id'],'scenario':scenario,'camera':camera,'clip_key':key,
                    'rgb_shard':row['shard_path_in_repo'],'expected_video_bytes':row['size'],
                    'annotation':{'path':str(dest),'source_url':url,'tar_member':member.name,
                                  'byte_offset':member.offset_data,'bytes':member.size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}}
            result.append(record)
            with lock:
                records.append(record);save('download_manifest.json',records)
            print('ANNOTATION',scenario,shard,len(result),key,flush=True)
            if len(result)>=count:break
    if len(result)<count:print('SHORT SHARD',scenario,shard,len(result),flush=True)
    return result

def video_job(job):
    shard,rows=job; targets={r['clip_key']+'.rgb.mp4':r for r in rows};found=set();url=URL+shard
    with tarfile.open(fileobj=Remote(url),mode='r:') as archive:
        for member in archive:
            name=Path(member.name).name
            if name not in targets:continue
            r=targets[name];dest=BASE/'videos'/name
            if member.size!=r['expected_video_bytes']:raise RuntimeError('Pinned metadata size mismatch')
            if not dest.exists():
                temp=dest.with_suffix('.part')
                with archive.extractfile(member) as stream,temp.open('wb') as output:
                    while block:=stream.read(4*1024*1024):output.write(block)
                temp.replace(dest)
            r['video']={'path':str(dest),'source_url':url,'tar_member':member.name,'byte_offset':member.offset_data,
                        'bytes':member.size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
            found.add(name)
            with lock:save('download_manifest.json',records)
            print('VIDEO',len(found),len(targets),name,flush=True)
            if found==set(targets):break
    if found!=set(targets):raise RuntimeError('Missing selected RGB members')

if __name__=='__main__':
    # Fetch 72 candidates to allow annotation integrity exclusions while retaining >=60.
    jobs=[('nearmiss',s,3) for s in range(13,29)]+[('box_pickup',s,2) for s in range(10,22)]
    if not args.videos_only:
        with futures.ThreadPoolExecutor(max_workers=6) as pool:
            for f in futures.as_completed([pool.submit(annotation_job,j) for j in jobs]):f.result()
    runs=[r['run_id'] for r in records]
    overlap=sorted(set(runs)&excluded)
    save('overlap_audit.json',{'dataset_id':SOURCE['dataset_id'],'dataset_revision':SOURCE['dataset_revision'],
         'excluded_runs_count':len(excluded),'selected_runs_count':len(runs),'unique_selected_runs':len(set(runs)),
         'overlap_count':len(overlap),'overlap':overlap,'selected_runs':runs})
    assert not overlap and len(runs)==len(set(runs))
    grouped={}
    for r in records:grouped.setdefault(r['rgb_shard'],[]).append(r)
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        for f in futures.as_completed([pool.submit(video_job,j) for j in grouped.items()]):f.result()
    save('download_manifest.json',records)
    print('FETCH COMPLETE',len(records),reserved,flush=True)
