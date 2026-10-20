#!/usr/bin/env python3
"""Fetch pinned individual evaluation videos or annotations by bounded HTTP byte ranges.

No whole tar is downloaded. This is an explicit network operation; inference
and cached result scoring do not call this downloader.
"""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/evaluation/sources.json');p.add_argument('--cohort',choices=['primary','routine'],default='primary');p.add_argument('--asset',choices=['video','annotation'],default='video');p.add_argument('--output',default='data/evaluation-videos');p.add_argument('--max-mib',type=int,default=1024);a=p.parse_args()
cap=a.max_mib*1024*1024;reserved=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
class Remote(io.RawIOBase):
    def __init__(self,url):self.url=url;self.pos=0;self.session=requests.Session()
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        if whence==0:self.pos=n
        elif whence==1:self.pos+=n
        else:raise ValueError('Only absolute/relative seeks supported')
        return self.pos
    def read(self,n=-1):
        global reserved
        if n<0 or n>8*1024*1024:raise ValueError('Maximum range response is8MiB')
        if not n:return b''
        if reserved+n>cap:raise RuntimeError('Download byte budget exceeded')
        reserved+=n
        with self.session.get(self.url+f'?range_start={self.pos}',headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},stream=True,timeout=120) as response:
            expected=f'bytes {self.pos}-{self.pos+n-1}/'
            if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(expected):
                raise RuntimeError('Refusing unexpected/non-range response without consuming body')
            content=response.raw.read(n)
        if len(content)!=n:raise RuntimeError('Incomplete range response')
        self.pos+=n;return content

rows=json.loads(Path(a.manifest).read_text())[a.cohort];by_url={}
for row in rows:
    item=row.get(a.asset)
    if not item or item.get('status')=='missing':
        print(f'SKIP missing {a.asset}: '+row['video']['filename'],flush=True);continue
    by_url.setdefault(item['source_url'],{})[item['tar_member']]=item
for url,targets in by_url.items():
    reader=Remote(url);found=set()
    with tarfile.open(fileobj=reader,mode='r:') as archive:
        for member in archive:
            if member.name not in targets:continue
            target=targets[member.name];destination=out/target['filename']
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==target['sha256']:
                found.add(member.name)
            else:
                temporary=destination.with_suffix(destination.suffix+'.part');digest=hashlib.sha256()
                with archive.extractfile(member) as source,temporary.open('wb') as sink:
                    while chunk:=source.read(4*1024*1024):sink.write(chunk);digest.update(chunk)
                if digest.hexdigest()!=target['sha256'] or temporary.stat().st_size!=target['bytes']:
                    raise RuntimeError(f'Hash/size mismatch: {target["filename"]}')
                temporary.replace(destination);found.add(member.name)
            print(f'{len(found)}/{len(targets)} {destination}',flush=True)
            if len(found)==len(targets):break
    if found!=set(targets):raise RuntimeError('Archive ended before all selected assets were found')
print(f'Completed. Reserved response bytes: {reserved}; cap: {cap}')
