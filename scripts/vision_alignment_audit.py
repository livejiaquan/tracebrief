#!/usr/bin/env python3
"""Render annotation-row/video-frame alignment for manual data integrity review."""
import argparse,hashlib,json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from vision_evaluate import boxes
p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--annotations',required=True);p.add_argument('--frames',default='0,90,150,197');p.add_argument('--output',required=True);a=p.parse_args()
cv2.setNumThreads(1)
rows=[json.loads(x) for x in Path(a.annotations).read_text().splitlines()];indices=[int(x) for x in a.frames.split(',')];cap=cv2.VideoCapture(a.video)
if not cap.isOpened():raise ValueError('Cannot decode video')
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));canvas=Image.new('RGB',(1280,((len(indices)+1)//2)*390),'white');draw=ImageDraw.Draw(canvas)
for i,idx in enumerate(indices):
 if idx<0 or idx>=len(rows):raise ValueError(f'No annotation row {idx}')
 cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read()
 if not ok:raise ValueError(f'No decoded frame {idx}')
 for d in boxes(rows[idx]):
  x,y,z,w=map(int,d['bbox_xyxy']);cv2.rectangle(im,(x,y),(z,w),(0,255,0),3)
 im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im.thumbnail((640,360));canvas.paste(im,(i%2*640,i//2*390+30));draw.text((i%2*640+4,i//2*390+5),f'GROUND TRUTH AUDIT ONLY — row/frame {idx}',fill='black')
cap.release();dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
dest.with_suffix('.json').write_text(json.dumps({'scope':'Manual annotation integrity review. Row index is not assumed to establish correct temporal alignment. No inference or time rescaling.','video_sha256':sha(a.video),'annotation_sha256':sha(a.annotations),'annotation_rows':len(rows),'video_frames':count,'inspected_indices':indices,'output_sha256':sha(dest)},indent=2))
