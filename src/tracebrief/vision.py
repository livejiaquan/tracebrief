"""Local RGB-only candidate discovery. Never consumes simulator annotations."""
from __future__ import annotations
import hashlib
import math
import os
import time
from pathlib import Path

DEFAULTS = {'sample_fps': 1.0, 'confidence': 0.35, 'text_threshold': 0.25, 'nms_iou': 0.45,
            'gap_ratio': 0.30, 'min_samples': 2, 'merge_gap_s': 2.1,
            'threads': 4, 'model_id': 'IDEA-Research/grounding-dino-tiny', 'model_revision': 'a2bb814dd30d776dcf7e30523b00659f4f141c71'}


def _iou(a, b):
    area = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area / union if union > 0 else 0.


def _gap(a,b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2],0), max(a[1]-b[3],b[1]-a[3],0))


def _pairs(detections, ratio):
    pairs=[]
    for p in [d for d in detections if d['class_name']=='person']:
        for f in [d for d in detections if d['class_name']=='forklift']:
            gap=_gap(p['bbox_xyxy'],f['bbox_xyxy'])
            norm=gap/max(1,p['bbox_xyxy'][3]-p['bbox_xyxy'][1])
            if norm<=ratio:
                pairs.append({'person_bbox':p['bbox_xyxy'],'forklift_bbox':f['bbox_xyxy'],
                              'gap_px':round(gap,2),'gap_person_height_ratio':round(norm,4),
                              'confidence':round(min(p['confidence'],f['confidence']),4)})
    return pairs


def analyze_video(path, settings=None):
    """Return observations and review candidates from all sampled frames of a video.

    Settings: model_dir (downloaded HF snapshot), output_dir (evidence JPEGs),
    sample_fps, confidence, gap_ratio, min_samples, merge_gap_s, threads,
    processor_size (optional shortest_edge/longest_edge dictionary).
    No network fetch at inference: a model must first be downloaded explicitly.
    """
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    cfg={**DEFAULTS, **(settings or {})}
    for field in ['sample_fps','confidence','gap_ratio','merge_gap_s']:
        cfg[field]=float(cfg[field])
    if not 0<cfg['sample_fps']<=30 or not 0<cfg['confidence']<1 or cfg['gap_ratio']<0:
        raise ValueError('Invalid sampling, confidence or gap settings')
    torch.set_num_threads(max(1,min(4,int(cfg['threads']))))
    cv2.setNumThreads(max(1,min(4,int(cfg['threads']))))
    model_dir=cfg.get('model_dir') or os.environ.get('TRACEBRIEF_MODEL_DIR')
    if not model_dir:
        raise FileNotFoundError('Set TRACEBRIEF_MODEL_DIR to a downloaded model snapshot; see docs/VISION.md')
    start=time.monotonic()
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    if cfg.get('processor_size'):
        size=cfg['processor_size']
        if set(size) != {'shortest_edge','longest_edge'} or not 32 <= int(size['shortest_edge']) <= int(size['longest_edge']):
            raise ValueError('processor_size requires valid shortest_edge and longest_edge')
        processor.image_processor.size={k:int(v) for k,v in size.items()}
    model=AutoModelForZeroShotObjectDetection.from_pretrained(model_dir,local_files_only=True).eval()
    is_dino=model.config.model_type == 'grounding-dino'
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError('Cannot decode video')
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or total<=0: cap.release();raise ValueError('Video has invalid duration')
    stride=max(1,round(fps/cfg['sample_fps'])); observations=[]; candidates=[]
    output_dir=Path(cfg.get('output_dir','.') ).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    frame_index=0
    try:
        while True:
            ok=cap.grab()
            if not ok: break
            if frame_index%stride:
                frame_index+=1;continue
            ok,im=cap.retrieve()
            if not ok: raise ValueError(f'Cannot decode frame {frame_index}')
            inputs=processor(text='a person. a forklift.' if is_dino else [['a person','a forklift']],images=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),return_tensors='pt')
            with torch.inference_mode(): raw=model(**inputs)
            if is_dino:
                pred=processor.post_process_grounded_object_detection(raw,inputs.input_ids,target_sizes=[(height,width)],threshold=cfg['confidence'],text_threshold=cfg['text_threshold'])[0]
                labels=pred['text_labels']
            else:
                pred=processor.post_process_grounded_object_detection(outputs=raw,target_sizes=torch.tensor([[height,width]]),threshold=cfg['confidence'])[0]
                labels=pred['labels'].tolist()
            det=[]
            for box,score,label in sorted(zip(pred['boxes'].tolist(),pred['scores'].tolist(),labels),key=lambda x:-x[1]):
                if is_dino:
                    words=label.lower().split()
                    if ('person' in words) == ('forklift' in words):continue
                    name='person' if 'person' in words else 'forklift'
                else: name=['person','forklift'][label]
                box=[round(max(0,min(width if i%2==0 else height,x)),2) for i,x in enumerate(box)]
                if box[2]<=box[0] or box[3]<=box[1]:continue
                if any(d['class_name']==name and _iou(box,d['bbox_xyxy'])>cfg['nms_iou'] for d in det):continue
                det.append({'class_name':name,'confidence':round(score,4),'bbox_xyxy':box})
            pairs=_pairs(det,cfg['gap_ratio']); sec=round(frame_index/fps,3)
            obs={'frame_index':frame_index,'time_s':sec,'detections':det,'proximity_pairs':pairs}
            observations.append(obs)
            if pairs:
                image_path=output_dir/f'frame_{frame_index:06d}.jpg'
                for d in det:
                    a,b,c,d_y=map(int,d['bbox_xyxy']); color=(245,175,45) if d['class_name']=='person' else (50,190,100)
                    cv2.rectangle(im,(a,b),(c,d_y),color,2)
                    cv2.putText(im,f"{d['class_name']} {d['confidence']:.2f}",(a,max(18,b-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                cv2.imwrite(str(image_path),im)
                obs['image_path']=str(image_path); candidates.append(obs)
            frame_index+=1
    finally: cap.release()
    groups=[]
    for obs in candidates:
        if not groups or obs['time_s']-groups[-1][-1]['time_s']>cfg['merge_gap_s']:groups.append([])
        groups[-1].append(obs)
    events=[]
    for group in groups:
        if len(group)<int(cfg['min_samples']): continue
        peak=min(group,key=lambda o:min(p['gap_person_height_ratio'] for p in o['proximity_pairs']))
        events.append({'id':f'candidate-{len(events)+1:03d}','kind':'person_forklift_proximity_candidate',
                       'start_s':max(0,round(group[0]['time_s']-1,3)), 'end_s':min(total/fps,round(group[-1]['time_s']+1,3)),
                       'peak_s':peak['time_s'],'evidence':[{'frame_index':o['frame_index'],'time_s':o['time_s'],'image_path':o['image_path'],'pairs':o['proximity_pairs'],
                       'fact':'Predicted person and forklift boxes are close in image space.'} for o in group],
                       'limitations':['Image-space proximity does not establish physical distance, collision, near miss, or unsafe conduct.','Predicted object labels and boxes require visual verification.']})
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return {'schema_version':'1.0','video':{'path':str(path),'sha256':digest.hexdigest(),'duration_s':total/fps,'fps':fps,'width':width,'height':height},
            'model':{'id':cfg['model_id'],'revision':cfg.get('model_revision','local snapshot'),'backend':'Grounding DINO CPU' if is_dino else 'OWLv2 CPU'},
            'settings':{k:v for k,v in cfg.items() if k!='output_dir'},'observations':observations,'events':events,
            'timing':{'wall_s':round(time.monotonic()-start,3),'sampled_frames':len(observations)},
            'warnings':['Synthetic-domain validation only; real warehouse generalization unverified.','No detected candidates does not establish safety.','Sampling may miss short events.']}
