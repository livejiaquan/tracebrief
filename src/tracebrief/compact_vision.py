"""Local two-class YOLOX ONNX inference; RGB video only, no annotations."""
from __future__ import annotations
import hashlib
import math
import os
import time
from pathlib import Path
from .vision import DEFAULTS as VISION_DEFAULTS, _iou, _pairs

DEFAULTS={**VISION_DEFAULTS,'backend':'compact-onnx','input_size':416,
          'model_id':'YOLOX-tiny-warehouse-person-forklift','model_revision':'local frozen checkpoint'}


def preprocess(image,input_size):
    """Official YOLOX non-legacy BGR 0..255, top-left letterbox."""
    import cv2
    import numpy as np
    ratio=min(input_size/image.shape[0],input_size/image.shape[1])
    canvas=np.full((input_size,input_size,3),114,dtype=np.uint8)
    h,w=int(image.shape[0]*ratio),int(image.shape[1]*ratio)
    canvas[:h,:w]=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32),ratio


def decode(raw,input_size,ratio,width,height,confidence,nms_iou):
    """Decode raw grids, objectness*class probability, class-specific NMS."""
    import numpy as np
    raw=np.asarray(raw)
    if raw.ndim==3 and raw.shape[0]==1:raw=raw[0]
    grids=[];strides=[]
    for stride in (8,16,32):
        h=w=input_size//stride
        yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        grids.append(np.stack((xx,yy),axis=-1).reshape(-1,2))
        strides.append(np.full((h*w,1),stride))
    grid=np.concatenate(grids);stride=np.concatenate(strides)
    if raw.shape!=(len(grid),7):raise ValueError(f'Unexpected two-class YOLOX output shape: {raw.shape}')
    with np.errstate(over='ignore',invalid='ignore'):
        centers=(raw[:,:2]+grid)*stride
        sizes=np.exp(raw[:,2:4])*stride
        boxes=np.concatenate((centers-sizes/2,centers+sizes/2),axis=1)/ratio
    classes=raw[:,5:].argmax(axis=1);scores=raw[:,4]*raw[np.arange(len(raw)),5+classes]
    valid=np.isfinite(raw).all(axis=1)&np.isfinite(boxes).all(axis=1)&(scores>=confidence)
    detections=[]
    for index in sorted(np.flatnonzero(valid),key=lambda i:-float(scores[i])):
        name=('person','forklift')[classes[index]]
        box=[round(max(0,min(width if i%2==0 else height,float(x))),2) for i,x in enumerate(boxes[index])]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        if any(d['class_name']==name and _iou(box,d['bbox_xyxy'])>nms_iou for d in detections):continue
        detections.append({'class_name':name,'confidence':round(float(scores[index]),4),'bbox_xyxy':box})
    return detections


def group_candidates(candidates,duration,merge_gap_s,min_samples):
    groups=[]
    for obs in candidates:
        if not groups or obs['time_s']-groups[-1][-1]['time_s']>merge_gap_s:groups.append([])
        groups[-1].append(obs)
    events=[]
    for group in groups:
        if len(group)<min_samples:continue
        peak=min(group,key=lambda o:min(p['gap_person_height_ratio'] for p in o['proximity_pairs']))
        events.append({'id':f'candidate-{len(events)+1:03d}','kind':'person_forklift_proximity_candidate',
            'start_s':max(0,round(group[0]['time_s']-1,3)),'end_s':min(duration,round(group[-1]['time_s']+1,3)),
            'peak_s':peak['time_s'],'evidence':[{'frame_index':o['frame_index'],'time_s':o['time_s'],'image_path':o['image_path'],
            'pairs':o['proximity_pairs'],'fact':'Predicted person and forklift boxes are close in image space.'} for o in group],
            'limitations':['Image-space proximity does not establish physical distance, collision, near miss, or unsafe conduct.',
                           'Predicted object labels and boxes require visual verification.']})
    return events


def analyze_video(path,settings=None):
    """Return the same observation/event schema as vision.analyze_video."""
    cfg={**DEFAULTS,**(settings or {})}
    for field in ['sample_fps','confidence','gap_ratio','merge_gap_s','nms_iou']:
        cfg[field]=float(cfg[field])
        if not math.isfinite(cfg[field]):raise ValueError(f'Invalid {field}')
    if not 0<cfg['sample_fps']<=30 or not 0<cfg['confidence']<1 or cfg['gap_ratio']<0 or cfg['merge_gap_s']<0 or not 0<=cfg['nms_iou']<=1:
        raise ValueError('Invalid sampling, confidence, NMS or gap settings')
    for field in ['input_size','min_samples','threads']:
        value=float(cfg[field])
        if not math.isfinite(value) or value!=int(value) or value<1:raise ValueError(f'Invalid {field}')
        cfg[field]=int(value)
    if cfg['input_size']%32:raise ValueError('input_size must be divisible by 32')
    model_path=cfg.get('model_path') or os.environ.get('TRACEBRIEF_COMPACT_MODEL')
    if not model_path:raise FileNotFoundError('Set model_path or TRACEBRIEF_COMPACT_MODEL to the frozen local ONNX model')
    model_path=Path(model_path).expanduser()
    if not model_path.is_absolute():model_path=Path(__file__).resolve().parents[2]/model_path
    if not model_path.is_file():raise FileNotFoundError(f'Local ONNX model missing: {model_path}')
    hash_start=time.monotonic()
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    hash_s=time.monotonic()-hash_start
    if cfg.get('model_sha256') and digest!=cfg['model_sha256']:raise ValueError('ONNX model SHA256 does not match frozen settings')
    import cv2
    import onnxruntime as ort
    threads=max(1,min(4,cfg['threads']));cv2.setNumThreads(threads)
    start=time.monotonic()-hash_s;options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    model_load_s=time.monotonic()-start
    if session.get_inputs()[0].shape!=[1,3,cfg['input_size'],cfg['input_size']]:raise ValueError('ONNX input shape does not match settings')
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():cap.release();raise ValueError('Cannot decode video')
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps<=0 or total<=0:cap.release();raise ValueError('Video has invalid duration')
    stride=max(1,round(fps/cfg['sample_fps']));output_dir=Path(cfg.get('output_dir','.')).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    observations=[];candidates=[];index=0;inference_s=0
    try:
        while cap.grab():
            if index%stride:index+=1;continue
            ok,image=cap.retrieve()
            if not ok:raise ValueError(f'Cannot decode frame {index}')
            tensor,ratio=preprocess(image,cfg['input_size']);tick=time.monotonic()
            raw=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            inference_s+=time.monotonic()-tick
            det=decode(raw,cfg['input_size'],ratio,width,height,cfg['confidence'],cfg['nms_iou'])
            pairs=_pairs(det,cfg['gap_ratio'])
            obs={'frame_index':index,'time_s':round(index/fps,3),'detections':det,'proximity_pairs':pairs}
            observations.append(obs)
            if pairs:
                image_path=output_dir/f'frame_{index:06d}.jpg'
                for d in det:
                    x1,y1,x2,y2=map(int,d['bbox_xyxy']);color=(245,175,45) if d['class_name']=='person' else (50,190,100)
                    cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                    cv2.putText(image,f"{d['class_name']} {d['confidence']:.2f}",(x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                if not cv2.imwrite(str(image_path),image):raise OSError(f'Cannot write evidence: {image_path}')
                obs['image_path']=str(image_path);candidates.append(obs)
            index+=1
    finally:cap.release()
    video_digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):video_digest.update(chunk)
    wall=time.monotonic()-start
    return {'schema_version':'1.0','video':{'path':str(path),'sha256':video_digest.hexdigest(),'duration_s':total/fps,'fps':fps,'width':width,'height':height},
        'model':{'id':cfg['model_id'],'revision':cfg['model_revision'],'backend':'YOLOX ONNXRuntime CPU','sha256':digest},
        'settings':{k:v for k,v in cfg.items() if k!='output_dir'},'observations':observations,
        'events':group_candidates(candidates,total/fps,cfg['merge_gap_s'],cfg['min_samples']),
        'timing':{'wall_s':round(wall,3),'sampled_frames':len(observations),'model_load_s':round(model_load_s,6),
                  'wall_without_model_load_s':round(wall-model_load_s,6),'inference_s':round(inference_s,6),
                  'inference_ms_per_frame':round(inference_s*1000/len(observations),3) if observations else None},
        'warnings':['Synthetic-domain validation only; real warehouse generalization unverified.','No detected candidates does not establish safety.','Sampling may miss short events.']}
