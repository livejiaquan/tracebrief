"""Reject annotation/video misalignment; sample 2fps and split by run."""
import concurrent.futures as futures
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import cv2

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evaluator',Path('tracebrief/scripts/vision_evaluate.py'))
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

def extract(record):
    cv2.setNumThreads(1)
    rows=[json.loads(line) for line in Path(record['annotation']['path']).read_text().splitlines()]
    cap=cv2.VideoCapture(record['video']['path'])
    fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info={**record,'video_frames':total,'annotation_rows':len(rows),'fps':fps,'width':w,'height':h}
    if fps<=0 or total!=len(rows) or (w,h)!=(1920,1080):
        cap.release();return {'rejected':info,'reason':'annotation_rows/frame_count mismatch or unexpected source geometry'}
    stride=max(1,round(fps/2));frames=[];i=0
    output=BASE/'coco'/'images';output.mkdir(parents=True,exist_ok=True)
    while cap.grab():
        if i%stride==0:
            ok,image=cap.retrieve()
            if not ok:raise RuntimeError(f'Decode failed {record["run_id"]}/{i}')
            boxes=ev.boxes(rows[i])
            name=record['run_id']+f'_{i:06d}.jpg'
            # Keep 640px images; coordinates are scaled explicitly for COCO.
            image=cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(output/name),image,[cv2.IMWRITE_JPEG_QUALITY,92]):raise RuntimeError('JPEG write failed')
            frames.append({'file_name':'images/'+name,'width':640,'height':360,'run_id':record['run_id'],
                           'frame_index':i,'time_s':i/fps,'boxes':boxes})
        i+=1
    cap.release()
    if i!=total:raise RuntimeError('Incomplete video decoding')
    return {'record':info,'frames':frames}

if __name__=='__main__':
    manifest=json.loads((BASE/'download_manifest.json').read_text());rejected=[];accepted=[]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(extract,manifest):
            if 'rejected' in result:rejected.append(result)
            else:accepted.append(result)
            print('EXTRACT',len(accepted),len(rejected),flush=True)
    (BASE/'annotation_integrity.json').write_text(json.dumps({'accepted':[r['record'] for r in accepted],'rejected':rejected},indent=2))
    splits={}
    for scenario in ['nearmiss','box_pickup']:
        cohort=sorted([r for r in accepted if r['record']['scenario']==scenario],key=lambda r:hashlib.sha256(('compact-20261012-'+r['record']['run_id']).encode()).hexdigest())
        nval=max(2,round(len(cohort)*.2))
        for k,r in enumerate(cohort):splits[r['record']['run_id']]='val' if k<nval else 'train'
    summary={'sample_fps':2,'stored_size':[640,360],'seed':'compact-20261012','runs':splits,'counts':dict(Counter(splits.values())),
             'scenario_counts':dict(Counter(r['record']['scenario'] for r in accepted)),'rejected_runs':len(rejected)}
    for split in ['train','val']:
        coco={'images':[],'annotations':[],'categories':[{'id':1,'name':'person'},{'id':2,'name':'forklift'}]}
        for result in accepted:
            if splits[result['record']['run_id']]!=split:continue
            for frame in result['frames']:
                image_id=len(coco['images'])+1
                coco['images'].append({k:v for k,v in frame.items() if k!='boxes'}|{'id':image_id})
                for box in frame['boxes']:
                    x1,y1,x2,y2=[x/3 for x in box['bbox_xyxy']]
                    coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':image_id,'category_id':1 if box['class_name']=='person' else 2,
                       'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
        (BASE/'coco'/f'{split}.json').write_text(json.dumps(coco))
        summary[split]={'frames':len(coco['images']),'boxes':len(coco['annotations']),'classes':dict(Counter(a['category_id'] for a in coco['annotations']))}
    (BASE/'split.json').write_text(json.dumps(summary,indent=2))
    prior=json.loads((BASE/'overlap_audit.json').read_text());excluded=set(json.loads((BASE/'excluded_runs.json').read_text()))
    train={r for r,s in splits.items() if s=='train'};val={r for r,s in splits.items() if s=='val'}
    prior.update({'accepted_runs_count':len(splits),'train_val_overlap':sorted(train&val),'accepted_excluded_overlap':sorted(set(splits)&excluded)})
    assert not train&val and not set(splits)&excluded
    (BASE/'overlap_audit.json').write_text(json.dumps(prior,indent=2))
    print('COCO COMPLETE',summary,flush=True)
