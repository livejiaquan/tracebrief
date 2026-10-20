"""Held-out scoring runner: requires recorded freeze; never tunes parameters."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
APP=ROOT/'tracebrief'
SOURCE=APP/'data/evaluation/phase2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',default=sys.executable,help='Python interpreter of the vision environment')
    p.add_argument('--videos',default='data/evaluation-videos',help='directory holding the phase-2 evaluation videos')
    p.add_argument('--summary-only',action='store_true',help='skip inference and rebuild comparison.json from saved scores')
    return p.parse_args()

if __name__=='__main__':
    args=parse_args();PYTHON=Path(args.python);VIDEOS=Path(args.videos)
    freeze=json.loads((BASE/'FREEZE.json').read_text())
    assert sha(BASE/'best.pth')==freeze['checkpoint_sha256']
    assert sha(BASE/'yolox_tiny_warehouse.onnx')==freeze['onnx_sha256']
    assert sha(APP/'src/tracebrief/compact_vision.py')==freeze['compact_vision_sha256']
    # Source manifest identifies exactly the same20 clips; gold is read only now.
    cohorts=json.loads((SOURCE/'sources.json').read_text())
    gold=json.loads((SOURCE/'gold.json').read_text())['videos']
    excluded=json.loads((SOURCE/'coverage_exclusions.json').read_text())['excluded_runs']
    timings=[];routine=[]
    for fps in ([1,5] if not args.summary_only else []):
        for cohort in ['primary','routine']:
            for row in cohorts[cohort]:
                name=row['video']['filename'];out=BASE/f'fps{fps}'/cohort/Path(name).stem
                out.mkdir(parents=True,exist_ok=True)
                req={'path':str(VIDEOS/name),'settings':{**freeze['settings'],'sample_fps':fps,'output_dir':str(out/'frames')}}
                (out/'request.json').write_text(json.dumps(req,indent=2))
                print('START',fps,cohort,name,flush=True);start=time.monotonic()
                subprocess.run([str(PYTHON),str(APP/'scripts/vision_worker.py'),str(out/'request.json'),str(out/'analysis.json')],check=True)
                process_wall=time.monotonic()-start
                result=json.loads((out/'analysis.json').read_text())
                timings.append({'fps':fps,'cohort':cohort,'video':name,'duration_s':result['video']['duration_s'],
                                'process_wall_s':process_wall,**result['timing']})
                if cohort=='routine':
                    # Include a first-frame overlay even for videos with no candidate.
                    sys.path.insert(0,str(APP/'src'))
                    import cv2
                    cap=cv2.VideoCapture(str(VIDEOS/name));ok,image=cap.read();cap.release()
                    if not ok:raise ValueError('Cannot render routine preview')
                    for d in result['observations'][0]['detections']:
                        x1,y1,x2,y2=map(int,d['bbox_xyxy']);color=(245,175,45) if d['class_name']=='person' else (50,190,100)
                        cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
                        cv2.putText(image,f"{d['class_name']} {d['confidence']:.2f}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
                    cv2.imwrite(str(out/'preview.jpg'),image)
                    routine.append({'fps':fps,'video':name,'groups':len(result['events']),
                                    'candidate_frames':sum(bool(o['proximity_pairs']) for o in result['observations']),
                                    'sampled_frames':len(result['observations']),'preview':str(out/'preview.jpg'),
                                    'event_evidence':[e['image_path'] for event in result['events'] for e in event['evidence']]})
                elif fps==5:
                    aligned=BASE/'fps5_gold_aligned'/'primary'/Path(name).stem;aligned.mkdir(parents=True,exist_ok=True)
                    # Excluded unscorable run must still be present to satisfy identical coverage check.
                    if name in excluded:
                        result['observations']=[o for o in result['observations'] if o['frame_index']%30==0]
                    elif name in gold:
                        result['observations']=[o for o in result['observations'] if str(o['frame_index']) in gold[name]['frames']]
                    else:result['observations']=[o for o in result['observations'] if o['frame_index']%30==0]
                    (aligned/'analysis.json').write_text(json.dumps(result,indent=2))
                (BASE/'timings.json').write_text(json.dumps(timings,indent=2))
                (BASE/'routine_candidates.json').write_text(json.dumps({'scope':'Candidate counts only; missing parked-forklift labels preclude false-positive rate.','runs':routine},indent=2))
                print('DONE',fps,cohort,name,result['timing'],flush=True)
            if cohort=='primary':
                predictions=BASE/'fps1'/'primary' if fps==1 else BASE/'fps5_gold_aligned'/'primary'
                cmd=[str(PYTHON),str(APP/'scripts/vision_evaluate.py'),'--predictions',str(predictions),
                     '--compact-gold',str(SOURCE/'gold.json'),'--coverage-exclusions',str(SOURCE/'coverage_exclusions.json'),
                     '--output',str(BASE/f'evaluation_fps{fps}.json')]
                with (BASE/f'scoring_fps{fps}.txt').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (BASE/'REPORT.md').open('a') as stream:
                stream.write(f'\n### Evaluation progress: {fps} fps / {cohort}\n\nInference complete for {len(cohorts[cohort])} clips; raw outputs in fps{fps}/{cohort}/. Per-clip model_load, wall (with and without load), process wall and ORT ms/frame are in timings.json. '+
                             (f'Scores with the same coverage exclusions are in evaluation_fps{fps}.json; at 5 fps only gold-aligned frames are scored.' if cohort=='primary' else 'Group counts and overlay previews are in routine_candidates.json; no false-alarm rate is computed.')+'\n')
    compact=json.loads((BASE/'evaluation_fps1.json').read_text());five=json.loads((BASE/'evaluation_fps5.json').read_text())
    broader=json.loads((SOURCE/'comparator_v1/evaluation.json').read_text())
    stricter=json.loads((SOURCE/'evaluation.json').read_text())
    assert [broader['pair_evidence'][k] for k in ['tp','fp','fn']]==[55,6,4]
    assert [stricter['pair_evidence'][k] for k in ['tp','fp','fn']]==[37,4,22]
    comparison={}
    for label,result in [('Broader',broader),('Stricter',stricter),('YOLOX-tiny 1fps',compact),('YOLOX-tiny 5fps (gold aligned)',five)]:
        assert result['n_runs']==9 and result['n_frames']==90
        comparison[label]={'pair_evidence':result['pair_evidence'],'detection':result['detection'],'n_runs':9,'n_frames':90,
                           'median_wall_s':statistics.median(r['wall_s'] for r in result['runs'])}
    (BASE/'comparison.json').write_text(json.dumps(comparison,indent=2))
    print('EVALUATION COMPLETE',json.dumps(comparison),flush=True)
