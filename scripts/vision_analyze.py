#!/usr/bin/env python3
"""Run the local detector on a full video; model is never downloaded implicitly."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tracebrief.vision import analyze_video
p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--settings',default=str(Path(__file__).resolve().parents[1]/'config/vision.json'));p.add_argument('--output',required=True);p.add_argument('--model-dir');a=p.parse_args()
s=json.loads(Path(a.settings).read_text());s['output_dir']=str(Path(a.output).parent/'frames')
if a.model_dir:s['model_dir']=a.model_dir
result=analyze_video(a.video,s);Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({'output':a.output,'events':len(result['events']),'timing':result['timing']}))
