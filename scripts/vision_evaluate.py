#!/usr/bin/env python3
"""Evaluate saved predictions against simulator annotations (inference independent)."""
import argparse,json,math
from pathlib import Path
from collections import Counter

def iou(a,b):
    z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z
    return z/union if union>0 else 0

def boxes(row):
    out=[]
    for a in row['agents'].values():
        group=a.get('metro_agent_data',{}).get('agent_group')
        label={'Worker':'person','forklift':'forklift'}.get(group)
        b=a.get('annotators',{}).get('bounding_box_2d_tight_fast')
        if label and b:
            xy=[max(0,min(1920 if i%2==0 else 1080,float(b[k]))) for i,k in enumerate(['x_min','y_min','x_max','y_max'])]
            if xy[2]>xy[0] and xy[3]>xy[1]:out.append({'class_name':label,'bbox_xyxy':xy})
    return out

def near(gs,threshold):
    for p in [x['bbox_xyxy'] for x in gs if x['class_name']=='person']:
        for f in [x['bbox_xyxy'] for x in gs if x['class_name']=='forklift']:
            gap=math.hypot(max(p[0]-f[2],f[0]-p[2],0),max(p[1]-f[3],f[1]-p[3],0))
            if gap/max(1,p[3]-p[1])<=threshold:return True
    return False

def metrics(c):
    d=dict(c);d['precision']=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else None;d['recall']=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None
    return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--annotations');p.add_argument('--compact-gold');p.add_argument('--output',required=True);p.add_argument('--coverage-exclusions',help='Explicit JSON excluded_runs mapping for annotation integrity failures; retain predictions');a=p.parse_args()
    if not (a.annotations or a.compact_gold):p.error('Provide --annotations or --compact-gold')
    compact=json.loads(Path(a.compact_gold).read_text())['videos'] if a.compact_gold else None
    coverage=json.loads(Path(a.coverage_exclusions).read_text()) if a.coverage_exclusions else {}
    excluded=coverage.get('excluded_runs',{});unscorable=[]
    counts={k:Counter(tp=0,fp=0,fn=0) for k in ['person','forklift']};candidate=Counter(tp=0,fp=0,fn=0,tn=0);pair_counts=Counter(tp=0,fp=0,fn=0);runs=[];failure=[]
    for pred in sorted(Path(a.predictions).glob('*/analysis.json')):
        r=json.loads(pred.read_text());video_name=Path(r['video']['path']).name
        if video_name in excluded:
            unscorable.append({'video':video_name,'sampled_frames':len(r['observations']),'reason':excluded[video_name],'events':len(r['events']),'wall_s':r['timing']['wall_s']});continue
        if compact is None:
            key=video_name.replace('.rgb.mp4','.object_detection.jsonl');matches=list(Path(a.annotations).glob('*'+key))
            if len(matches)!=1:raise ValueError(f'Expected exactly one annotation for {video_name}, got {len(matches)}')
            gold=[json.loads(x) for x in matches[0].read_text().splitlines()]
        cc=Counter(tp=0,fp=0,fn=0,tn=0);pc=Counter(tp=0,fp=0,fn=0);per={k:Counter(tp=0,fp=0,fn=0) for k in counts}
        for obs in r['observations']:
            gs=compact[video_name]['frames'][str(obs['frame_index'])] if compact is not None else boxes(gold[obs['frame_index']]);gt=near(gs,r['settings']['gap_ratio']);pr=bool(obs['proximity_pairs']);key2=('tp' if gt else 'fp') if pr else ('fn' if gt else 'tn');candidate[key2]+=1;cc[key2]+=1
            if gt!=pr:failure.append({'video':Path(r['video']['path']).name,'frame_index':obs['frame_index'],'type':key2,'gold_boxes':gs,'detections':obs['detections']})
            # Supplementary stricter audit: a candidate pair must localize BOTH gold actors.
            gold_pairs=[(p['bbox_xyxy'],f['bbox_xyxy']) for p in gs for f in gs
                        if p['class_name']=='person' and f['class_name']=='forklift'
                        and near([p,f],r['settings']['gap_ratio'])]
            used_pairs=set()
            for pair in sorted(obs['proximity_pairs'],key=lambda q:-q['confidence']):
                options=[(min(iou(pair['person_bbox'],p),iou(pair['forklift_bbox'],f)),i)
                         for i,(p,f) in enumerate(gold_pairs) if i not in used_pairs]
                overlap,idx=max(options,default=(0,-1));kind='tp' if overlap>=.5 else 'fp'
                pair_counts[kind]+=1;pc[kind]+=1
                if kind=='tp':used_pairs.add(idx)
            missed=len(gold_pairs)-len(used_pairs);pair_counts['fn']+=missed;pc['fn']+=missed
            for cls in counts:
                targets=[x['bbox_xyxy'] for x in gs if x['class_name']==cls];used=set()
                for d in sorted([d for d in obs['detections'] if d['class_name']==cls],key=lambda d:-d['confidence']):
                    options=[(iou(d['bbox_xyxy'],b),i) for i,b in enumerate(targets) if i not in used]
                    score,idx=max(options,default=(0,-1))
                    k='tp' if score>=.5 else 'fp';counts[cls][k]+=1;per[cls][k]+=1
                    if k=='tp':used.add(idx)
                n=len(targets)-len(used);counts[cls]['fn']+=n;per[cls]['fn']+=n
        runs.append({'video':Path(r['video']['path']).name,'sampled_frames':len(r['observations']),'events':len(r['events']),'candidate':metrics(cc),'pair_evidence':metrics(pc),'detection':{k:metrics(v) for k,v in per.items()},'wall_s':r['timing']['wall_s']})
    result={'scope':'Synthetic ceiling_00 sampled-frame image-space proxy; not near-miss/collision safety accuracy','n_runs':len(runs),'n_frames':sum(x['sampled_frames'] for x in runs),'candidate':metrics(candidate),'pair_evidence':metrics(pair_counts),'pair_evidence_scope':'Supplementary post-freeze audit: each predicted proximity pair requires both actor boxes IoU>=0.50 with a gold proximity pair. Not physical safety truth.','detection':{k:metrics(v) for k,v in counts.items()},'runs':runs,'failures':failure}
    if a.coverage_exclusions:
        if coverage.get('pair_evidence_scope'):result['pair_evidence_scope']=coverage['pair_evidence_scope']
        if set(excluded)!={x['video'] for x in unscorable}:raise ValueError('Coverage exclusions do not match prediction inputs')
        result.update({'n_inferred_runs':len(runs)+len(unscorable),'n_sampled_frames':sum(x['sampled_frames'] for x in runs+unscorable),'n_scored_runs':len(runs),'n_scored_frames':sum(x['sampled_frames'] for x in runs),'unscorable_runs':unscorable,'coverage_scope':'All predictions retained; annotation integrity exclusions only, no replacements or outcome-based filtering.'})
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['runs','failures']},indent=2))
if __name__=='__main__':main()
