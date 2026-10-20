#!/usr/bin/env python3
"""Run-cluster bootstrap intervals for the frozen synthetic proxy evaluation."""
import argparse,json,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evaluation');p.add_argument('--output',required=True);a=p.parse_args()
j=json.loads(Path(a.evaluation).read_text());runs=j['runs'];rng=random.Random(20261004);values={'precision':[],'recall':[]};pair_values={'precision':[],'recall':[]}
for _ in range(2000):
    selected=[rng.choice(runs) for _ in runs]
    sample=[r['candidate'] for r in selected]
    tp=sum(x['tp'] for x in sample);fp=sum(x['fp'] for x in sample);fn=sum(x['fn'] for x in sample)
    if tp+fp:values['precision'].append(tp/(tp+fp))
    if tp+fn:values['recall'].append(tp/(tp+fn))
    pair_sample=[r['pair_evidence'] for r in selected];ptp=sum(x['tp'] for x in pair_sample);pfp=sum(x['fp'] for x in pair_sample);pfn=sum(x['fn'] for x in pair_sample)
    if ptp+pfp:pair_values['precision'].append(ptp/(ptp+pfp))
    if ptp+pfn:pair_values['recall'].append(ptp/(ptp+pfn))
def quantile(v,p):
    v=sorted(v);ix=(len(v)-1)*p;lo=int(ix);hi=min(len(v)-1,lo+1);return v[lo]+(v[hi]-v[lo])*(ix-lo)
result={'n_runs':len(runs),'method':'2000 percentile bootstrap replicates resampling entire runs; seed20261004. Same generator/scene family; does not estimate real-world generalization.',
'candidate_ci95':{k:[quantile(v,.025),quantile(v,.975)] for k,v in values.items()},
'pair_evidence_ci95':{k:[quantile(v,.025),quantile(v,.975)] for k,v in pair_values.items()},
'median_video_wall_s':statistics.median(r['wall_s'] for r in runs),
'min_video_wall_s':min(r['wall_s'] for r in runs),'max_video_wall_s':max(r['wall_s'] for r in runs),
'runs_with_positive_sampled_frames':sum(r['candidate']['tp']+r['candidate']['fn']>0 for r in runs),
'runs_with_negative_sampled_frames':sum(r['candidate']['tn']+r['candidate']['fp']>0 for r in runs),
'fully_negative_runs':sum(r['candidate']['tp']+r['candidate']['fn']==0 for r in runs),
'candidate_groups':sum(r['events'] for r in runs)}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
