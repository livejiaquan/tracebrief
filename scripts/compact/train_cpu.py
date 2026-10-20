"""Official YOLOX-tiny architecture/SimOTA loss, explicit CPU finetuning."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.ops import batched_nms,box_iou

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'YOLOX'))
from yolox.models import YOLOX,YOLOPAFPN,YOLOXHead
from yolox.data import TrainTransform,ValTransform
from yolox.utils import ModelEMA

def model():
    result=YOLOX(YOLOPAFPN(.33,.375),YOLOXHead(2,.375))
    for m in result.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eps=1e-3;m.momentum=.03
    result.head.initialize_biases(.01)
    return result

class CocoFrames(Dataset):
    def __init__(self,split):
        coco=json.loads((BASE/'coco'/f'{split}.json').read_text())
        self.images=coco['images'];self.targets=defaultdict(list);self.train=split=='train'
        self.transform=TrainTransform(max_labels=80,flip_prob=.5,hsv_prob=.5) if self.train else ValTransform()
        for a in coco['annotations']:
            x,y,w,h=a['bbox'];self.targets[a['image_id']].append([x,y,x+w,y+h,a['category_id']-1])
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        row=self.images[i];im=cv2.imread(str(BASE/'coco'/row['file_name']))
        if im is None:raise ValueError(row['file_name'])
        targets=np.array(self.targets[row['id']],dtype=np.float32).reshape(-1,5)
        if self.train:
            im,labels=self.transform(im,targets,(416,416));return torch.from_numpy(im),torch.from_numpy(labels)
        im,_=self.transform(im,None,(416,416));return torch.from_numpy(im),targets[:,:4]*(416/640),targets[:,4].astype(np.int64)

def predictions(raw,confidence=.01):
    scores,classes=raw[:,5:].max(1);scores=scores*raw[:,4]
    xy=raw[:,:4].clone();xy[:,:2]=raw[:,:2]-raw[:,2:4]/2;xy[:,2:4]=raw[:,:2]+raw[:,2:4]/2
    keep=torch.isfinite(raw).all(1)&(scores>=confidence)&(xy[:,2]>xy[:,0])&(xy[:,3]>xy[:,1])
    xy,scores,classes=xy[keep],scores[keep],classes[keep]
    keep=batched_nms(xy,scores,classes,.45)[:200]
    return xy[keep],scores[keep],classes[keep]

@torch.inference_mode()
def validate(net,data):
    net.eval(); outcomes=defaultdict(list);total=[0,0]
    for i in range(len(data)):
        image,gold,labels=data[i];boxes,scores,classes=predictions(net(image.unsqueeze(0))[0])
        gold=torch.from_numpy(gold)
        for cls in range(2):
            target=gold[labels==cls];total[cls]+=len(target);used=set()
            for b,s,c in zip(boxes,scores,classes):
                if int(c)!=cls:continue
                overlaps=box_iou(b.reshape(1,4),target)[0]
                for idx in used:overlaps[idx]=-1
                if len(overlaps):value,idx=overlaps.max(0);tp=bool(value>=.5)
                else:tp=False
                if tp:used.add(int(idx))
                outcomes[cls].append((float(s),int(tp)))
    choices=[]
    for threshold in [.01,.02,.03,.05,.075,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]:
        metrics={}
        for cls,name in enumerate(['person','forklift']):
            kept=[tp for s,tp in outcomes[cls] if s>=threshold];tp=sum(kept);fp=len(kept)-tp;fn=total[cls]-tp
            p=tp/(tp+fp) if tp+fp else 0;r=tp/total[cls] if total[cls] else 0
            metrics[name]={'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}
        choices.append({'confidence':threshold,'detection':metrics})
    qualified=[c for c in choices if all(m['recall']>=.9 for m in c['detection'].values())]
    selected=max(qualified or choices,key=lambda c:sum(m['f1'] for m in c['detection'].values()))
    selected['both_recall_ge_0_9']=bool(qualified)
    return {'selected':selected,'threshold_sweep':choices,'n_frames':len(data),'gold_counts':total}

def rank(val):
    chosen=val['selected'];ms=list(chosen['detection'].values())
    return (chosen['both_recall_ge_0_9'],sum(m['f1'] for m in ms))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=35);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--budget-hours',type=float,default=3);p.add_argument('--resume');a=p.parse_args()
    torch.set_num_threads(a.threads);cv2.setNumThreads(1);torch.manual_seed(20261012);np.random.seed(20261012);random.seed(20261012)
    train=CocoFrames('train');val=CocoFrames('val');net=model()
    init=torch.load(BASE/'yolox_tiny_coco.pth',map_location='cpu',weights_only=True)['model']
    partial={k:v for k,v in init.items() if k in net.state_dict() and v.shape==net.state_dict()[k].shape}
    net.load_state_dict(partial,strict=False)
    # Preserve the COCO person row and use COCO truck as the starting vehicle feature.
    with torch.no_grad():
        for i in range(3):
            for field in ['weight','bias']:
                getattr(net.head.cls_preds[i],field).copy_(init[f'head.cls_preds.{i}.{field}'][[0,7]])
    if a.resume:net.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=True)['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.0005)
    ema=ModelEMA(net,decay=.99)
    # Restricted runtime disallows multiprocessing tensor-sharing sockets.
    loader=DataLoader(train,batch_size=a.batch,shuffle=True,num_workers=0,drop_last=False)
    config={'architecture':'YOLOX-tiny','depth':.33,'width':.375,'num_classes':2,'input_size':[416,416],'optimizer':'AdamW',
            'base_lr':.0005,'weight_decay':.0005,'batch_size':a.batch,'threads':a.threads,'epochs':a.epochs,
            'mosaic':False,'mixup':False,'flip_prob':.5,'hsv_prob':.5,'ema_decay':.99,'seed':20261012,'data_workers':0,
            'initial_person_coco_class':0,'initial_forklift_coco_class':7,'validation_iou':.5,'nms_iou':.45,
            'pretrained_sha256':hashlib.sha256((BASE/'yolox_tiny_coco.pth').read_bytes()).hexdigest(),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (BASE/'training_config.json').write_text(json.dumps(config,indent=2));history=[];best=None;qualified_epochs=0;start=time.monotonic();steps=0
    for epoch in range(1,a.epochs+1):
        net.train();losses=[];epoch_start=time.monotonic()
        net.head.use_l1=epoch>=max(5,a.epochs-5)
        for batch,(images,labels) in enumerate(loader):
            steps+=1;warm=min(1,steps/max(1,len(loader)));cos=.1+.9*(1+math.cos(math.pi*(epoch-1)/a.epochs))/2
            for g in opt.param_groups:g['lr']=.0005*warm*cos
            opt.zero_grad(set_to_none=True);result=net(images,labels);loss=result['total_loss']
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss {epoch}/{batch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step();ema.update(net);losses.append(float(loss.detach()))
            if batch%30==0:print('TRAIN',epoch,batch,len(loader),'loss',losses[-1],'elapsed_s',round(time.monotonic()-start,1),flush=True)
            if time.monotonic()-start>a.budget_hours*3600:break
        result={'epoch':epoch,'mean_loss':sum(losses)/len(losses),'train_wall_s':time.monotonic()-epoch_start}
        evaluation=validate(ema.ema,val);result['validation']=evaluation;history.append(result)
        torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'config':config},BASE/'latest.pth')
        if best is None or rank(evaluation)>rank(best):
            best=evaluation;torch.save({'model':ema.ema.state_dict(),'epoch':epoch,'config':config},BASE/'best.pth');(BASE/'best_validation.json').write_text(json.dumps(evaluation,indent=2))
        (BASE/'training_history.json').write_text(json.dumps(history,indent=2))
        print('VAL',epoch,json.dumps(evaluation['selected']),flush=True)
        with (BASE/'REPORT.md').open('a') as f:
            f.write(f'\n### Training epoch {epoch} ({time.strftime("%H:%M UTC",time.gmtime())})\n\nLoss {result["mean_loss"]:.4f}; validation-selected confidence {evaluation["selected"]["confidence"]}; '+
                    '; '.join(f'{name} P={m["precision"]:.3f}, R={m["recall"]:.3f}' for name,m in evaluation['selected']['detection'].items())+f'. Elapsed {time.monotonic()-start:.0f} s; no inference on the evaluation clips yet.\n')
        qualified_epochs=qualified_epochs+1 if evaluation['selected']['both_recall_ge_0_9'] else 0
        if qualified_epochs>=4 or time.monotonic()-start>a.budget_hours*3600:break
    print('TRAIN COMPLETE',json.dumps(best['selected']),flush=True)
