"""Export fixed two-class ONNX and freeze BEFORE any held-out inference."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from train_cpu import BASE,model,CocoFrames

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',default='weights/yolox-tiny-warehouse.onnx',help='model_path written to the frozen settings')
    args=p.parse_args()
    torch.set_num_threads(4)
    selected=json.loads((BASE/'best_validation.json').read_text())
    checkpoint=torch.load(BASE/'best.pth',map_location='cpu',weights_only=True)
    net=model();net.load_state_dict(checkpoint['model']);net.eval();net.head.decode_in_inference=False
    target=BASE/'yolox_tiny_warehouse.onnx'
    torch.onnx.export(net,torch.zeros(1,3,416,416),str(target),opset_version=11,dynamo=False,
                      input_names=['images'],output_names=['raw_predictions'])
    graph=onnx.load(str(target));onnx.checker.check_model(graph)
    meta={'class_names':'person,forklift','input_format':'BGR float32 0..255 top-left letterbox 114',
          'output_format':'YOLOX raw grids; xy offsets, log-wh, sigmoid objectness and class probabilities',
          'checkpoint_sha256':sha(BASE/'best.pth'),'source_commit':'6ddff4824372906469a7fae2dc3206c7aa4bbaee'}
    for k,v in meta.items():item=graph.metadata_props.add();item.key=k;item.value=v
    onnx.save(graph,str(target))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),sess_options=opts,providers=['CPUExecutionProvider'])
    data=CocoFrames('val');parity=[]
    for i in np.linspace(0,len(data)-1,8,dtype=int):
        image,*_=data[int(i)];tensor=image.unsqueeze(0)
        with torch.inference_mode():reference=net(tensor).numpy()
        actual=session.run(None,{'images':tensor.numpy()})[0]
        delta=float(np.abs(reference-actual).max())
        np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-3)
        parity.append({'val_image_index':int(i),'max_abs_error':delta})
    (BASE/'onnx_validation_parity.json').write_text(json.dumps(parity,indent=2))
    shape=lambda value:[d.dim_value if d.HasField('dim_value') else d.dim_param for d in value.type.tensor_type.shape.dim]
    info={'opset':[{ 'domain':o.domain,'version':o.version} for o in graph.opset_import],
          'inputs':{v.name:shape(v) for v in graph.graph.input},'outputs':{v.name:shape(v) for v in graph.graph.output},
          'operators':dict(Counter(n.op_type for n in graph.graph.node)),'node_count':len(graph.graph.node),
          'conv_nodes':[{'name':n.name,'outputs':list(n.output)} for n in graph.graph.node if n.op_type=='Conv'],
          'sha256':sha(target),'bytes':target.stat().st_size,'metadata':meta,'compiled_hef':False}
    (BASE/'onnx_manifest.json').write_text(json.dumps(info,indent=2))
    config={'backend':'compact-onnx','model_path':args.model_path,
            'model_sha256':sha(target),'model_id':'YOLOX-tiny-warehouse-person-forklift',
            'model_revision':sha(BASE/'best.pth'),'input_size':416,'sample_fps':1.0,
            'confidence':selected['selected']['confidence'],'nms_iou':.45,'gap_ratio':.3,'min_samples':2,'merge_gap_s':2.1,'threads':4}
    freeze={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'checkpoint_sha256':sha(BASE/'best.pth'),'onnx_sha256':sha(target),'checkpoint_epoch':checkpoint['epoch'],
            'selection_source':'Only disjoint validation runs; common confidence maximizing mean class F1 among thresholds meeting both recall >=0.9, else maximum mean F1',
            'validation':selected,'settings':config,'split_sha256':sha(BASE/'split.json'),'overlap_audit_sha256':sha(BASE/'overlap_audit.json'),
            'evaluation_tuning_permitted':False,'onnx_val_parity':parity,'compiled_hef':False,
            'evaluation_plan':{'sample_fps':[1,5],'primary':'same phase2 10 runs; same coverage exclusions; 9runs/90 gold frames',
                               'routine':'same phase2 10 runs; candidate groups and overlays only; no false-positive rate'},
            'compact_vision_sha256':sha('tracebrief/src/tracebrief/compact_vision.py')}
    # No inference on held-out primary/routine has been run before this write.
    (BASE/'FREEZE.json').write_text(json.dumps(freeze,indent=2))
    Path('tracebrief/config/vision-compact.json').write_text(json.dumps(config,indent=2))
    with (BASE/'REPORT.md').open('a') as stream:
        stream.write(f'\n### Validation, export and freeze\n\nFrozen at `{freeze["timestamp_utc"]}`; best epoch {checkpoint["epoch"]}; confidence {config["confidence"]}. Validation recall >= 0.9 for both classes: {selected["selected"]["both_recall_ge_0_9"]}. ONNX opset 11, input 1x3x416x416, raw two-class grid output; PyTorch/ONNX numerical parity passed on 8 validation frames. Weight and ONNX SHA-256 values and all inference settings are in `FREEZE.json`. No HEF has been compiled.\n')
    print('FROZEN',json.dumps(config),flush=True)
