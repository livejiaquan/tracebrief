"""Compact decoding/grouping tests use synthetic arrays, never model weights."""
import math
import pytest
from tracebrief import compact_vision as compact


def test_preprocess_preserves_bgr_scale_and_top_left_padding():
    np=pytest.importorskip('numpy');pytest.importorskip('cv2')
    image=np.full((10,20,3),[10,20,30],dtype=np.uint8)
    tensor,ratio=compact.preprocess(image,32)
    assert tensor.shape==(1,3,32,32)
    assert ratio==1.6
    assert tensor[0,:,0,0].tolist()==[10,20,30]
    assert tensor[0,:,31,31].tolist()==[114,114,114]


def test_decode_multiplies_objectness_and_keeps_overlapping_different_classes():
    np=pytest.importorskip('numpy')
    raw=np.zeros((21,7),dtype=np.float32)
    raw[0]=[1,1,math.log(2),math.log(2),.8,.75,.1]
    raw[1]=[0,1,math.log(2),math.log(2),.9,.6,.1]
    raw[2]=[-1,1,math.log(2),math.log(2),.8,.1,.75]
    detections=compact.decode(raw,input_size=32,ratio=1,width=32,height=32,confidence=.55,nms_iou=.45)
    assert [d['class_name'] for d in detections]==['person','forklift']
    assert detections[0]['confidence']==.6
    assert detections[0]['bbox_xyxy']==[0.,0.,16.,16.]


def test_decode_rejects_nonfinite_and_padding_only_boxes():
    np=pytest.importorskip('numpy')
    raw=np.zeros((21,7),dtype=np.float32)
    raw[0]=[float('nan'),0,0,0,1,1,0]
    raw[15]=[0,0,0,0,1,1,0]  # bottom/right padding beyond a narrow original frame
    assert compact.decode(raw,32,1,4,4,.5,.45)==[]


def test_decode_requires_expected_grid_and_two_classes():
    np=pytest.importorskip('numpy')
    with pytest.raises(ValueError,match='shape'):
        compact.decode(np.zeros((1,8)),32,1,32,32,.5,.45)


def test_grouping_matches_gap_minimum_peak_and_video_bounds():
    def obs(t,gap,index):
        return {'frame_index':index,'time_s':t,'image_path':f'{index}.jpg',
                'proximity_pairs':[{'gap_person_height_ratio':gap}]}
    events=compact.group_candidates([obs(0,.2,0),obs(2,.1,60),obs(5,.01,150)],10,2.1,2)
    assert len(events)==1
    assert events[0]['id']=='candidate-001'
    assert events[0]['start_s']==0
    assert events[0]['end_s']==3
    assert events[0]['peak_s']==2
    assert [e['frame_index'] for e in events[0]['evidence']]==[0,60]


@pytest.mark.parametrize('setting,value',[('sample_fps',0),('sample_fps',float('nan')),('confidence',1),('gap_ratio',-1),('min_samples',0),('merge_gap_s',-1),('input_size',415)])
def test_invalid_settings_fail_before_loading_model(setting,value):
    with pytest.raises(ValueError):
        compact.analyze_video('absent.mp4',{setting:value})


def test_missing_model_has_actionable_error():
    with pytest.raises(FileNotFoundError,match='ONNX'):
        compact.analyze_video('absent.mp4',{'model_path':'/tmp/nonexistent-compact-detector.onnx'})


def test_frozen_model_hash_mismatch_fails_before_onnx_loading(tmp_path):
    path=tmp_path/'modified.onnx';path.write_bytes(b'not the frozen weights')
    with pytest.raises(ValueError,match='SHA256'):
        compact.analyze_video('absent.mp4',{'model_path':str(path),'model_sha256':'0'*64})


def test_pair_rule_and_defaults_are_shared_with_existing_vision():
    from tracebrief import vision
    for key in ['gap_ratio','merge_gap_s','min_samples','nms_iou']:
        assert compact.DEFAULTS[key]==vision.DEFAULTS[key]
    boxes=[{'class_name':'person','bbox_xyxy':[0,0,10,100],'confidence':.8},
           {'class_name':'forklift','bbox_xyxy':[40,0,100,100],'confidence':.9}]
    assert compact._pairs(boxes,.3)==vision._pairs(boxes,.3)
    assert compact._pairs(boxes,.3)[0]['gap_person_height_ratio']==.3
