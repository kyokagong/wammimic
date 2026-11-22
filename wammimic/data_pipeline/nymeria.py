"""Nymeria VRS + unmodified XSens references, kept separate from optical labels."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .common import sha256, write_json
from .human_ego import human_features, human_history, optical_quality
from .openhlm import encode_video

NAME='nymeria_ego'
DATASET='wam_lerobot_v2/'+NAME
PROVENANCE='xsens_inertial_mocap_reference'
POINTS=[0,21,17,14,10]  # Official XSens order: pelvis, L/R foot, L/R hand.


def causal_indices(times, query, maximum_age):
    if not np.isfinite(times).all() or (np.diff(times)<=0).any():raise ValueError('Nonmonotone sensor clock')
    index=np.searchsorted(times,query,side='right')-1
    valid=(index>=0)&(index<len(times));index=index.clip(0,len(times)-1)
    return index,valid&(query-times[index]>=-1e-9)&(query-times[index]<=maximum_age)


def valid_blocks(grid, body_t, body_indices, quality, image_valid, segments):
    valid=np.zeros(len(grid),dtype=bool)
    # Every raw 240 Hz measurement is checked, including differences at history start.
    bad=np.r_[0,np.cumsum(~quality)]
    gaps=np.r_[0,np.cumsum(abs(np.diff(body_t)-1/240)>.001)]
    for i in range(3,len(grid)-12):
        a,b=body_indices[i-3]-1,body_indices[i+12]
        span=slice(i-3,i+13)
        valid[i]=(a>=0 and image_valid[span].all() and segments[i]>=0 and
            (segments[span]==segments[i]).all() and bad[b+1]==bad[a] and gaps[b]==gaps[a])
    return valid


def annotation_labels(grid, intervals, tasks):
    segments=np.full(len(grid),-1,dtype=np.int64);task_ids=segments.copy();counts=np.zeros(len(grid),dtype=int)
    for n,(start,end,text) in enumerate(intervals):
        if not np.isfinite([start,end]).all() or end<=start:raise ValueError('Invalid narration interval')
        active=(grid>=start)&(grid<end);counts[active]+=1
        segments[active]=n;task_ids[active]=tasks.setdefault(text,len(tasks))
    # Chunked official annotations overlap: ambiguous spans cannot supply single-task windows.
    segments[counts!=1]=-1;task_ids[counts!=1]=-1
    return segments,task_ids,int((counts>1).sum())


def verify_manifest(raw, manifest):
    import concurrent.futures
    def _check(row):
        p=raw/row['path']
        if not p.resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe source path')
        if not p.is_file() or p.stat().st_size!=row['size']:raise FileNotFoundError(p)
        h=hashlib.sha1()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha1']:raise ValueError('Nymeria source checksum failed: '+p.name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_check, manifest['files']))


def _decode_one_sequence(args):
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain
    index, entry, raw_str, output_str, split_name = args
    raw, output = Path(raw_str), Path(output_str)
    dataset = output / DATASET
    sequence = entry['sequence']; folder = raw / sequence
    work = output / 'source_arrays' / sequence; work.mkdir(parents=True, exist_ok=True)
    npz = work / 'xdata.npz'
    if not npz.exists():
        with zipfile.ZipFile(next(folder.glob('*_body_motion.zip'))) as z, z.open('body/xdata.npz') as src, npz.with_suffix('.tmp').open('wb') as dst:
            shutil.copyfileobj(src, dst)
        npz.with_suffix('.tmp').replace(npz)
    with np.load(npz, allow_pickle=False) as source:
        body_t = source['timestamps_us'].astype(np.float64) / 1e6
        xyz = source['segment_tXYZ'].reshape(-1, 23, 3)[:, POINTS]
        quat = source['segment_qWXYZ'].reshape(-1, 23, 4)[:, POINTS]
        rest = source['identity_segment_tXYZ'].reshape(23, 3)
        leg = float(np.mean([np.linalg.norm(rest[a] - rest[b]) + np.linalg.norm(rest[b] - rest[c]) for a, b, c in [(15, 16, 17), (19, 20, 21)]]))
        poses = np.concatenate((xyz, quat), axis=-1)
    finite = np.isfinite(poses).all((1, 2)) & (abs(np.linalg.norm(quat, axis=-1) - 1) < 1e-3).all(1)
    safe = poses.copy(); safe[~finite] = 0; safe[~finite, :, 3] = 1
    quality, jumps = optical_quality(body_t, safe, finite)
    provider = data_provider.create_vrs_data_provider(str(next(folder.glob('*_recording_head_data_data.vrs'))))
    stream = StreamId('214-1')
    if not provider.supports_time_domain(stream, TimeDomain.TIME_CODE):
        raise ValueError('VRS has no recorded timecode mapping')
    image_t = np.asarray(provider.get_timestamps_ns(stream, TimeDomain.TIME_CODE), dtype=np.float64) / 1e9
    start = np.ceil(max(body_t[0], image_t[0]) * 10) / 10; end = min(body_t[-1], image_t[-1])
    grid = start + np.arange(int(np.floor((end - start) * 10)) + 1) / 10
    nearest, body_ok = causal_indices(body_t, grid, 1 / 240 + .001)
    image_indices, image_ok = causal_indices(image_t, grid, 1 / 15 + .002)
    with next(folder.glob('*_narration_activity_summarization.csv')).open() as f:
        annotations = list(csv.DictReader(f))
    intervals = []
    for annotation in annotations:
        text = annotation['Describe my activity'].strip()
        if not text:
            continue
        a = provider.convert_from_device_time_to_timecode_ns(int(float(annotation['start_time']) * 1e9)) / 1e9
        b = provider.convert_from_device_time_to_timecode_ns(int(float(annotation['end_time']) * 1e9)) / 1e9
        intervals.append((a, b, text))
    local_tasks = {}
    segments, local_task_ids, ambiguous = annotation_labels(grid, intervals, local_tasks)
    valid = valid_blocks(grid, body_t, nearest, quality, image_ok & body_ok, segments)
    frames = np.flatnonzero(valid)
    if not len(frames):
        raise ValueError('No synchronized complete Nymeria windows: ' + sequence)
    actions = np.zeros((len(grid), 10, 45), dtype=np.float32); states = np.zeros((len(grid), 4, 75), dtype=np.float32)
    for i in frames:
        actions[i] = human_features(poses[nearest[i + 1:i + 11]], poses[nearest[i]], leg)
    states[frames] = human_history(poses, body_t, nearest[frames[:, None] + np.arange(-3, 1)], leg)
    video = dataset / f'videos/chunk-000/observation.images.cam_head/episode_{index:06d}.mp4'
    video.parent.mkdir(parents=True, exist_ok=True)
    actual_image_times = []
    def images():
        for j in image_indices:
            image, meta = provider.get_image_data_by_index(stream, int(j)); rgb = image.to_numpy_array()
            stamp = provider.convert_from_device_time_to_timecode_ns(meta.capture_timestamp_ns) / 1e9
            actual_image_times.append(stamp)
            buffer = io.BytesIO(); Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_270).resize((224, 224), Image.Resampling.LANCZOS).save(buffer, format='PNG')
            yield {'bytes': buffer.getvalue()}
    encode_video(images(), video)
    if not np.allclose(actual_image_times, image_t[image_indices], atol=2e-6, rtol=0):
        raise ValueError('Decoded RGB timestamp differs from index clock')
    labels = dataset / f'labels/episode_{index:06d}.npz'; labels.parent.mkdir(parents=True, exist_ok=True)
    np.savez(labels, poses_world=poses[nearest], time_s=grid, source_body_index=nearest,
             image_index=image_indices, image_time_s=image_t[image_indices], body_time_s=body_t[nearest],
             valid=valid, segment_id=segments, human_leg_length=leg)
    staged = work / f'staged_ep_{index:06d}.npz'
    np.savez(staged, actions=actions, states=states, valid=valid, grid=grid, local_task_ids=local_task_ids)
    print('Nymeria decoded', sequence, len(frames), flush=True)
    return {
        'index': index, 'sequence': sequence, 'participant': entry['metadata']['fake_name'],
        'split': split_name, 'length': len(grid), 'valid_samples': int(len(frames)),
        'local_tasks': local_tasks, 'human_leg_length': leg, 'tracking_jumps': jumps,
        'timestamp_gap_intervals': int((abs(np.diff(body_t) - 1 / 240) > .001).sum()),
        'ambiguous_narration_grid_frames': ambiguous,
        'video': str(video), 'labels': str(labels), 'npz': str(npz), 'staged': str(staged),
    }


def import_nymeria(raw_root, manifest_path, output):
    import concurrent.futures
    import os
    from .cache import cache_owner
    raw,output=Path(raw_root),Path(output);manifest=json.loads(Path(manifest_path).read_text())
    verify_manifest(raw,manifest)
    dataset=output/DATASET
    identity={'schema':'nymeria-inertial-five-point-v1','subject_kind':'human','label_provenance':PROVENANCE,
        'source_manifest_sha256':sha256(manifest_path),'observation_state':{'shape':[4,75],'rate_hz':10,'measurement_rate_hz':240},
        'coordinate_frame':'full t0 pelvis 3D; lower positions/velocities scaled by measured leg length',
        'image_clock':'VRS DEVICE_TIME mapped through recorded TIME_CODE synchronization',
        'image_sampling':'latest RGB at or before 10Hz grid; maximum age 1/15s + 2ms',
        'image_orientation':'Aria native RGB rotated clockwise 90 degrees; no mirroring or undistortion',
        'text_source':'official activity_summarization; narrative task supervision',
        'end_effector_supervision':False}
    cache_owner(dataset/'source.json',identity,[dataset/'data'])
    selection=json.loads((Path(manifest_path).parent/manifest.get('selection_file','nymeria-selection.json')).read_text())['sequences']
    people=sorted({r['metadata']['fake_name'] for r in selection})
    if len(people)<3:raise ValueError('Need three independent participants for train/val/test')
    if len(people)==3:
        assignments={p:['train','val','test'][i] for i,p in enumerate(people)}
    else:
        n_val=max(1,round(0.1*len(people)));n_test=max(1,round(0.1*len(people)));n_train=len(people)-n_val-n_test
        assignments={p:('train' if i<n_train else 'val' if i<n_train+n_val else 'test') for i,p in enumerate(people)}
    tasks={};episodes=[]
    todo=[]
    for index,entry in enumerate(selection):
        sequence=entry['sequence']
        receipt=dataset/f'receipts/episode_{index:06d}.json'
        if receipt.exists():
            row=json.loads(receipt.read_text())
            if row['sequence']!=sequence:raise ValueError('Changed sequence order')
            for text,task in row['tasks'].items():
                if tasks.setdefault(text,task)!=task:raise ValueError('Changed task mapping')
            if any(sha256(output/p)!=h for p,h in row['outputs'].items()):raise ValueError('Changed Nymeria output')
            episodes.append(row)
        else:
            episodes.append(None)
            todo.append((index,entry,str(raw),str(output),assignments[entry['metadata']['fake_name']]))
    workers=int(os.environ.get('NYMERIA_WORKERS','1'))
    decoded={}
    if todo:
        if workers>1 and len(todo)>1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers,len(todo))) as pool:
                for res in pool.map(_decode_one_sequence,todo):
                    decoded[res['index']]=res
        else:
            for item in todo:
                res=_decode_one_sequence(item)
                decoded[res['index']]=res
    offset=0
    for index,entry in enumerate(selection):
        if episodes[index] is not None:
            offset+=episodes[index]['length'];continue
        res=decoded[index]
        inv_local={v:k for k,v in res['local_tasks'].items()}
        remap={lid:tasks.setdefault(txt,len(tasks)) for lid,txt in sorted(inv_local.items())}
        with np.load(res['staged'],allow_pickle=False) as st:
            actions=st['actions'];states=st['states'];valid=st['valid'];grid=st['grid'];local_ids=st['local_task_ids']
        task_ids=np.array([remap[int(x)] if x>=0 else -1 for x in local_ids],dtype=np.int64)
        table=pa.table({'action':pa.array(actions.tolist(),type=pa.list_(pa.list_(pa.float32(),45),10)),
            'observation.state':pa.array(states.tolist(),type=pa.list_(pa.list_(pa.float32(),75),4)),
            'valid_for_training':valid,'timestamp':grid-grid[0],'sensor_timestamp':grid,
            'frame_index':np.arange(len(grid)),'task_index':task_ids,'episode_index':np.full(len(grid),index),
            'index':np.arange(offset,offset+len(grid))})
        parquet=dataset/f'data/chunk-000/episode_{index:06d}.parquet';parquet.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,parquet,compression='zstd')
        Path(res['staged']).unlink(missing_ok=True)
        row={'episode_index':index,'sequence':res['sequence'],'participant':res['participant'],
            'split':res['split'],'length':res['length'],'valid_samples':res['valid_samples'],
            'tasks':dict(tasks),
            'human_leg_length':res['human_leg_length'],'tracking_jumps':res['tracking_jumps'],
            'timestamp_gap_intervals':res['timestamp_gap_intervals'],
            'ambiguous_narration_grid_frames':res['ambiguous_narration_grid_frames'],
            'outputs':{str(Path(p).relative_to(output)):sha256(Path(p)) for p in [res['video'],res['labels'],parquet,res['npz']]}}
        receipt=dataset/f'receipts/episode_{index:06d}.json'
        write_json(receipt,row);episodes[index]=row;offset+=res['length']
        print('Nymeria prepared',res['sequence'],res['valid_samples'],flush=True)
    meta=dataset/'meta';meta.mkdir(parents=True,exist_ok=True)
    for name,rows in [('episodes.jsonl',episodes),('tasks.jsonl',[{'task_index':i,'task':t} for t,i in tasks.items()])]:
        (meta/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    statistics={}
    for key,width in [('action',45),('observation.state',75)]:
        arrays=[]
        for row in episodes:
            if row['split']!='train':continue
            table=pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
            arrays.append(np.asarray(table[key].to_pylist(),dtype=np.float32)[table['valid_for_training'].to_numpy()].reshape(-1,width))
        a=np.concatenate(arrays);statistics[key]={'count':len(a),'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),'q01':np.quantile(a,.01,axis=0).tolist(),'q99':np.quantile(a,.99,axis=0).tolist()}
    write_json(meta/'stats.json',statistics);write_json(meta/'splits.json',{s:[r['episode_index'] for r in episodes if r['split']==s] for s in ['train','val','test']})
    write_json(meta/'info.json',{**identity,'codebase_version':'v2.1','format_variant':'human_prechunked_action_10x45','world_model':'Cosmos-Predict2.5-2B','fps':10,'total_episodes':len(episodes),'total_frames':sum(r['length'] for r in episodes),'total_tasks':len(tasks),'features':{'action':{'dtype':'float32','shape':[10,45]},'observation.state':{'dtype':'float32','shape':[4,75]},'observation.images.cam_head':{'dtype':'video','shape':[3,224,224]}}})
    write_json(meta/'embodiment.json',{'embodiment_tag':'human_five_point','label_provenance':PROVENANCE,'end_effector_supervision':False})
    write_json(meta/'modality.json',{'action':{'five_point':[0,45]},'action_horizon':10,'history_frames':4,'state':{'pose':[0,45],'linear_velocity':[45,60],'angular_velocity':[60,75]}})
    report={'episodes':len(episodes),'valid_windows':sum(r['valid_samples'] for r in episodes),'label_provenance':PROVENANCE,'ready':False}
    write_json(output/'qa_reports/nymeria_import.json',report);return report


def validate_nymeria(output):
    import torch
    torch.set_num_threads(4)
    from safetensors import safe_open
    from scipy.spatial.transform import Rotation
    from .datasets import NymeriaWAMDataset
    from .geometry import matrices
    output=Path(output);dataset=output/DATASET
    if json.loads((dataset/'source.json').read_text())['label_provenance']!=PROVENANCE:raise ValueError('Provenance lost')
    for path in [dataset/'source.json',dataset/'meta/info.json']:
        if json.loads(path.read_text())['observation_state']!={'shape':[4,75],'rate_hz':10,'measurement_rate_hz':240}:raise ValueError('History rate metadata mismatch')
    episodes=list(map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines()))
    splits=json.loads((dataset/'meta/splits.json').read_text());people={};samples=0;pos_error=0.;rot_error=0.;rgb_age=0.;body_age=0.;train={'action':[],'observation.state':[]}
    split_ids=[i for values in splits.values() for i in values]
    if len(split_ids)!=len(set(split_ids)) or set(split_ids)!={r['episode_index'] for r in episodes}:raise ValueError('Split overlap or omission')
    for row in episodes:
        for path,digest in row['outputs'].items():
            if sha256(output/path)!=digest:raise ValueError('Changed artifact: '+path)
        if row['episode_index'] not in splits[row['split']]:raise ValueError('Split mismatch')
        if row['participant'] in people and people[row['participant']]!=row['split']:raise ValueError('Participant leakage')
        people[row['participant']]=row['split']
        with np.load(output/'source_arrays'/row['sequence']/'xdata.npz',allow_pickle=False) as original:
            times=original['timestamps_us'].astype(float)/1e6
            poses=np.concatenate((original['segment_tXYZ'].reshape(-1,23,3)[:,POINTS],original['segment_qWXYZ'].reshape(-1,23,4)[:,POINTS]),axis=-1)
        with np.load(dataset/f"labels/episode_{row['episode_index']:06d}.npz",allow_pickle=False) as labels:
            grid=labels['time_s'];idx=labels['source_body_index'];image_t=labels['image_time_s'];leg=float(labels['human_leg_length'])
            expected_idx,paired=causal_indices(times,grid,1/240+.001)
            if not np.array_equal(idx,expected_idx) or not np.array_equal(labels['poses_world'],poses[idx],equal_nan=True):raise ValueError('Measured pose identity mismatch')
            finite=np.isfinite(poses).all((1,2))&(abs(np.linalg.norm(poses[...,3:],axis=-1)-1)<1e-3).all(1)
            safe=poses.copy();safe[~finite]=0;safe[~finite,:,3]=1
            quality,_=optical_quality(times,safe,finite)
            valid=valid_blocks(grid,times,idx,quality,paired&(image_t<=grid+1e-9)&(grid-image_t<=1/15+.002),labels['segment_id'])
        table=pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        if not np.array_equal(valid,table['valid_for_training'].to_numpy()):raise ValueError('Invalid window mask')
        frames=np.flatnonzero(valid);actions=np.asarray(table['action'].to_pylist(),dtype=np.float32);states=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        rgb_age=max(rgb_age,float((grid[frames]-image_t[frames]).max()));body_age=max(body_age,float((grid[frames]-times[idx[frames]]).max()))
        for start in range(0,len(frames),128):
            f=frames[start:start+128];future=poses[idx[f[:,None]+np.arange(1,11)]];anchor=poses[idx[f]]
            if not np.allclose(actions[f],human_features(future,anchor,leg),atol=1e-5,rtol=0):raise ValueError('Incorrect action labels')
            if not np.allclose(states[f],human_history(poses,times,idx[f[:,None]+np.arange(-3,1)],leg),atol=1e-5,rtol=0):raise ValueError('Incorrect causal history')
            basis=matrices(anchor[:,0,3:]);p=actions[f,:,:15].reshape(-1,10,5,3).astype(float);p[:,:,:3]*=leg
            restored=np.einsum('bij,btpj->btpi',basis,p)+anchor[:,None,:1,:3]
            pos_error=max(pos_error,float(np.linalg.norm(restored-future[...,:3],axis=-1).max()))
            cols=actions[f,:,15:].reshape(-1,10,5,2,3).swapaxes(-1,-2);relative=np.concatenate((cols,np.cross(cols[...,0],cols[...,1])[...,None]),axis=-1)
            delta=(basis[:,None,None]@relative).swapaxes(-1,-2)@matrices(future[...,3:])
            rot_error=max(rot_error,float(np.rad2deg(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude()).max()))
        path=dataset/f"latents/chunk-000/episode_{row['episode_index']:06d}.safetensors";receipt=json.loads(path.with_suffix('.json').read_text())
        if sha256(path)!=receipt['sha256']:raise ValueError('Changed latent cache')
        with safe_open(str(path),framework='pt') as encoded:
            if not np.array_equal(encoded.get_tensor('frame_index').numpy(),frames) or encoded.get_slice('latents').get_shape()!=[len(frames),16,5,28,28]:raise ValueError('Missing encoded windows')
            for i in range(0,len(frames),32):
                if not torch.isfinite(encoded.get_slice('latents')[i:i+32]).all():raise ValueError('Nonfinite encoded input')
        if row['split']=='train':
            train['action'].append(actions[frames].reshape(-1,45));train['observation.state'].append(states[frames].reshape(-1,75))
        samples+=len(frames)
    if any(not splits[k] for k in ['train','val','test']) or len(people)<3:raise ValueError('Missing held-out participants')
    stats=json.loads((dataset/'meta/stats.json').read_text())
    for key,arrays in train.items():
        values=np.concatenate(arrays)
        if not np.isfinite(values).all() or not np.allclose(values.mean(0),stats[key]['mean'],atol=1e-6) or not np.allclose(values.std(0),stats[key]['std'],atol=1e-6):raise ValueError('Train-only statistics failed')
    if pos_error>=.001 or rot_error>=.1:raise ValueError('Coordinate roundtrip failed')
    text=json.loads((output/'prompt_bank.json').read_text())
    if not text['complete'] or sha256(output/'prompt_bank_bf16.safetensors')!=text['sha256']:raise ValueError('Text cache failed')
    tasks={str(r['task_index']):r['task'] for r in map(json.loads,(dataset/'meta/tasks.jsonl').read_text().splitlines())}
    if text['tasks']!=tasks:raise ValueError('Text task mapping changed')
    with safe_open(str(output/'prompt_bank_bf16.safetensors'),framework='pt') as bank:
        if set(bank.keys())!={'task_'+i for i in tasks}:raise ValueError('Missing text task')
        for key in bank.keys():
            tensor=bank.get_tensor(key)
            if tensor.shape!=(512,100352) or not torch.isfinite(tensor).all():raise ValueError('Invalid text embedding')
    from wammimic.training.model import LinearBaseline,action_loss
    split_counts={};head=LinearBaseline()
    for split in ['train','val','test']:
        reader=NymeriaWAMDataset(output,split);split_counts[split]=len(reader)
        batch=next(iter(torch.utils.data.DataLoader(reader,batch_size=2)))
        if batch['actions'].shape!=(2,10,45) or batch['proprio'].shape!=(2,4,75) or not all(torch.isfinite(v).all() for v in batch.values()):raise ValueError('Invalid reader batch')
        head.zero_grad();prediction=head(batch['video_latents'],batch['text_embeddings'])
        loss,_=action_loss(prediction,batch['actions']);loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Nonfinite WAM baseline gradient')
    previews=preview_nymeria(output,episodes)
    result={'ready':True,'episodes':len(episodes),'cached_samples':samples,'split_samples':split_counts,'paired_grid_frames':sum(r['length'] for r in episodes),'maximum_causal_rgb_age_seconds':rgb_age,'maximum_causal_body_age_seconds':body_age,'label_provenance':PROVENANCE,'participant_splits':people,'maximum_position_roundtrip_m':pos_error,'maximum_rotation_roundtrip_degrees':rot_error,'finite_gradient':True,'gradient_check':'existing WAM RGB/text baseline; no optimizer update or checkpoint','rerun_previews':previews,'limits':'Three-participant inertial reference subset; no optical/floor/contact/finger labels; not a robot closed-loop validation'}
    write_json(output/'qa_reports/nymeria_wam_training_ready.json',result);return result


def preview_nymeria(output,episodes):
    import rerun as rr
    from PIL import ImageDraw
    from .cache import read_video
    from .geometry import matrices
    destination=output/'qa_previews';destination.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(5*224,4*248),'white');cases=[];rng=np.random.default_rng(20261120)
    names=['pelvis','left_foot','right_foot','left_hand','right_hand']
    for row in episodes:
        episode=row['episode_index'];root=output/DATASET
        with np.load(root/f'labels/episode_{episode:06d}.npz',allow_pickle=False) as labels:
            valid=np.flatnonzero(labels['valid']);poses=labels['poses_world'];times=labels['time_s']
        pixels=read_video(root/f'videos/chunk-000/observation.images.cam_head/episode_{episode:06d}.mp4',row['length'])
        for frame in sorted(rng.choice(valid,20//len(episodes)+(episode<20%len(episodes)),replace=False)):
            path=destination/f'episode_{episode:06d}_frame_{frame:06d}.rrd'
            stream=rr.RecordingStream('Nymeria-'+path.stem);stream.save(str(path))
            stream.log('world',rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
            stream.log('source',rr.TextDocument(f"{PROVENANCE}; {row['sequence']}; {row['split']}. Measured human reference, no robot mesh or calibrated ground. RGB rotated clockwise 90 degrees; no mirroring."),static=True)
            origin=poses[frame,0,:3]
            for i in range(frame-3,frame+13):
                stream.set_time('relative_seconds',duration=float(times[i]-times[frame]))
                stream.log('rgb',rr.Image(pixels[i]));points=poses[i,:,:3]-origin
                stream.log('world/points',rr.Points3D(points,radii=.02,labels=names))
                rotations=matrices(poses[i,:,3:])
                for j,name in enumerate(names):
                    stream.log('world/axes/'+name,rr.Arrows3D(origins=np.repeat(points[j][None],3,axis=0),vectors=rotations[j].T*.1,colors=[[255,0,0],[0,180,0],[0,90,255]]))
                stream.log('world/hand_paths',rr.LineStrips3D([(poses[frame-3:i+1,j,:3]-origin) for j in [3,4]]))
            stream.flush()
            tile=Image.new('RGB',(224,248),'white');tile.paste(Image.fromarray(pixels[frame]),(0,24));ImageDraw.Draw(tile).text((3,4),f'{episode} / {frame} / {row["split"]}',fill='black')
            k=len(cases);sheet.paste(tile,((k%5)*224,(k//5)*248))
            cases.append({'episode_index':episode,'frame_index':int(frame),'split':row['split'],'path':str(path.relative_to(output)),'sha256':sha256(path)})
    if len(cases)!=20:raise ValueError('Expected twenty source previews')
    sheet.save(destination/'contact_sheet.png');write_json(destination/'index.json',{'seed':20261120,'cases':cases})
    return len(cases)
