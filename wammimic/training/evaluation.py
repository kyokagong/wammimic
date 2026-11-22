"""Physical-unit pose metrics; no per-frame alignment and no oracle best-of-N."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from wammimic.data_pipeline.common import write_json
from wammimic.data_pipeline.human_ego import DATASET,human_features
from wammimic.data_pipeline.geometry import matrices


def autocast(device):return torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else nullcontext()


def rotation6d(values):
    a,b=values[...,:3],values[...,3:]
    length=np.linalg.norm(a,axis=-1,keepdims=True);u=a/np.maximum(length,1e-12)
    second=b-(u*b).sum(-1,keepdims=True)*u
    second_length=np.linalg.norm(second,axis=-1,keepdims=True);v=second/np.maximum(second_length,1e-12)
    valid=(length[...,0]>1e-6)&(second_length[...,0]>1e-6)&np.isfinite(values).all(-1)
    return np.stack([u,v,np.cross(u,v)],-1),valid


def pose_errors(prediction,reference,leg_length):
    prediction,reference=np.asarray(prediction,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if prediction.shape!=(10,45) or reference.shape!=(10,45):raise ValueError('Expected one 10x45 action chunk')
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite output: evaluation failed, sample must not be silently dropped')
    positions=prediction[:,:15].reshape(10,5,3).copy();target=reference[:,:15].reshape(10,5,3).copy()
    positions[:,:3]*=leg_length;target[:,:3]*=leg_length
    distance=np.linalg.norm(positions-target,axis=-1)*1000
    rotations,valid=rotation6d(prediction[:,15:].reshape(10,5,6));truth,truth_valid=rotation6d(reference[:,15:].reshape(10,5,6))
    if not truth_valid.all():raise ValueError('Degenerate optical reference rotation')
    relative=rotations.swapaxes(-1,-2)@truth
    cosine=(np.trace(relative,axis1=-2,axis2=-1)-1)/2
    angle=np.rad2deg(np.arccos(cosine.clip(-1,1)));angle[~valid]=180
    return {'position_mm':distance,'rotation_deg':angle,'invalid_rotation':~valid}


def temporal_errors(prediction,reference,leg_length,times):
    times=np.asarray(times,dtype=float)
    if times.shape!=(10,) or not np.isfinite(times).all() or (np.diff(times)<=0).any():raise ValueError('Invalid diagnostic timestamps')
    p=np.asarray(prediction,dtype=float)[:,:15].reshape(10,5,3).copy()
    g=np.asarray(reference,dtype=float)[:,:15].reshape(10,5,3).copy();p[:,:3]*=leg_length;g[:,:3]*=leg_length
    result={};t=times
    for name in ['velocity_rmse_m_s','acceleration_rmse_m_s2','jerk_rmse_m_s3']:
        interval=np.diff(t);p=np.diff(p,axis=0)/interval[:,None,None];g=np.diff(g,axis=0)/interval[:,None,None]
        result[name]=float(np.sqrt(np.mean(np.sum((p-g)**2,axis=-1))));t=(t[1:]+t[:-1])/2
    speed=[]
    for action in [prediction,reference]:
        rotation,valid=rotation6d(np.asarray(action)[:,15:].reshape(10,5,6));relative=rotation[:-1].swapaxes(-1,-2)@rotation[1:]
        angle=np.arccos(((np.trace(relative,axis1=-2,axis2=-1)-1)/2).clip(-1,1));angle[~(valid[:-1]&valid[1:])]=np.pi
        speed.append(angle/np.diff(times)[:,None])
    result['angular_speed_rmse_rad_s']=float(np.sqrt(np.mean((speed[0]-speed[1])**2)))
    return result


def aggregate(errors):
    p=np.stack([r['position_mm'] for r in errors]);r=np.stack([r['rotation_deg'] for r in errors])
    return {'samples':len(errors),'ade_mm':float(p.mean()),'fde_mm':float(p[:,-1].mean()),
        'ade_per_point_mm':p.mean((0,1)).tolist(),'fde_per_point_mm':p[:,-1].mean(0).tolist(),
        'position_p95_mm':float(np.quantile(p,.95)),'rotation_mean_deg':float(r.mean()),
        'rotation_p95_deg':float(np.quantile(r,.95)),
        'invalid_rotation_rate':float(np.stack([e['invalid_rotation'] for e in errors]).mean()),
        'horizon_mean_position_mm':{str(t):float(p[:,i].mean()) for i,t in [(1,.2),(4,.5),(9,1.)]}}


def summarize(records,errors):
    overall=aggregate(errors);groups={}
    for key in ['scope','family','recording','participant']:
        groups[key]={}
        for value in sorted({r[key] for r in records}):
            groups[key][value]=aggregate([e for r,e in zip(records,errors) if r[key]==value])
    overall['recording_macro_ade_mm']=float(np.mean([v['ade_mm'] for v in groups['recording'].values()]))
    return {'overall':overall,'groups':groups}


def privileged_reference(descriptor):
    path=Path(descriptor['root'])/DATASET/f'labels/episode_{descriptor["episode_index"]:06d}.npz'
    with np.load(path,allow_pickle=False) as labels:
        frame=descriptor['frame_index'];world=labels['poses_world'];t=labels['video_timestamp'];anchor=world[frame].copy()
        future=np.repeat(anchor[None],10,axis=0)
        velocity=(world[frame,:,:3]-world[frame-1,:,:3])/(t[frame]-t[frame-1])
        future[...,:3]+=(t[frame+1:frame+11]-t[frame])[:,None,None]*velocity
        return human_features(future,anchor,descriptor['human_leg_length'])


def world_actions(action,anchor,leg):
    position=np.asarray(action[:,:15],dtype=float).reshape(10,5,3).copy();position[:,:3]*=leg
    rotation=matrices(anchor[0,3:])
    world=position@rotation.T+anchor[0,:3]
    orientation,valid=rotation6d(action[:,15:].reshape(10,5,6))
    return world,rotation@orientation,valid


@torch.inference_mode()
def overlap_diagnostic(model,dataset,device,steps):
    results=[];seen=set();stats=dataset.protocol['statistics']['action'];mean=np.asarray(stats['mean']);std=np.maximum(stats['std'],1e-6)
    for index,(source_id,local) in enumerate(dataset.samples):
        reader=dataset.readers[source_id];episode,_,frame=reader.samples[local]
        if (source_id,episode) in seen:continue
        next_local=next((j for j,(ep,_,f) in enumerate(reader.samples) if ep==episode and f==frame+1),None)
        if next_local is None:continue
        seen.add((source_id,episode));chunks=[]
        for sample in [reader[local],reader[next_local]]:
            video=sample['video_latents'][None].to(device);text=sample['text_embeddings'][None].to(device)
            with autocast(device):
                context=model.encode_history(video,text)
                action=model.sample(context,torch.zeros((1,10,45),device=device),steps)[0].cpu().numpy()*std+mean
            chunks.append(action)
        descriptor=dataset.descriptor(index)
        with np.load(Path(descriptor['root'])/DATASET/f'labels/episode_{episode:06d}.npz',allow_pickle=False) as labels:
            a=world_actions(chunks[0],labels['poses_world'][frame],descriptor['human_leg_length'])
            b=world_actions(chunks[1],labels['poses_world'][frame+1],descriptor['human_leg_length'])
        error=np.linalg.norm(a[0][1:]-b[0][:-1],axis=-1)*1000
        valid=a[2][1:]&b[2][:-1];relative=a[1][1:].swapaxes(-1,-2)@b[1][:-1]
        angle=np.rad2deg(np.arccos(((np.trace(relative,axis1=-2,axis2=-1)-1)/2).clip(-1,1)));angle[~valid]=180
        results.append({'source_id':source_id,'episode':episode,'frame':frame,'position_mm':float(error.mean()),'rotation_deg':float(angle.mean())})
    return {'condition':'adjacent t0 frames, fixed zero initial action noise; compare nine shared future targets in world coordinates',
        'recordings':results,'mean_position_mm':float(np.mean([r['position_mm'] for r in results])) if results else None}


@torch.inference_mode()
def evaluate(model,baseline,dataset,device,seed=0,steps=16,output=None,save_predictions=False,diagnose_overlap=False):
    model.eval();stats=dataset.protocol['statistics']['action']
    mean=np.asarray(stats['mean']);std=np.maximum(np.asarray(stats['std']),1e-6)
    baselines={'training_mean':'active training set only',
               'privileged_constant_velocity':'past measured poses available only to this reference, not to WAM'}
    if baseline is not None:
        baseline.eval()
        baselines['rgb_text_baseline']='same RGB/text observations, trained lightweight MLP'
    records=[];errors={k:[] for k in ['wam',*baselines]};predictions=[];targets=[];linear=[];latency=[]
    for index,batch in enumerate(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)):
        descriptor=dataset.descriptor(index);video=batch['video_latents'].to(device);text=batch['text_embeddings'].to(device)
        identity=f'{seed}/{descriptor["source_id"]}/{descriptor["episode_index"]}/{descriptor["frame_index"]}'
        sample_seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')%(2**63-1)
        noise=torch.randn((1,10,45),device=device,generator=torch.Generator(device=device).manual_seed(sample_seed))
        if torch.device(device).type=='cuda':torch.cuda.synchronize()
        started=time.perf_counter()
        with autocast(device):
            context=model.encode_history(video,text)
            prediction=model.sample(context,noise,steps)[0].float().cpu().numpy()*std+mean
            if torch.device(device).type=='cuda':torch.cuda.synchronize()
            latency.append((time.perf_counter()-started)*1000)
            reference=batch['actions'][0].numpy()*std+mean
            if baseline is not None:
                simple=baseline(video,text)[0].float().cpu().numpy()*std+mean
        record={k:descriptor[k] for k in ['source_id','episode_index','frame_index','family','participant','human_leg_length']}
        record['scope']='operation_session_holdout' if descriptor['source_id']==1 else 'walking_subject_holdout'
        record['recording']=str(descriptor['source_id'])+'/'+descriptor['source_session'];records.append(record)
        for name,action in [('wam',prediction),('training_mean',np.repeat(mean[None],10,axis=0)),('privileged_constant_velocity',privileged_reference(descriptor))]:
            errors[name].append(pose_errors(action,reference,descriptor['human_leg_length']))
        if baseline is not None:
            errors['rgb_text_baseline'].append(pose_errors(simple,reference,descriptor['human_leg_length']))
            if save_predictions:linear.append(simple)
        if save_predictions:predictions.append(prediction);targets.append(reference)
    report={'split':dataset.split,'seed':seed,'denoising_steps':steps,'reference':'optical mocap solved skeleton',
        'sampling':'one fixed seed per sample; no oracle selection; source-disjoint 1.6s windows',
        'baselines':baselines,
        'metrics':{k:summarize(records,v) for k,v in errors.items()},'records':records}
    report['inference_ms']={'p50':float(np.median(latency[1:] or latency)),'p95':float(np.quantile(latency[1:] or latency,.95)),
        'scope':'batch=1, cached encoder inputs already on device; includes Cosmos conditioning and action denoising, excludes RGB/text encoding'}
    if diagnose_overlap:report['adjacent_chunks']=overlap_diagnostic(model,dataset,device,steps)
    if output:
        output=Path(output);write_json(output,report)
        if save_predictions:
            arrays={'prediction':predictions,'reference':targets}
            if baseline is not None:arrays['rgb_text_baseline']=linear
            np.savez(output.with_suffix('.npz'),**{k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()})
    return report
