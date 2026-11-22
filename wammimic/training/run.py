"""Train and evaluate a small, genuine Predict2.5-2B human WAM pilot."""
import argparse
from importlib.metadata import version
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file,save_file
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wammimic.data_pipeline.common import sha256,write_json
from .data import build_protocol,PilotDataset
from .evaluation import autocast,evaluate
from .model import CosmosWAM,LinearBaseline,action_loss,flow_time,load_backbone


def trainable_state(model):return {k:p.detach().cpu().contiguous() for k,p in model.named_parameters() if p.requires_grad}


def load_trainable(model,path):
    state=load_file(str(path),device='cpu');parameters={k:p for k,p in model.named_parameters() if p.requires_grad}
    if set(state)!=set(parameters):raise ValueError('Trainable checkpoint keys do not match this model')
    with torch.no_grad():
        for name,p in parameters.items():
            if state[name].shape!=p.shape:raise ValueError('Checkpoint shape mismatch')
            p.copy_(state[name].to(device=p.device,dtype=p.dtype))


def save_models(model,baseline,directory,include=('wam','baseline')):
    directory.mkdir(parents=True,exist_ok=True)
    for name,module in [('wam',model),('baseline',baseline)]:
        if name not in include:continue
        temporary=directory/(name+'.tmp');save_file(trainable_state(module),str(temporary));temporary.chmod(0o644);temporary.replace(directory/(name+'.safetensors'))


def preserved_mimic(protocol):
    p=Path(protocol['sources'][0]['root'])/'qa_reports/mimic_preserved_before.json';values=json.loads(p.read_text())
    if any(sha256(path)!=digest for path,digest in values.items()):raise ValueError('Existing Mimic preservation fingerprints changed')
    return values


def train(args):
    run=Path(args.run_root)
    if (run/'run.json').exists():raise FileExistsError('Preserve existing run; choose a new run root')
    run.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('The 2B pilot requires CUDA')
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
    train_data=PilotDataset(args.protocol,'train',args.train_per_episode);validation=PilotDataset(args.protocol,'val',args.eval_per_episode)
    fingerprints=preserved_mimic(train_data.protocol)
    model=CosmosWAM(load_backbone(args.backbone,args.backbone_manifest,device)).to(device)
    baseline=LinearBaseline().to(device)
    backbone_parameters=list(model.backbone.transformer_blocks[-1].parameters());backbone_ids={id(p) for p in backbone_parameters}
    experts=[p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([{'params':experts,'lr':args.learning_rate},
        {'params':backbone_parameters,'lr':args.backbone_learning_rate}],weight_decay=.01)
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),lr=args.learning_rate,weight_decay=.01)
    initial={k:p.detach().cpu().clone() for k,p in model.backbone.transformer_blocks[-1].named_parameters()}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};config.update(backbone_name='Cosmos-Predict2.5-2B',protocol_sha256=sha256(args.protocol),
        backbone_manifest=json.loads(Path(args.backbone_manifest).read_text()),mimic_fingerprints=fingerprints,
        train_samples=len(train_data),validation_samples=len(validation),device=torch.cuda.get_device_name(),
        action_dim=45,conditioning_latents=2,future_conditioning=False,
        action_expert={'width':model.width,'layers':len(model.decoder.layers),'horizon':10,'dimensions':45},
        software={'torch':str(torch.__version__),'diffusers':version('diffusers'),'numpy':np.__version__},
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        time_sampling='s=sigmoid(N(0,1)); t=5s/(1+4s); action flows from noise at t=0 to data at t=1')
    write_json(run/'run.json',config)
    writer=SummaryWriter(str(run/'tensorboard'));log=(run/'training.jsonl').open('a',buffering=1)
    first=next(iter(DataLoader(validation,batch_size=1)));video=first['video_latents'].to(device);text=first['text_embeddings'].to(device)
    model.eval()
    with torch.no_grad(),autocast(device):
        before=model.encode_history(video,text);changed=video.clone();changed[:,:,2:]=1000
        after=model.encode_history(changed,text)
        if not torch.equal(before,after):raise ValueError('Future latent leakage into the action expert')
    del before,after,changed,first,video,text
    initial_report=evaluate(model,baseline,validation,device,seed=0,steps=args.denoising_steps,output=run/'initial_validation.json')
    best=initial_report['metrics']['wam']['overall']['recording_macro_ade_mm'];save_models(model,baseline,run/'best');best_epoch=0
    best_baseline=initial_report['metrics']['rgb_text_baseline']['overall']['recording_macro_ade_mm'];best_baseline_epoch=0
    optimizer.zero_grad(set_to_none=True);baseline_optimizer.zero_grad(set_to_none=True)
    updates=0;step=0;last_block_nonzero=False;started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();baseline.train()
        order=DataLoader(train_data,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed+epoch))
        sums=np.zeros(5)
        for batch_index,batch in enumerate(order):
            video=batch['video_latents'].to(device);text=batch['text_embeddings'].to(device);target=batch['actions'].to(device)
            time_value=flow_time(len(target),device);noise=torch.randn_like(target)
            mixed=(1-time_value[:,None,None])*noise+time_value[:,None,None]*target
            # The final incomplete accumulation group uses its actual size.
            group_start=(batch_index//args.accumulation)*args.accumulation
            divisor=min(args.accumulation,len(order)-group_start)
            with autocast(device):
                context=model.encode_history(video,text)
                velocity=model.action_velocity(context,mixed,time_value)
                fm,groups=action_loss(velocity,target-noise)
                simple=baseline(video,text);base_loss,_=action_loss(simple,target)
            if not torch.isfinite(fm+base_loss):raise ValueError('Nonfinite action training loss')
            (fm/divisor).backward();(base_loss/divisor).backward()
            sigma=1-time_value;video_noise=torch.randn_like(video);noisy=(1-sigma[:,None,None,None,None])*video+sigma[:,None,None,None,None]*video_noise
            noisy[:,:,:2]=video[:,:,:2]
            with autocast(device):
                predicted_video=model.video_velocity(noisy,text,sigma)
                video_loss=(predicted_video[:,:,2:].float()-(video_noise-video)[:,:,2:].float()).square().mean()
            if not torch.isfinite(video_loss):raise ValueError('Nonfinite video flow loss')
            (args.video_weight*video_loss/divisor).backward()
            if (batch_index+1)%args.accumulation==0 or batch_index+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite WAM gradient')
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.)):raise ValueError('Nonfinite baseline gradient')
                gradients=[p.grad for p in backbone_parameters if p.grad is not None]
                last_block_nonzero|=any(g.abs().sum().item()>0 for g in gradients)
                optimizer.step();baseline_optimizer.step();optimizer.zero_grad(set_to_none=True);baseline_optimizer.zero_grad(set_to_none=True);updates+=1
            values=[float(fm.detach()),float(base_loss.detach()),float(video_loss.detach()),float(groups[0].detach()),float(groups[1].detach())];sums+=values;step+=1
            event={'epoch':epoch,'microstep':step,'optimizer_updates':updates,'flow_loss':values[0],'baseline_loss':values[1],
                'video_flow_loss':values[2],'lower_position_flow_loss':values[3],'wrist_flow_loss':values[4],'rotation_flow_loss':float(groups[2].detach())}
            log.write(json.dumps(event)+'\n')
            for key,value in event.items():
                if key.endswith('loss'):writer.add_scalar('train/'+key,value,step)
            if step%16==0:print(f'epoch={epoch} microstep={step} updates={updates} flow={values[0]:.4f} video={values[2]:.4f}',flush=True)
        report=evaluate(model,baseline,validation,device,seed=0,steps=args.denoising_steps,output=run/f'validation_epoch_{epoch:03d}.json')
        score=report['metrics']['wam']['overall']['recording_macro_ade_mm']
        for name,entry in report['metrics'].items():writer.add_scalar('validation/'+name+'_macro_ade_mm',entry['overall']['recording_macro_ade_mm'],epoch)
        if score<best:best=score;best_epoch=epoch;save_models(model,baseline,run/'best',include=('wam',))
        baseline_score=report['metrics']['rgb_text_baseline']['overall']['recording_macro_ade_mm']
        if baseline_score<best_baseline:
            best_baseline=baseline_score;best_baseline_epoch=epoch;save_models(model,baseline,run/'best',include=('baseline',))
        print(f'epoch={epoch} validation_macro_ADE_mm={score:.3f} best_epoch={best_epoch}',flush=True)
    save_models(model,baseline,run/'last');writer.close();log.close()
    change=sum((p.detach().cpu()-initial[k]).square().sum().item() for k,p in model.backbone.transformer_blocks[-1].named_parameters())**.5
    if not last_block_nonzero or change==0:raise ValueError('Cosmos last-block weights did not actually train')
    if any(p.grad is not None for p in model.backbone.transformer_blocks[0].parameters()):raise ValueError('Frozen backbone unexpectedly received gradients')
    preserved_mimic(train_data.protocol)
    result={'completed':True,'microsteps':step,'optimizer_updates':updates,'epochs':args.epochs,'selected_epoch':best_epoch,
        'best_validation_macro_ade_mm':best,'initial_validation_macro_ade_mm':initial_report['metrics']['wam']['overall']['recording_macro_ade_mm'],
        'baseline_selected_epoch':best_baseline_epoch,'baseline_best_validation_macro_ade_mm':best_baseline,
        'last_block_parameter_change_l2':change,'last_block_nonzero_gradient':last_block_nonzero,'causal_history_check':True,
        'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started,
        'mimic_preserved':True,'base_weights_are_pretrained':True,'checkpoint':'best/wam.safetensors'}
    write_json(run/'training_summary.json',result);print(json.dumps(result),flush=True)


def evaluate_run(args):
    run=Path(args.run_root);config=json.loads((run/'run.json').read_text())
    if sha256(args.protocol)!=config['protocol_sha256']:raise ValueError('Evaluation protocol changed after training')
    device=torch.device('cuda:0');model=CosmosWAM(load_backbone(args.backbone,args.backbone_manifest,device)).to(device);baseline=LinearBaseline().to(device)
    load_trainable(model,run/'best/wam.safetensors');load_trainable(baseline,run/'best/baseline.safetensors')
    reports=[]
    for split in ['val','test']:
        data=PilotDataset(args.protocol,split,config['eval_per_episode'])
        for seed in [0,1,2]:
            report=evaluate(model,baseline,data,device,seed=seed,steps=config['denoising_steps'],
                output=run/f'evaluation_{split}_seed{seed}.json',save_predictions=seed==0,diagnose_overlap=seed==0)
            reports.append({'split':split,'seed':seed,'wam':report['metrics']['wam']['overall'],
                'rgb_text_baseline':report['metrics']['rgb_text_baseline']['overall']})
            print(split,'seed',seed,'ADE mm',report['metrics']['wam']['overall']['ade_mm'],flush=True)
    write_json(run/'evaluation_summary.json',{'complete':True,'reports':reports,'selection':'validation only; test never selects checkpoints',
        'limits':'Operation recordings share some participants with training; walking retains official participant holdouts. This small pilot is not a robot closed-loop success benchmark.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['protocol','train','evaluate','visualize'])
    parser.add_argument('--protocol',type=Path);parser.add_argument('--base-data',type=Path);parser.add_argument('--operation-data',type=Path)
    parser.add_argument('--operation-manifest',type=Path);parser.add_argument('--raw-roots',nargs=2,type=Path)
    parser.add_argument('--backbone',type=Path);parser.add_argument('--backbone-manifest',type=Path);parser.add_argument('--run-root',type=Path)
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--train-per-episode',type=int,default=8)
    parser.add_argument('--eval-per-episode',type=int,default=4);parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=3e-4);parser.add_argument('--backbone-learning-rate',type=float,default=1e-5)
    parser.add_argument('--video-weight',type=float,default=.1);parser.add_argument('--denoising-steps',type=int,default=16);parser.add_argument('--seed',type=int,default=20261118)
    args=parser.parse_args()
    if min(args.epochs,args.train_per_episode,args.eval_per_episode,args.accumulation,args.denoising_steps)<1:
        parser.error('Epochs, samples, accumulation and denoising steps must be positive')
    if args.stage=='protocol':
        if not all([args.base_data,args.operation_data,args.operation_manifest,args.protocol,args.raw_roots]):parser.error('protocol needs both data roots, both raw roots, manifest and output path')
        result=build_protocol(args.base_data,args.operation_data,args.operation_manifest,args.protocol.parent,args.raw_roots);print(json.dumps(result['counts']))
    elif args.stage=='visualize':
        from .visualization import visualize
        visualize(args.protocol,args.run_root)
    elif args.stage=='train':train(args)
    else:evaluate_run(args)


if __name__=='__main__':main()
