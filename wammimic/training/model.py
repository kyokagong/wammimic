"""Cosmos-Predict2.5 history conditioning and a human 45D flow-matching expert."""
import json
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from wammimic.data_pipeline.common import sha256


def load_backbone(root,manifest,device):
    from diffusers import CosmosTransformer3DModel
    root=Path(root);manifest=json.loads(Path(manifest).read_text())
    if manifest['backbone']!='nvidia/Cosmos-Predict2.5-2B':raise ValueError('Wrong backbone identity')
    for row in manifest['files']:
        path=root/row['path']
        if not path.is_file() or path.stat().st_size!=row['size']:raise ValueError('Incomplete backbone checkpoint')
        if row.get('sha256') and sha256(path)!=row['sha256']:raise ValueError('Backbone checksum mismatch')
        if not row.get('sha256') and hashlib.sha1(f"blob {row['size']}\0".encode()+path.read_bytes()).hexdigest()!=row['blob_sha1']:
            raise ValueError('Backbone configuration checksum mismatch')
    backbone,loading=CosmosTransformer3DModel.from_pretrained(root,subfolder='transformer',
        torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('Backbone did not load strictly: '+str(loading))
    expected={'num_layers':28,'num_attention_heads':16,'attention_head_dim':128,'in_channels':17,
        'out_channels':16,'crossattn_proj_in_channels':100352,'use_crossattn_projection':True}
    if any(getattr(backbone.config,k)!=v for k,v in expected.items()):raise ValueError('Unexpected Predict2.5-2B architecture')
    count=sum(p.numel() for p in backbone.parameters())
    if not 1_900_000_000<count<2_200_000_000:raise ValueError('Unexpected backbone parameter count')
    return backbone.to(device)


class CosmosWAM(nn.Module):
    def __init__(self,backbone,width=256,layers=3,*,train_backbone=True,unfreeze_blocks=None,use_proprio=False,num_embodiments=0):
        super().__init__();self.backbone=backbone;self.backbone.requires_grad_(False)
        if unfreeze_blocks is None:
            self.unfreeze_blocks=1 if train_backbone else 0
        else:
            self.unfreeze_blocks=int(unfreeze_blocks)
        self.train_backbone=self.unfreeze_blocks>0
        self.use_proprio=bool(use_proprio)
        self.num_embodiments=int(num_embodiments)
        # FP32 master weights make small last-block updates survive BF16 quantization.
        if self.unfreeze_blocks>0:
            for block in self.backbone.transformer_blocks[-self.unfreeze_blocks:]:
                block.float().requires_grad_(True)
        hidden=backbone.config.num_attention_heads*backbone.config.attention_head_dim
        self.context=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,width))
        self.action_in=nn.Linear(45,width);self.action_out=nn.Linear(width,45)
        self.queries=nn.Parameter(torch.randn(1,10,width)*.02)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,8,width*4,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True),layers,norm=nn.LayerNorm(width))
        if self.use_proprio:
            self.proprio_enc=nn.Sequential(nn.LayerNorm(75),nn.Linear(75,width),nn.SiLU(),nn.Linear(width,width))
            self.ee_hist_enc=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,width))
            self.proprio_pos=nn.Parameter(torch.randn(1,4,width)*.02)
        if self.num_embodiments>0:
            self.embodiment_emb=nn.Embedding(self.num_embodiments,width)
            self.action_in_ee=nn.ModuleList([nn.Linear(2,width) for _ in range(self.num_embodiments)])
            self.ee_out=nn.ModuleList([nn.Linear(width,2) for _ in range(self.num_embodiments)])
        self.width=width;self._tokens=None
        self.backbone.transformer_blocks[-1].register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):self._tokens=output

    def encode_history(self,video,text,proprio=None,ee_hist=None,embodiment_id=None):
        # Only two causal latents enter this branch; future pixels never reach the expert.
        history=video[:,:,:2];mask=torch.ones_like(history[:,:1])
        timestep=torch.full((len(video),1,2,1,1),.0001,device=video.device,dtype=video.dtype)
        padding=torch.zeros((1,1,*video.shape[-2:]),device=video.device,dtype=video.dtype)
        self._tokens=None
        with torch.set_grad_enabled(self.train_backbone):
            self.backbone(history,timestep,text,condition_mask=mask,padding_mask=padding,fps=10,return_dict=False)
        tokens=self._tokens;self._tokens=None
        if tokens is None:raise RuntimeError('Missing Cosmos history features')
        ctx=self.context(tokens)
        extras=[]
        if self.use_proprio and proprio is not None:
            p_tok=self.proprio_enc(proprio.to(ctx.dtype))+self.proprio_pos.to(ctx.dtype)
            if ee_hist is not None:
                p_tok=p_tok+self.ee_hist_enc(ee_hist.to(ctx.dtype))
            extras.append(p_tok)
        if self.num_embodiments>0 and embodiment_id is not None:
            extras.append(self.embodiment_emb(embodiment_id).to(ctx.dtype)[:,None,:])
        return torch.cat([ctx,*extras],dim=1) if extras else ctx

    def action_velocity(self,context,noisy_actions,time,embodiment_id=None):
        frequency=torch.exp(-math.log(10000)*torch.arange(self.width//2,device=time.device)/(self.width//2-1))
        angle=time[:,None].float()*frequency[None]*1000
        embedding=self.time(torch.cat([angle.sin(),angle.cos()],-1))
        body_in=noisy_actions[...,:45]
        hidden=self.action_in(body_in)
        if self.num_embodiments>0 and embodiment_id is not None:
            emb_vec=self.embodiment_emb(embodiment_id).to(hidden.dtype)
            hidden=hidden+emb_vec[:,None,:]
            if noisy_actions.shape[-1]==47:
                ee_in=noisy_actions[...,45:47]
                ee_h=torch.zeros_like(hidden)
                for eid in range(self.num_embodiments):
                    sel=(embodiment_id==eid)
                    if sel.any():
                        ee_h[sel]=self.action_in_ee[eid](ee_in[sel])
                hidden=hidden+ee_h
        hidden=hidden+self.queries.to(hidden.dtype)+embedding[:,None].to(hidden.dtype)
        decoded=self.decoder(hidden,context)
        body_out=self.action_out(decoded)
        if self.num_embodiments>0 and embodiment_id is not None and noisy_actions.shape[-1]==47:
            ee_pred=torch.zeros((len(decoded),decoded.shape[1],2),device=decoded.device,dtype=body_out.dtype)
            for eid in range(self.num_embodiments):
                sel=(embodiment_id==eid)
                if sel.any():
                    ee_pred[sel]=self.ee_out[eid](decoded[sel]).to(body_out.dtype)
            return torch.cat([body_out,ee_pred],dim=-1)
        return body_out

    def forward(self,video,text,noisy_actions,time,*,proprio=None,ee_hist=None,embodiment_id=None):
        context=self.encode_history(video,text,proprio=proprio,ee_hist=ee_hist,embodiment_id=embodiment_id)
        return self.action_velocity(context,noisy_actions,time,embodiment_id=embodiment_id)

    def video_velocity(self,noisy_video,text,sigma):
        mask=torch.zeros_like(noisy_video[:,:1]);mask[:,:,:2]=1
        timestep=mask[:,:,:,:1,:1]*.0001+(1-mask[:,:,:,:1,:1])*sigma[:,None,None,None,None]
        padding=torch.zeros((1,1,*noisy_video.shape[-2:]),device=noisy_video.device,dtype=noisy_video.dtype)
        result=self.backbone(noisy_video,timestep,text,condition_mask=mask,padding_mask=padding,fps=10,return_dict=False)[0]
        self._tokens=None
        return result

    @torch.no_grad()
    def sample(self,context,noise,steps=16,embodiment_id=None,prefix_target=None,prefix_weights=None):
        if steps<1:raise ValueError('At least one flow integration step is required')
        action=noise.clone()
        use_rtc=prefix_target is not None and prefix_weights is not None
        if use_rtc:
            w=prefix_weights.to(device=action.device,dtype=action.dtype)
            if w.ndim==1:w=w[None,:,None]
            elif w.ndim==2:w=w[:,:,None]
            p_tgt=prefix_target.to(device=action.device,dtype=action.dtype)
        for i in range(steps):
            t_val=i/steps
            if use_rtc and i>0:
                noisy_prefix=(1.0-t_val)*noise+t_val*p_tgt
                action=(1.0-w)*action+w*noisy_prefix
            time=torch.full((len(action),),t_val,device=action.device)
            action=action+self.action_velocity(context,action,time,embodiment_id=embodiment_id)/steps
        if use_rtc:
            action=(1.0-w)*action+w*p_tgt
        return action.float()


class LinearBaseline(nn.Module):
    """Same RGB/text observations, without access to measured past body poses."""
    def __init__(self,text_dim=100352):
        super().__init__();self.text_dim=text_dim
        text_features=3584 if text_dim==100352 else text_dim
        self.head=nn.Sequential(nn.Linear(32+text_features,256),nn.GELU(),nn.Linear(256,450))

    def forward(self,video,text):
        history=video[:,:,:2].float().mean((-1,-2)).flatten(1)
        language=text.float().mean(1)
        if self.text_dim==100352:language=language.reshape(len(text),28,3584).mean(1)
        return self.head(torch.cat([history,language],-1)).reshape(-1,10,45)


def action_loss(prediction,target,ee_mask=None):
    error=(prediction.float()-target.float()).square()
    groups=torch.stack([error[...,:9].mean(),error[...,9:15].mean(),error[...,15:45].mean()])
    loss=groups.mean()
    if prediction.shape[-1]==47 and ee_mask is not None:
        m=ee_mask.float().view(-1,1,1)
        denom=m.sum().clamp_min(1.0)*prediction.shape[1]*2.0
        ee_loss=(error[...,45:47]*m).sum()/denom
        loss=loss+0.5*ee_loss
        return loss,torch.cat([groups,ee_loss.view(1)])
    return loss,groups


def flow_time(batch,device):
    time=torch.randn(batch,device=device).sigmoid()
    return 5*time/(1+4*time)
