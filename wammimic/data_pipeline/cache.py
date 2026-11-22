"""Checkpoint-matched Cosmos-Predict2.5 caches, without the distributed SDK."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from .common import sha256, write_json

VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
REASON_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
CORE_COMMIT = "a2c298b0a3df3778b973fe65e9e58877b292d8a7"
RECIPE = "cosmos-predict25-wam17-v1"
MEAN = [-.7571,-.7089,-.9113,.1075,-.1745,.9653,-.1517,1.5508,.4134,-.0715,.5517,-.3632,-.1922,-.9497,.2503,-.2921]
STD = [2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160]


def clip_frame_indices(index):
    """Five conditioning pixels (four unique) + twelve future pixels = 17.

    Duplicating the first history frame puts t0 exactly at latent index 1.
    The two conditioned latent frames therefore contain no future pixels.
    """
    if index < 3:
        raise ValueError("Four history frames required")
    return np.r_[index-3,np.arange(index-3,index+13)].astype(np.int64)


def concatenate_reason_layers(hidden_states):
    # Same sample-standard-deviation normalization and layer ordering as the SDK.
    layers=hidden_states[1:]
    return torch.cat([(x-x.mean(dim=-1,keepdim=True))/(x.std(dim=-1,keepdim=True)+1e-8) for x in layers],dim=-1)


def prompt_ids(tokenizer,text):
    messages=[{"role":"system","content":[{"type":"text","text":"You are a helpful assistant who will provide prompts to an image generator."}]},
              {"role":"user","content":[{"type":"text","text":text}]}]
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=True)["input_ids"][:512]
    if not all(isinstance(i,int) for i in ids):raise ValueError("Unexpected chat token layout")
    return ids+[tokenizer.pad_token_id]*(512-len(ids))


def model_manifest(root):
    return json.loads((Path(root)/"models.json").read_text())


def cache_owner(marker,identity,existing_paths):
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError("Cache belongs to a different recipe/source")
    else:
        if any(path.exists() for path in existing_paths):raise FileExistsError("Preserved cache without ownership marker")
        write_json(marker,identity)


def verify_components(root,kind):
    root=Path(root)
    manifest=model_manifest(root)
    for f in manifest["files"]:
        if (kind=="video") != (f["path"]=="tokenizer.pth"):
            continue
        path=root/f["path"]
        if not path.is_file() or path.stat().st_size!=f["size"]:
            raise FileNotFoundError(f"Missing complete model component: {f['path']}")
        if f.get("sha256") and sha256(path)!=f["sha256"]:
            raise ValueError(f"Invalid model component: {f['path']}")
    return manifest


def cache_text(model_root,tasks_path,output):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    model_root,output=Path(model_root),Path(output)
    manifest=verify_components(model_root,"text")
    if any(f["revision"]!=REASON_REVISION for f in manifest["files"] if f["path"].startswith("reason1/")):
        raise ValueError("Reason1 revision differs from official Cosmos checkpoint registry")
    tasks={str(r["task_index"]):r["task"] for r in map(json.loads,Path(tasks_path).read_text().splitlines())}
    identity={"recipe":RECIPE,"backbone":"Cosmos-Predict2.5-2B","text_encoder":"nvidia/Cosmos-Reason1-7B",
        "revision":REASON_REVISION,"tasks":tasks,"shape":[512,100352],"dtype":"bfloat16",
        "weight_loading_verified":True,
        "normalization":"per-token unbiased std, concatenate hidden layers 1..28",
        "source_commit":CORE_COMMIT}
    bank=output/"prompt_bank_bf16.safetensors";receipt=output/"prompt_bank.json"
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if any(saved.get(k)!=v for k,v in identity.items()) or sha256(bank)!=saved["sha256"]:
            raise ValueError("Text cache identity or checksum changed")
        return saved
    cache_owner(output/"prompt_bank.source.json",identity,[bank])
    tokenizer=AutoTokenizer.from_pretrained(model_root/"reason1",local_files_only=True)
    model,loading=AutoModelForImageTextToText.from_pretrained(model_root/"reason1",local_files_only=True,
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa",output_loading_info=True)
    if any(loading.get(key) for key in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs","conversion_errors"]):
        raise ValueError(f"Incomplete Reason1 weight loading: {loading}")
    model.eval().requires_grad_(False)
    import struct
    temporary=bank.with_suffix(".tmp")
    tensor_bytes=512*100352*2
    ordered_tasks=list(tasks.items())
    header={"__metadata__":{"recipe":RECIPE,"revision":REASON_REVISION}}
    for idx,(task,_) in enumerate(ordered_tasks):
        header["task_"+task]={"dtype":"BF16","shape":[512,100352],"data_offsets":[idx*tensor_bytes,(idx+1)*tensor_bytes]}
    header_bytes=json.dumps(header,separators=(",",":")).encode("utf-8")
    with temporary.open("wb") as dst:
        dst.write(struct.pack("<Q",len(header_bytes)))
        dst.write(header_bytes)
        with torch.inference_mode():
            for task,text in ordered_tasks:
                ids=prompt_ids(tokenizer,text)
                tokens=torch.tensor([ids],device="cuda:0")
                result=model(input_ids=tokens,output_hidden_states=True,use_cache=False,return_dict=True)
                if len(result.hidden_states)!=29:
                    raise ValueError("Expected 28 Reason1 decoder layers")
                embedding=concatenate_reason_layers(result.hidden_states)[0]
                if embedding.shape!=(512,100352) or not torch.isfinite(embedding).all():
                    raise ValueError("Invalid Cosmos text embedding")
                cpu_t=embedding.to(device="cpu",dtype=torch.bfloat16).contiguous()
                dst.write(cpu_t.view(torch.uint8).numpy().tobytes())
                del tokens,result,embedding,cpu_t
                print(f"Text task {task}: cached",flush=True)
    temporary.chmod(0o644)
    os.replace(temporary,bank)
    report={**identity,"sha256":sha256(bank),"tasks_cached":len(tasks),"complete":True}
    write_json(receipt,report)
    return report


class VideoEncoder:
    def __init__(self,model_root):
        from wammimic._vendor.cosmos_wan import WanVAE_
        verify_components(model_root,"video")
        path=Path(model_root)/"tokenizer.pth"
        if sha256(path)!=VAE_SHA256:
            raise ValueError("VAE is not the exact Cosmos-Predict2.5 tokenizer")
        with torch.device("meta"):
            self.model=WanVAE_(dim=96,z_dim=16,dim_mult=[1,2,4,4],num_res_blocks=2,
                              attn_scales=[],temperal_downsample=[False,True,True],dropout=0.,temporal_window=4)
        self.model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True,assign=True)
        self.model.to(device="cuda:0",dtype=torch.bfloat16).eval().requires_grad_(False)
        self.scale=[torch.tensor(MEAN,device="cuda:0",dtype=torch.bfloat16),
                    1/torch.tensor(STD,device="cuda:0",dtype=torch.bfloat16)]

    @torch.inference_mode()
    def encode(self,pixels):
        if pixels.shape[1:]!=(3,17,224,224):
            raise ValueError("Expected Bx3x17x224x224 RGB pixels")
        result=self.model.encode(pixels.to(device="cuda:0",dtype=torch.bfloat16),self.scale)
        if result.shape[1:]!=(16,5,28,28) or not torch.isfinite(result).all():
            raise ValueError("Invalid Cosmos video latent")
        return result


def check_causal_prefix(encoder,pixels):
    paired=pixels[:1].repeat(2,1,1,1,1)
    paired[1,:,5:]=-1
    latents=encoder.encode(paired)
    if not torch.allclose(latents[0,:,:2],latents[1,:,:2],atol=1e-5,rtol=1e-5):
        raise ValueError("Future video leaked into conditioned latent frames")


def read_video(path,frames):
    result=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","rawvideo","-pix_fmt","rgb24","pipe:1"],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    expected=frames*224*224*3
    if len(result.stdout)!=expected:
        raise ValueError(f"Video/frame table mismatch: {path}")
    return np.frombuffer(result.stdout,dtype=np.uint8).reshape(frames,224,224,3)


def cache_videos(model_root,output,batch_size=2,*,dataset_name="g1_parallel_gripper"):
    if dataset_name not in {"g1_parallel_gripper","human_ego","q1_stub","nymeria_ego"}:raise ValueError("Unknown WAM dataset")
    output=Path(output);dataset=output/"wam_lerobot_v2"/dataset_name
    info=json.loads((dataset/"meta/info.json").read_text())
    if info["world_model"]!="Cosmos-Predict2.5-2B" or info["fps"]!=10:
        raise ValueError("Unsupported WAM video contract")
    encoder=None;episodes=[];causal_checked=False
    rows=[json.loads(line) for line in (dataset/"meta/episodes.jsonl").read_text().splitlines()]
    latent_root=dataset/"latents"
    cache_owner(latent_root/"source.json",{"recipe":RECIPE,"vae_sha256":VAE_SHA256,
        "dataset_source_sha256":sha256(dataset/"source.json"),"source_commit":CORE_COMMIT},
        list(latent_root.glob("chunk-*")))
    only_eps=os.environ.get("VIDEO_CACHE_EPISODES","").strip()
    allowed={int(x) for x in only_eps.split(",") if x.strip()} if only_eps else None
    for row in rows:
        episode=row["episode_index"]
        if allowed is not None and episode not in allowed:
            continue
        stem=f"episode_{episode:06d}"
        video=dataset/f"videos/chunk-000/observation.images.cam_head/{stem}.mp4"
        parquet=dataset/f"data/chunk-000/{stem}.parquet"
        dest=dataset/f"latents/chunk-000/{stem}.safetensors";receipt=dest.with_suffix(".json")
        identity={"recipe":RECIPE,"source_video_sha256":sha256(video),"source_parquet_sha256":sha256(parquet),
            "vae_sha256":VAE_SHA256,"frames":17,"fps":10,"conditioned_latent_frames":2,
            "shape":[16,5,28,28],"source_commit":CORE_COMMIT,"dtype":"bfloat16","causal_conditioning_verified":True}
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if any(saved.get(k)!=v for k,v in identity.items()) or sha256(dest)!=saved["sha256"]:
                raise ValueError(f"Changed cached episode {episode}")
            episodes.append(saved);continue
        table=pq.read_table(parquet,columns=["valid_for_training","frame_index"])
        indices=np.flatnonzero(table["valid_for_training"].to_numpy())
        indices=indices[indices+12<row["length"]]
        if not len(indices):raise ValueError(f"No causal training windows in {episode}")
        pixels=read_video(video,row["length"])
        encoder=encoder or VideoEncoder(model_root)
        latents=[]
        for start in range(0,len(indices),batch_size):
            batch=indices[start:start+batch_size]
            frame_map=np.stack([clip_frame_indices(int(i)) for i in batch])
            tensor=torch.from_numpy(pixels[frame_map].copy()).permute(0,4,1,2,3).float()/127.5-1
            if not causal_checked:
                check_causal_prefix(encoder,tensor)
                causal_checked=True
            latents.append(encoder.encode(tensor).cpu())
            if (start//batch_size)%32==0:
                print(f"Video episode {episode}: {min(start+batch_size,len(indices))}/{len(indices)} windows",flush=True)
        tensor=torch.cat(latents).contiguous()
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix(".tmp")
        save_file({"latents":tensor,"frame_index":torch.from_numpy(indices.astype(np.int64))},str(temporary),
                  metadata={"recipe":RECIPE,"conditioned_latent_frames":"2","vae_sha256":VAE_SHA256})
        temporary.chmod(0o644)
        os.replace(temporary,dest)
        entry={**identity,"episode_index":episode,"samples":len(indices),"sha256":sha256(dest),
               "path":str(dest.relative_to(output))}
        write_json(receipt,entry);episodes.append(entry)
        print(f"Video episode {episode}: {len(indices)} windows",flush=True)
    if allowed is not None:
        return {"partial":True,"episodes":len(episodes),"samples":sum(e["samples"] for e in episodes)}
    report={"complete":len(episodes)==len(rows),"recipe":RECIPE,"episodes":episodes,
            "samples":sum(e["samples"] for e in episodes),"excluded_tail":"two extra visual future frames required for causal 4k+1 alignment"}
    write_json(output/"qa_reports/cache_video.json",report)
    return report
