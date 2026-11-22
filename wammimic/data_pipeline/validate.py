"""Verify training inputs, checksums, split isolation and a real loaded batch."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader

from .common import sha256,write_json
from .datasets import MimicDataset,WAMDataset


def validate_mimic(output):
    root=Path(output)/"morph_5pt_store";report={};groups={}
    for robot in ["q1","bumi","g1"]:
        clips=json.loads((root/robot/"manifest.json").read_text())["clips"]
        for row in clips:
            if "reference_floor_offset_m" not in row or "reference_floor_valid" not in row:
                raise ValueError("Missing Mimic reference-floor calibration")
            floor=row["reference_floor_offset_m"];known=row["reference_floor_valid"]
            if type(known) is not bool or not np.isfinite(floor) or abs(floor)>.12 or (not known and floor!=0):
                raise ValueError("Invalid Mimic reference-floor calibration")
            groups.setdefault(row["group_id"],set()).add(row["split"])
            folder=root/row["path"]
            for name,digest in row["output_sha256"].items():
                if sha256(folder/name)!=digest:raise ValueError(f"Changed Mimic file: {folder/name}")
        if any(len(s)!=1 for s in groups.values()):raise ValueError("Mimic split leakage")
        dataset=MimicDataset(root,robot)
        batch=next(iter(DataLoader(dataset,batch_size=2,shuffle=False)))
        if batch["five_point_window"].shape!=(2,465):raise ValueError("Wrong Mimic batch shape")
        if not all(torch.isfinite(t).all() for t in batch.values()):raise ValueError("Nonfinite Mimic batch")
        # A tiny differentiable consumer checks the loader's dtype and target contract.
        head=torch.nn.Linear(465,len(dataset.joint_names))
        loss=(head(batch["five_point_window"])-batch["joint_pos"]).square().mean()
        loss.backward()
        if not torch.isfinite(head.weight.grad).all():raise ValueError("Invalid Mimic gradients")
        report[robot]={"ready":True,"manifest_sha256":sha256(root/robot/"manifest.json"),"clips":len(clips),"train_frames":len(dataset),"batch_shape":[2,465],"finite_gradient":True,
                       "reference_floor_unknown_clips":sum(not r["reference_floor_valid"] for r in clips)}
    write_json(Path(output)/"qa_reports/mimic_training_ready.json",report)
    return report


def validate_wam(output):
    output=Path(output);dataset=output/"wam_lerobot_v2/g1_parallel_gripper"
    video=json.loads((output/"qa_reports/cache_video.json").read_text())
    text=json.loads((output/"prompt_bank.json").read_text())
    if not video["complete"] or not text["complete"]:raise ValueError("Incomplete caches")
    if sha256(output/"prompt_bank_bf16.safetensors")!=text["sha256"]:raise ValueError("Changed prompt bank")
    splits=json.loads((dataset/"meta/splits.json").read_text())
    memberships=[s for values in splits.values() for s in values]
    if len(memberships)!=len(set(memberships)):raise ValueError("WAM episode split leakage")
    episodes={r["episode_index"]:r for r in map(json.loads,(dataset/"meta/episodes.jsonl").read_text().splitlines())}
    samples=0
    for row in video["episodes"]:
        path=output/row["path"]
        if sha256(path)!=row["sha256"]:raise ValueError(f"Changed latent cache: {path}")
        episode=episodes[row["episode_index"]]
        for name,digest in episode["outputs"].items():
            if sha256(output/name)!=digest:raise ValueError(f"Changed paired source: {name}")
        with safe_open(str(path),framework="pt",device="cpu") as source:
            frames=source.get_tensor("frame_index").numpy()
            if (frames<3).any() or (frames+12>=episode["length"]).any():raise ValueError("Invalid causal visual window")
            if source.get_slice("latents").get_shape()!=[len(frames),16,5,28,28]:raise ValueError("Wrong latent dimensions")
            for start in range(0,len(frames),16):
                if not torch.isfinite(source.get_slice("latents")[start:start+16]).all():raise ValueError("Nonfinite latent")
        table=pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        actions=np.array(table["action"].to_pylist(),dtype=np.float32)[frames]
        rot=actions[...,:45][...,15:].reshape(-1,5,2,3)
        if not np.allclose(np.linalg.norm(rot,axis=-1),1,atol=1e-4) or not np.allclose((rot[:,:,0]*rot[:,:,1]).sum(axis=-1),0,atol=1e-4):
            raise ValueError("Nonorthogonal action rotations")
        if not np.isfinite(actions).all() or ((actions[...,45:]<0)|(actions[...,45:]>1)).any():raise ValueError("Invalid action targets")
        samples+=len(frames)
    reader=WAMDataset(output,"train")
    batch=next(iter(DataLoader(reader,batch_size=1)))
    expected={"video_latents":(1,16,5,28,28),"text_embeddings":(1,512,100352),"proprio":(1,68),"actions":(1,10,47)}
    if reader.has_5pt_hist:
        expected["proprio_5pt"]=(1,4,79)
    for key,shape in expected.items():
        if batch[key].shape!=shape or not torch.isfinite(batch[key]).all():raise ValueError(f"Invalid WAM batch: {key}")
    head=torch.nn.Linear(16,470)
    prediction=head(batch["video_latents"].float().mean(dim=(2,3,4)))
    loss=(prediction-batch["actions"].flatten(1)).square().mean();loss.backward()
    if not torch.isfinite(head.weight.grad).all():raise ValueError("Invalid WAM training gradients")
    report={"ready":True,"backbone":"Cosmos-Predict2.5-2B","episodes":len(episodes),"cached_samples":samples,
        "dataset_source_sha256":sha256(dataset/"source.json"),"episodes_sha256":sha256(dataset/"meta/episodes.jsonl"),
        "video_cache_report_sha256":sha256(output/"qa_reports/cache_video.json"),"text_cache_sha256":text["sha256"],
        "batch_shapes":{k:list(v) for k,v in expected.items()},"finite_gradient":True,
        "full_backbone_forward":"not run: main DiT weights require separate repository access; data contract is validated",
        "reference_limits":"OpenHLM root translation uses stance odometry; dynamic stability is not certified by this check"}
    write_json(output/"qa_reports/wam_training_ready.json",report)
    source_report=output/"qa_reports/openhlm.json"
    if source_report.exists():
        prepared=json.loads(source_report.read_text());prepared["dit_latents_ready"]=True
        prepared.pop("latent_blocker",None);write_json(source_report,prepared)
    return report
