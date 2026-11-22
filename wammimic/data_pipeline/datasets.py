"""Training readers for the prepared trajectory and latent contracts."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from .geometry import mimic_window


class MimicDataset(Dataset):
    """One robot per reader: joint targets have a fixed, recorded joint order."""
    def __init__(self,root,robot,split="train"):
        self.root=Path(root)
        manifest=json.loads((self.root/robot/"manifest.json").read_text())
        self.clips=[r for r in manifest["clips"] if r["split"]==split]
        self.ends=np.cumsum([r["frames"] for r in self.clips])
        if not self.clips:raise ValueError(f"No {robot}/{split} clips")
        if len({tuple(r["joint_names"]) for r in self.clips})!=1:
            raise ValueError("Mixed joint orders within robot reader")
        self.joint_names=self.clips[0]["joint_names"]
        self._loaded=None

    def __len__(self):return int(self.ends[-1])

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        clip=int(np.searchsorted(self.ends,index,side="right"))
        frame=index-(int(self.ends[clip-1]) if clip else 0)
        row=self.clips[clip]
        if self._loaded!=clip:
            self._poses=np.load(self.root/row["path"]/"poses.npy",mmap_mode="r")
            self._joints=np.load(self.root/row["path"]/"joints.npy",mmap_mode="r")
            self._loaded=clip
        floor=row.get("reference_floor_offset_m",0.)
        window=mimic_window(self._poses,frame,row["leg_length"],floor_height=floor)
        root_pose=self._poses[frame,0].copy();root_pose[2]-=floor
        return {"five_point_window":torch.from_numpy(window),
                "joint_pos":torch.from_numpy(self._joints[frame,0].copy()),
                "joint_vel":torch.from_numpy(self._joints[frame,1].copy()),
                "root_pose":torch.from_numpy(root_pose),
                "reference_floor_valid":torch.tensor(row.get("reference_floor_valid",False),dtype=torch.bool)}


class WAMDataset(Dataset):
    """Cached inputs with a causal five-frame latent and ten-step action block."""
    def __init__(self,root,split="train",*,dataset_name="g1_parallel_gripper",episode_ids=None,statistics=None):
        self.root=Path(root);self.dataset=self.root/"wam_lerobot_v2"/dataset_name
        self.human=dataset_name in {"human_ego","nymeria_ego"}
        source_meta=json.loads((self.dataset/"source.json").read_text())
        self.history=self.human and "observation_state" in source_meta
        self.has_5pt_hist=(not self.human) and ("observation.state_5pt_hist" in source_meta.get("extra_state_columns", []))
        self.episodes={r["episode_index"]:r for r in map(json.loads,(self.dataset/"meta/episodes.jsonl").read_text().splitlines())}
        selected=json.loads((self.dataset/"meta/splits.json").read_text())[split] if episode_ids is None else list(episode_ids)
        if len(selected)!=len(set(selected)) or any(i not in self.episodes for i in selected):
            raise ValueError("Unknown or repeated selected episode")
        self.samples=[]
        for episode in selected:
            with safe_open(str(self._latent_path(episode)),framework="pt",device="cpu") as source:
                frames=source.get_tensor("frame_index").tolist()
            self.samples.extend((episode,i,frame) for i,frame in enumerate(frames))
        if not self.samples:raise ValueError(f"No cached {split} samples")
        self.statistics=json.loads((self.dataset/"meta/stats.json").read_text()) if statistics is None else statistics
        self._loaded=None

    def _latent_path(self,episode):
        return self.dataset/f"latents/chunk-000/episode_{episode:06d}.safetensors"

    def __len__(self):return len(self.samples)

    def _normalize(self,key,tensor):
        stats=self.statistics[key]
        mean=torch.tensor(stats["mean"],dtype=tensor.dtype)
        std=torch.tensor(stats["std"],dtype=tensor.dtype).clamp_min(1e-6)
        return (tensor-mean)/std

    def __getitem__(self,index):
        episode,sample,frame=self.samples[index]
        if self._loaded!=episode:
            cols=["action", "task_index" if self.human else "observation.state"] + (["observation.state"] if self.history else []) + (["observation.state_5pt_hist"] if self.has_5pt_hist else [])
            table=pq.read_table(self.dataset/f"data/chunk-000/episode_{episode:06d}.parquet", columns=cols)
            if self.human:self._tasks=table["task_index"].to_numpy()
            if not self.human or self.history:self._states=np.array(table["observation.state"].to_pylist(),dtype=np.float32)
            if self.has_5pt_hist:self._states_5pt=np.array(table["observation.state_5pt_hist"].to_pylist(),dtype=np.float32)
            if self.human and table["action"].type == pa.list_(pa.list_(pa.float32(),45),10):
                self._actions=table["action"].combine_chunks().flatten().flatten().to_numpy().reshape(-1,10,45)
            else:
                self._actions=np.array(table["action"].to_pylist(),dtype=np.float32)
            self._loaded=episode
        with safe_open(str(self._latent_path(episode)),framework="pt",device="cpu") as source:
            latents=source.get_slice("latents")[sample]
        task=int(self._tasks[frame]) if self.human else self.episodes[episode]["task_index"]
        with safe_open(str(self.root/"prompt_bank_bf16.safetensors"),framework="pt",device="cpu") as bank:
            text=bank.get_tensor("task_"+str(task))
        mask=torch.zeros((1,5,28,28));mask[:,:2]=1
        if self.human:
            return {"video_latents":latents,"video_condition_mask":mask,"text_embeddings":text,
                    "actions":self._normalize("action",torch.from_numpy(self._actions[frame].copy()))} | (
                    {"proprio":self._normalize("observation.state",torch.from_numpy(self._states[frame].copy()))} if self.history else {})
        item={"video_latents":latents,"video_condition_mask":mask,"text_embeddings":text,
              "proprio":self._normalize("observation.state",torch.from_numpy(self._states[frame].copy())),
              "actions":self._normalize("action",torch.from_numpy(self._actions[frame].copy())),
              "embodiment_id":torch.tensor(0 if self.dataset.name=="q1_stub" else 1,dtype=torch.int64)}
        if self.has_5pt_hist:
            item["proprio_5pt"]=self._normalize("observation.state_5pt_hist",torch.from_numpy(self._states_5pt[frame].copy()))
        return item


class HumanWAMDataset(WAMDataset):
    """Human RGB/text -> 10x45 pose targets, with no robot state or invented EE."""
    def __init__(self,root,split="train",*,episode_ids=None,statistics=None):
        from .human_ego import SCHEMA,PROVENANCE,DATASET
        source=json.loads((Path(root)/DATASET/"source.json").read_text())
        if source.get("schema")!=SCHEMA or source.get("subject_kind")!="human" or source.get("label_provenance")!=PROVENANCE:
            raise ValueError("Human WAM requires measured optical skeleton provenance")
        super().__init__(root,split,dataset_name="human_ego",episode_ids=episode_ids,statistics=statistics)


class NymeriaWAMDataset(WAMDataset):
    """Inertial-reference input, kept separate from optical supervision."""
    def __init__(self,root,split='train',*,episode_ids=None,statistics=None):
        from .nymeria import DATASET,NAME,PROVENANCE
        source=json.loads((Path(root)/DATASET/'source.json').read_text())
        if source.get('schema')!='nymeria-inertial-five-point-v1' or source.get('label_provenance')!=PROVENANCE:
            raise ValueError('Explicit Nymeria inertial provenance required')
        super().__init__(root,split,dataset_name=NAME,episode_ids=episode_ids,statistics=statistics)
