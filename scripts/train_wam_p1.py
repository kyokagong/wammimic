"""Train the P1 multi-embodiment Cosmos-Predict2.5 WAM Action Expert from scratch.

Consumes all four verified P1 branches (human-protocol, sim, gripper, nymeria) with
4-frame causal 5-point proprioception history and GR00T N2-style embodiment routing.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
from queue import Queue
import threading
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from wammimic.data_pipeline.common import sha256, write_json
from wammimic.data_pipeline.datasets import HumanWAMDataset, NymeriaWAMDataset, WAMDataset
from wammimic.training.evaluation import autocast
from wammimic.training.model import CosmosWAM, action_loss, flow_time, load_backbone
from wammimic.training.run import trainable_state


class _FastBranchReader:
    """Wraps a validated WAM branch with zero-copy Arrow episode caching and prompt caching."""

    def __init__(self, reader, embodiment_id, ee_dim, ee_hist_dim, *, use_proprio_5pt=False, max_cached_prompts=128, max_open_latents=128):
        self.reader = reader
        self.embodiment_id = int(embodiment_id)
        self.ee_dim = int(ee_dim)
        self.ee_hist_dim = int(ee_hist_dim)
        self.use_proprio_5pt = bool(use_proprio_5pt)
        self.max_cached_prompts = int(max_cached_prompts)
        self.max_open_latents = int(max_open_latents)
        self._prompt_cache = {}
        self._prompt_file = None
        self._ep_cache = {}
        self._latent_handles = OrderedDict()
        self._lock = threading.Lock()
        self._act_dim = 45 + self.ee_dim
        self._st_dim = 75 + self.ee_hist_dim

    def __len__(self):
        return len(self.reader)

    def _get_prompt(self, task_idx):
        with self._lock:
            if task_idx not in self._prompt_cache:
                if self._prompt_file is None:
                    self._prompt_file = safe_open(str(self.reader.root / "prompt_bank_bf16.safetensors"), framework="pt", device="cpu")
                t = self._prompt_file.get_tensor("task_" + str(task_idx))
                if len(self._prompt_cache) >= self.max_cached_prompts:
                    self._prompt_cache.pop(next(iter(self._prompt_cache)))
                self._prompt_cache[task_idx] = t
            return self._prompt_cache[task_idx]

    def _get_episode_arrays(self, episode):
        with self._lock:
            entry = self._ep_cache.get(episode)
            if entry is not None:
                return entry
        r = self.reader
        state_col = "observation.state_5pt_hist" if self.use_proprio_5pt else "observation.state"
        cols = ["action", state_col] + (["task_index"] if r.human else [])
        table = pq.read_table(r.dataset / f"data/chunk-000/episode_{episode:06d}.parquet", columns=cols)
        tasks = table["task_index"].to_numpy() if r.human else None
        states = table[state_col].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 4, self._st_dim)
        actions = table["action"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(-1, 10, self._act_dim)
        entry = (tasks, states, actions)
        with self._lock:
            self._ep_cache[episode] = entry
        return entry

    def _read_latent(self, episode, sample_idx):
        with self._lock:
            src = self._latent_handles.get(episode)
            if src is not None:
                self._latent_handles.move_to_end(episode)
            else:
                src = safe_open(str(self.reader._latent_path(episode)), framework="pt", device="cpu")
                self._latent_handles[episode] = src
                if len(self._latent_handles) > self.max_open_latents:
                    self._latent_handles.popitem(last=False)
        return src.get_slice("latents")[sample_idx].clone()

    def sample(self, index):
        r = self.reader
        episode, sample_idx, frame = r.samples[index]
        tasks, states, actions = self._get_episode_arrays(episode)
        latents = self._read_latent(episode, sample_idx)
        task = int(tasks[frame]) if r.human else r.episodes[episode]["task_index"]
        text = self._get_prompt(task)

        state_key = "observation.state_5pt_hist" if self.use_proprio_5pt else "observation.state"
        st = r._normalize(state_key, torch.from_numpy(states[frame].copy()))
        act = r._normalize("action", torch.from_numpy(actions[frame].copy()))

        proprio_75 = st[:, :75]
        ee_hist = torch.zeros((4, 4), dtype=torch.float32)
        if self.ee_hist_dim > 0:
            ee_hist[:, :self.ee_hist_dim] = st[:, 75:75 + self.ee_hist_dim]

        act_47 = torch.zeros((10, 47), dtype=torch.float32)
        act_47[:, :45] = act[:, :45]
        if self.ee_dim > 0:
            act_47[:, 45:45 + self.ee_dim] = act[:, 45:45 + self.ee_dim]

        return {
            "video_latents": latents,
            "text_embeddings": text,
            "proprio": proprio_75,
            "ee_hist": ee_hist,
            "actions": act_47,
            "embodiment_id": torch.tensor(self.embodiment_id, dtype=torch.int64),
            "ee_mask": torch.tensor(1.0 if self.ee_dim > 0 else 0.0, dtype=torch.float32),
        }


class MultiEmbodimentP1Dataset(Dataset):
    """Samples across weighted real-world P1 branches (human-protocol, gripper, nymeria-40, hiw500; optional sim).

    In v2.0 Canonical Human mode (auto-enabled when canonical_human_5pt_stats.json is present),
    all active branches share the single global 5-point normalization statistics and support
    motion-bucket balanced sampling (Bucket A locomotion / Bucket B manipulation / Bucket C idle).
    """

    def __init__(
        self,
        p1_processed,
        split="train",
        *,
        weights=(0.25, 0.0, 0.25, 0.25, 0.25),
        bucket_weights=(0.50, 0.40, 0.10),
        global_stats_path=None,
        virtual_length=200000,
        seed=20261121,
    ):
        self.root = Path(p1_processed)
        self.split = split
        self.branches = []
        self.branch_names = []
        active_weights = []

        if global_stats_path is None:
            candidate = self.root / "canonical_human_5pt_stats.json"
            if candidate.is_file():
                global_stats_path = candidate
        self.global_stats_path = Path(global_stats_path) if global_stats_path is not None else None
        human_stats = None
        robot_stats = None
        if self.global_stats_path is not None:
            gstats = json.loads(self.global_stats_path.read_text())
            human_stats = {
                "action": gstats["action_human"],
                "observation.state": gstats["observation.state"],
            }
            robot_stats = {
                "action": gstats["action"],
                "observation.state_5pt_hist": gstats["observation.state_5pt_hist"],
            }

        def _add_branch(name, readers, w):
            if w > 0.0 and readers and sum(len(r) for r in readers) > 0:
                self.branches.append(readers)
                self.branch_names.append(name)
                active_weights.append(float(w))

        w_list = list(weights) + [0.0] * max(0, 5 - len(weights))

        # 0: human-protocol (embodiment_id=2, human_5pt_only)
        if w_list[0] > 0.0:
            proto = json.loads((self.root / "human-protocol/protocol.json").read_text())
            human_sub = []
            for sid, src in enumerate(proto["sources"]):
                ep_ids = [r["episode_index"] for r in proto["episodes"] if r["source_id"] == sid and r["split"] == split]
                if ep_ids:
                    src_root = Path(src["root"])
                    if not src_root.exists():
                        src_root = self.root / src_root.name
                    human_sub.append(_FastBranchReader(
                        HumanWAMDataset(src_root, split, episode_ids=ep_ids, statistics=human_stats or proto["statistics"]),
                        embodiment_id=2, ee_dim=0, ee_hist_dim=0,
                    ))
            _add_branch("human_protocol", human_sub, w_list[0])

        # 1: sim (optional; skipped when weight=0 or directory absent)
        if w_list[1] > 0.0 and (self.root / "sim").is_dir():
            _add_branch("sim", [_FastBranchReader(
                WAMDataset(self.root / "sim", split, dataset_name="q1_stub", statistics=robot_stats),
                embodiment_id=0, ee_dim=2, ee_hist_dim=2,
            )], w_list[1])

        # 2: gripper (OpenHLM G1, embodiment_id=1, bimanual_parallel_gripper)
        if w_list[2] > 0.0 and (self.root / "gripper").is_dir():
            _add_branch("gripper", [_FastBranchReader(
                WAMDataset(self.root / "gripper", split, dataset_name="g1_parallel_gripper", statistics=robot_stats),
                embodiment_id=1, ee_dim=2, ee_hist_dim=4, use_proprio_5pt=True,
            )], w_list[2])

        # 3: nymeria-40 (embodiment_id=2, human_5pt_only; legacy seated 'nymeria' excluded)
        if w_list[3] > 0.0:
            nym_sub = []
            if (self.root / "nymeria-40/wam_lerobot_v2/nymeria_ego").is_dir():
                nym_sub.append(_FastBranchReader(
                    NymeriaWAMDataset(self.root / "nymeria-40", split, statistics=human_stats),
                    embodiment_id=2, ee_dim=0, ee_hist_dim=0,
                ))
            _add_branch("nymeria", nym_sub, w_list[3])

        # 4: hiw500 (BitRobot/HIW-500 G1 open-world, embodiment_id=1, bimanual_parallel_gripper)
        if w_list[4] > 0.0 and (self.root / "hiw500/wam_lerobot_v2/g1_hiw500").is_dir():
            _add_branch("hiw500", [_FastBranchReader(
                WAMDataset(self.root / "hiw500", split, dataset_name="g1_hiw500", statistics=robot_stats),
                embodiment_id=1, ee_dim=2, ee_hist_dim=4, use_proprio_5pt=True,
            )], w_list[4])

        if not self.branches:
            raise ValueError("No active training branches with non-zero weight")
        self.branch_sizes = [sum(len(s) for s in group) for group in self.branches]
        w = np.asarray(active_weights, dtype=np.float64)
        self.probs = w / w.sum()
        self.virtual_length = int(virtual_length)
        self.seed = int(seed)

        # Optional motion-bucket balanced sampling (train split only)
        self.bucket_probs = None
        self.branch_buckets = None
        if split == "train" and bucket_weights is not None and len(bucket_weights) == 3 and sum(bucket_weights) > 0:
            bw = np.asarray(bucket_weights, dtype=np.float64)
            self.bucket_probs = bw / bw.sum()
            self.branch_buckets = [self._index_buckets(group) for group in self.branches]

    @staticmethod
    def _index_buckets(group):
        by_bucket = [[], [], []]
        offset = 0
        for sub in group:
            r = sub.reader
            ep_buckets = {}
            for ep in {s[0] for s in r.samples}:
                pq_path = r.dataset / f"data/chunk-000/episode_{ep:06d}.parquet"
                schema = pq.read_schema(pq_path)
                if "motion_bucket" in schema.names:
                    ep_buckets[ep] = pq.read_table(pq_path, columns=["motion_bucket"])["motion_bucket"].to_numpy()
                else:
                    return None
            for local_idx, (ep, _, frame) in enumerate(r.samples):
                b = int(ep_buckets[ep][frame])
                by_bucket[b if 0 <= b <= 2 else 2].append(offset + local_idx)
            offset += len(sub)
        return [np.asarray(arr, dtype=np.int64) for arr in by_bucket]

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index * 1000003)
        b_idx = int(rng.choice(len(self.branches), p=self.probs))
        group = self.branches[b_idx]
        buckets = self.branch_buckets[b_idx] if self.branch_buckets is not None else None
        if buckets is not None and self.bucket_probs is not None:
            non_empty = [i for i in range(3) if len(buckets[i]) > 0]
            p = self.bucket_probs[non_empty]
            p = p / p.sum()
            chosen_b = int(rng.choice(non_empty, p=p))
            pool = buckets[chosen_b]
            pos = int(pool[rng.integers(0, len(pool))])
        else:
            total = self.branch_sizes[b_idx]
            pos = int(rng.integers(0, total))
        for sub in group:
            if pos < len(sub):
                return sub.sample(pos)
            pos -= len(sub)
        return group[-1].sample(0)

    def sample_branch_batch(self, branch_idx, n_samples, seed=0):
        group = self.branches[branch_idx]
        total = self.branch_sizes[branch_idx]
        indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
        items = []
        for pos in indices:
            p = int(pos)
            for sub in group:
                if p < len(sub):
                    items.append(sub.sample(p))
                    break
                p -= len(sub)
        return items


def save_checkpoint(run, step, model, optimizer, scheduler, keep, *, samples_seen):
    directory = run / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{step:08d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "expert": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "samples_seen": samples_seen,
    }, temporary)
    temporary.replace(path)
    checkpoints = sorted(directory.glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-")))
    for expired in checkpoints[:-keep]:
        expired.unlink()
    return path.relative_to(run).as_posix()


@torch.inference_mode()
def evaluate_validation(model, val_ds, device, n_per_branch=32):
    model.eval()
    results = {}
    for b_idx, name in enumerate(val_ds.branch_names):
        items = val_ds.sample_branch_batch(b_idx, n_per_branch)
        losses = []
        for start in range(0, len(items), 16):
            chunk = items[start:start + 16]
            video = torch.stack([s["video_latents"] for s in chunk]).to(device)
            text = torch.empty((len(chunk), *chunk[0]["text_embeddings"].shape), dtype=chunk[0]["text_embeddings"].dtype, device=device)
            for ri, s in enumerate(chunk):
                text[ri].copy_(s["text_embeddings"])
            proprio = torch.stack([s["proprio"] for s in chunk]).to(device)
            ee_hist = torch.stack([s["ee_hist"] for s in chunk]).to(device)
            target = torch.stack([s["actions"] for s in chunk]).to(device)
            emb_id = torch.stack([s["embodiment_id"] for s in chunk]).to(device)
            ee_mask = torch.stack([s["ee_mask"] for s in chunk]).to(device)
            gen = torch.Generator(device=device).manual_seed(1234 + start)
            time_val = torch.full((len(target),), 0.5, device=device)
            noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=gen)
            mixed = torch.lerp(noise, target, time_val[:, None, None])
            with autocast(device):
                ctx = model.encode_history(video, text, proprio=proprio, ee_hist=ee_hist, embodiment_id=emb_id)
                vel = model.action_velocity(ctx, mixed, time_val, embodiment_id=emb_id)
                loss, _ = action_loss(vel, target - noise, ee_mask=ee_mask)
            losses.append(float(loss.item()))
            del video, text, proprio, ee_hist, target, emb_id, ee_mask
        results[name] = float(np.mean(losses))
    model.train()
    model.backbone.eval()
    return results


def train(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA device to this run (or launch via torchrun for multi-GPU DDP)")
        device = torch.device("cuda:0")

    run = args.run_root
    if is_main:
        run.mkdir(parents=True, exist_ok=True)
        if args.resume is None and (run / "run.json").exists():
            raise FileExistsError(f"Run already exists at {run}")
    if is_distributed:
        dist.barrier()

    torch.manual_seed(args.seed + rank)
    torch.cuda.reset_peak_memory_stats(device)

    effective_batch = args.micro_batch * args.accumulation * world_size
    total_samples = args.steps * effective_batch
    train_ds = MultiEmbodimentP1Dataset(
        args.p1_processed_root, "train",
        weights=tuple(args.branch_weights),
        bucket_weights=tuple(args.bucket_weights),
        global_stats_path=args.global_stats,
        virtual_length=total_samples + effective_batch * 4,
        seed=args.seed,
    )
    val_ds = MultiEmbodimentP1Dataset(
        args.p1_processed_root, "val",
        weights=tuple(args.branch_weights),
        bucket_weights=None,
        global_stats_path=args.global_stats,
        virtual_length=256,
        seed=args.seed + 1,
    ) if is_main else None

    backbone = load_backbone(args.backbone, args.backbone_manifest, device)
    raw_model = CosmosWAM(
        backbone,
        width=args.width,
        layers=args.layers,
        unfreeze_blocks=args.unfreeze_dit_blocks,
        use_proprio=True,
        num_embodiments=3,
    ).to(device)
    raw_model.backbone.eval()

    dit_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    expert_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    parameters = expert_params + dit_params
    param_groups = [{"params": expert_params, "lr": args.learning_rate}]
    if dit_params:
        param_groups.append({"params": dit_params, "lr": args.dit_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    samples_seen = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = int(ckpt["step"])
        raw_model.load_state_dict(ckpt["expert"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if not is_distributed and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        samples_seen = int(ckpt.get("samples_seen", start_step * effective_batch))
        if is_main:
            print(f"Resumed P1 WAM from {args.resume} at step={start_step} (samples_seen={samples_seen})", flush=True)
    elif args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        raw_model.load_state_dict(ckpt["expert"], strict=False)
        if is_main:
            print(f"Warm-started P1 WAM weights from {args.init_from} (fresh optimizer & {args.steps}-step LR schedule)", flush=True)

    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True) if is_distributed else raw_model

    if is_main:
        config = {
            "schema": "wam-p1-multimodal-v1",
            "p1_processed_root": str(args.p1_processed_root),
            "backbone": str(args.backbone),
            "backbone_manifest": str(args.backbone_manifest),
            "backbone_manifest_sha256": sha256(args.backbone_manifest),
            "init_from": str(args.init_from) if args.init_from is not None else None,
            "steps": args.steps,
            "checkpoint_every": args.checkpoint_every,
            "keep_checkpoints": args.keep_checkpoints,
            "world_size": world_size,
            "micro_batch": args.micro_batch,
            "accumulation": args.accumulation,
            "effective_batch_size": effective_batch,
            "learning_rate": args.learning_rate,
            "dit_lr": args.dit_lr,
            "unfreeze_dit_blocks": args.unfreeze_dit_blocks,
            "warmup_steps": args.warmup_steps,
            "width": args.width,
            "layers": args.layers,
            "use_proprio": True,
            "num_embodiments": 3,
            "embodiment_map": {"bimanual_stub": 0, "bimanual_parallel_gripper": 1, "human_5pt_only": 2},
            "branch_names": train_ds.branch_names,
            "branch_sizes_train": train_ds.branch_sizes,
            "branch_sizes_val": val_ds.branch_sizes,
            "branch_weights": list(args.branch_weights),
            "bucket_weights": list(args.bucket_weights),
            "seed": args.seed,
            "backbone_frozen": args.unfreeze_dit_blocks == 0,
            "device": f"{world_size}x {torch.cuda.get_device_name(device)}",
            "trainable_parameters": sum(p.numel() for p in parameters),
            "conditioning_latents": 2,
        }
        write_json(run / "run.json", config)

    # Pre-allocate GPU text buffer once to avoid 3.1 GiB CPU stack/shm churn every step.
    text_buf = torch.empty((args.micro_batch, 512, 100352), dtype=torch.bfloat16, device=device)

    # Multi-threaded async prefetcher overlaps parallel safetensors/Arrow reads with GPU compute (zero /dev/shm).
    total_microsteps = (args.steps - start_step) * args.accumulation
    prefetch_q = Queue(maxsize=6)
    stop_event = threading.Event()
    prefetch_threads = max(1, int(args.prefetch_threads))

    def _prefetch_worker():
        cursor = samples_seen
        try:
            with ThreadPoolExecutor(max_workers=prefetch_threads) as pool:
                for _ in range(total_microsteps):
                    if stop_event.is_set():
                        break
                    base = cursor + rank * args.micro_batch
                    if prefetch_threads > 1:
                        items = list(pool.map(train_ds.__getitem__, range(base, base + args.micro_batch)))
                    else:
                        items = [train_ds[base + i] for i in range(args.micro_batch)]
                    cursor += args.micro_batch * world_size
                    packed = (
                        torch.stack([s["video_latents"] for s in items]),
                        [s["text_embeddings"] for s in items],
                        torch.stack([s["proprio"] for s in items]),
                        torch.stack([s["ee_hist"] for s in items]),
                        torch.stack([s["actions"] for s in items]),
                        torch.stack([s["embodiment_id"] for s in items]),
                        torch.stack([s["ee_mask"] for s in items]),
                    )
                    prefetch_q.put(packed)
        except Exception as err:
            prefetch_q.put(err)

    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    initial = {k: v.clone() for k, v in trainable_state(raw_model).items()} if is_main else None
    started = perf_counter()
    checkpoint = None
    log_mode = "a" if args.resume is not None else "w"

    log = (run / "training.jsonl").open(log_mode, buffering=1) if is_main else None
    writer = SummaryWriter(run / "tensorboard", flush_secs=5) if is_main else None
    try:
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(5, device=device)
            for micro_i in range(args.accumulation):
                packed = prefetch_q.get()
                if isinstance(packed, Exception):
                    raise packed
                v_cpu, texts, p_cpu, e_cpu, a_cpu, emb_cpu, m_cpu = packed
                video = v_cpu.to(device, non_blocking=True)
                for row_i, t_tensor in enumerate(texts):
                    text_buf[row_i].copy_(t_tensor, non_blocking=True)
                proprio = p_cpu.to(device, non_blocking=True)
                ee_hist = e_cpu.to(device, non_blocking=True)
                target = a_cpu.to(device, non_blocking=True)
                emb_id = emb_cpu.to(device, non_blocking=True)
                ee_mask = m_cpu.to(device, non_blocking=True)

                time_val = flow_time(len(target), device)
                noise = torch.randn_like(target)
                mixed = torch.lerp(noise, target, time_val[:, None, None])
                sync_ctx = model.no_sync() if (is_distributed and micro_i < args.accumulation - 1) else nullcontext()
                with sync_ctx:
                    with autocast(device):
                        velocity = model(video, text_buf, mixed, time_val, proprio=proprio, ee_hist=ee_hist, embodiment_id=emb_id)
                        loss, groups = action_loss(velocity, target - noise, ee_mask=ee_mask)
                    (loss / args.accumulation).backward()
                totals += torch.cat((loss.detach().view(1), groups.detach()))
                del packed, v_cpu, texts, p_cpu, e_cpu, a_cpu, emb_cpu, m_cpu, video, proprio, ee_hist, target, emb_id, ee_mask

            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            samples_seen += effective_batch

            if is_distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.AVG)

            if is_main:
                vals = (totals / args.accumulation).tolist()
                metrics = {
                    "flow_loss": vals[0],
                    "lower_position_flow_loss": vals[1],
                    "wrist_flow_loss": vals[2],
                    "rotation_flow_loss": vals[3],
                    "ee_flow_loss": vals[4],
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": norm.item(),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                for name, value in metrics.items():
                    writer.add_scalar("train/" + name, value, step)
                row = {"optimizer_step": step, "samples_seen": samples_seen, **metrics}

                if step % args.checkpoint_every == 0 or step == args.steps:
                    val_metrics = evaluate_validation(raw_model, val_ds, device, n_per_branch=32)
                    for k, v in val_metrics.items():
                        writer.add_scalar("val/flow_loss_" + k, v, step)
                    row["val_flow_loss"] = val_metrics
                    checkpoint = save_checkpoint(run, step, raw_model, optimizer, scheduler, args.keep_checkpoints, samples_seen=samples_seen)
                    print(f"step={step}/{args.steps} train_loss={vals[0]:.5f} val={val_metrics} ckpt={checkpoint}", flush=True)
                elif step == start_step + 1 or step % 50 == 0:
                    elapsed = perf_counter() - started
                    sps = (step - start_step) / max(elapsed, 1e-3)
                    print(f"step={step}/{args.steps} flow_loss={vals[0]:.5f} lower={vals[1]:.4f} wrist={vals[2]:.4f} rot={vals[3]:.4f} ee={vals[4]:.4f} lr={metrics['lr']:.2e} ({sps:.2f} steps/s)", flush=True)

                log.write(json.dumps(row, allow_nan=False) + "\n")

            if is_distributed and (step % args.checkpoint_every == 0 or step == args.steps):
                dist.barrier()
    finally:
        stop_event.set()
        worker.join(timeout=5)
        if writer is not None:
            writer.close()
        if log is not None:
            log.close()

    if is_main:
        changed = any(not torch.equal(v, initial[k]) for k, v in trainable_state(raw_model).items())
        frozen_backbone_grad = any(p.grad is not None for p in raw_model.backbone.parameters() if not p.requires_grad)
        if not changed or frozen_backbone_grad:
            raise RuntimeError("Expert update or frozen backbone invariant failed")
        write_json(run / "training_summary.json", {
            "completed": True,
            "optimizer_updates": args.steps,
            "samples_seen": samples_seen,
            "effective_batch_size": effective_batch,
            "world_size": world_size,
            "checkpoint": checkpoint,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
            "elapsed_seconds": perf_counter() - started,
            "expert_weights_changed": changed,
        })
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-processed-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dit-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-dit-blocks", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--branch-weights", type=float, nargs="+", default=(0.25, 0.0, 0.25, 0.25, 0.25))
    parser.add_argument("--bucket-weights", type=float, nargs=3, default=(0.50, 0.40, 0.10))
    parser.add_argument("--global-stats", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
