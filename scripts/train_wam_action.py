"""Train the human five-point Action Expert with a frozen Cosmos backbone."""

import argparse
from itertools import islice
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wammimic.data_pipeline.common import sha256, write_json
from wammimic.training.data import PilotDataset
from wammimic.training.evaluation import autocast
from wammimic.training.model import CosmosWAM, action_loss, flow_time, load_backbone
from wammimic.training.run import trainable_state


def save_checkpoint(run, step, model, optimizer, keep, *, samples_seen=None, microsteps_seen=None):
    directory = run / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"step-{step:08d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step, "expert": trainable_state(model), "optimizer": optimizer.state_dict(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "samples_seen": samples_seen, "microsteps_seen": microsteps_seen,
    }, temporary)
    temporary.replace(path)
    checkpoints = sorted(directory.glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-")))
    for expired in checkpoints[:-keep]:
        expired.unlink()
    return path.relative_to(run).as_posix()


def shuffled_batches(dataset, seed, skip=0):
    # Recreate the original sampler without loading samples already seen by the optimizer.
    loader = DataLoader(range(len(dataset)), batch_size=None, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    while True:
        for index in loader:
            if skip:
                skip -= 1
                continue
            yield dataset[index]


def training_tensors(batches, size, device):
    samples = list(islice(batches, size))
    video = torch.stack([sample["video_latents"] for sample in samples]).to(device)
    target = torch.stack([sample["actions"] for sample in samples]).to(device)
    first_prompt = samples[0]["text_embeddings"]
    # Fill the device batch directly because each prompt is 98 MiB.
    text = torch.empty((size, *first_prompt.shape), dtype=first_prompt.dtype, device=device)
    for row, sample in zip(text, samples):
        row.copy_(sample["text_embeddings"])
    return video, text, target


def resume_training(args, model, optimizer, identity):
    if args.resume is None:
        return 0, 0, 0
    parent = json.loads((args.resume.parent.parent / "run.json").read_text())
    for key, value in identity.items():
        if parent.get(key) != value:
            raise ValueError(f"Checkpoint run uses a different {key}")
    state = torch.load(args.resume, map_location="cpu", weights_only=True)
    step = state["step"]
    if type(step) is not int or not 0 < step < args.steps:
        raise ValueError("Resume checkpoint must precede the requested final optimizer step")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if state["expert"].keys() != expected:
        raise ValueError("Checkpoint does not match this Action Expert")
    model.load_state_dict(state["expert"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    if state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"])
    else:
        # The 500-step pilot stored AdamW but did not store its CUDA RNG.
        torch.cuda.manual_seed(args.seed + step)
    microsteps_seen = state.get("microsteps_seen")
    if microsteps_seen is None:
        microsteps_seen = step * parent["accumulation"]
    samples_seen = state.get("samples_seen")
    if samples_seen is None:
        samples_seen = microsteps_seen * parent["micro_batch"]
    print(f"Resumed Action Expert and AdamW at optimizer step {step} after {samples_seen} windows", flush=True)
    return step, samples_seen, microsteps_seen


def train(args):
    run = args.run_root
    run.mkdir(parents=True, exist_ok=True)
    if (run / "run.json").exists():
        raise FileExistsError(f"Run already exists at {run}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device to this run")
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    dataset = PilotDataset(args.protocol, "train", per_episode=None, min_frame_gap=1)
    if len(dataset) != dataset.protocol["counts"]["train"]["windows"]:
        raise ValueError("Training cache count differs from the protocol")
    model = CosmosWAM(load_backbone(args.backbone, args.backbone_manifest, device),
                      train_backbone=False).to(device)
    model.backbone.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    identity = {
        "protocol_sha256": sha256(args.protocol), "backbone_manifest_sha256": sha256(args.backbone_manifest),
        "seed": args.seed, "learning_rate": args.learning_rate, "backbone_frozen": True,
    }
    start_step, start_samples, start_microsteps = resume_training(args, model, optimizer, identity)
    effective_batch = args.micro_batch * args.accumulation
    initial = {key: value.clone() for key, value in trainable_state(model).items()}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()} | identity | {
        "resume_step": start_step, "resume_samples_seen": start_samples,
        "effective_batch_size": effective_batch,
        "device": torch.cuda.get_device_name(device), "train_samples": len(dataset),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "conditioning_latents": 2,
    }
    batches = shuffled_batches(dataset, args.seed, start_samples)
    started = perf_counter()
    checkpoint = None
    with (run / "training.jsonl").open("x", buffering=1) as log, SummaryWriter(run / "tensorboard", flush_secs=5) as writer:
        write_json(run / "run.json", config)
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(4, device=device)
            for _ in range(args.accumulation):
                video, text, target = training_tensors(batches, args.micro_batch, device)
                time_value = flow_time(len(target), device)
                noise = torch.randn_like(target)
                mixed = torch.lerp(noise, target, time_value[:, None, None])
                with autocast(device):
                    context = model.encode_history(video, text)
                    velocity = model.action_velocity(context, mixed, time_value)
                    loss, groups = action_loss(velocity, target - noise)
                (loss / args.accumulation).backward()
                totals += torch.cat((loss.detach().view(1), groups.detach()))
                del video, text, target
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            values = (totals / args.accumulation).tolist()
            metrics = dict(zip(("flow_loss", "lower_position_flow_loss", "wrist_flow_loss",
                                "rotation_flow_loss"), values))
            metrics["grad_norm"] = norm.item()
            metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            for name, value in metrics.items():
                writer.add_scalar("train/" + name, value, step)
            updates = step - start_step
            samples_seen = start_samples + updates * effective_batch
            microsteps_seen = start_microsteps + updates * args.accumulation
            log.write(json.dumps({"optimizer_step": step, "samples_seen": samples_seen, **metrics}, allow_nan=False) + "\n")
            if step == start_step + 1 or step % 10 == 0 or step == args.steps:
                print(f"step={step}/{args.steps} flow_loss={values[0]:.5f}", flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint = save_checkpoint(run, step, model, optimizer, args.keep_checkpoints,
                                             samples_seen=samples_seen, microsteps_seen=microsteps_seen)
                print(f"checkpoint={checkpoint}", flush=True)
    changed = any(not torch.equal(value, initial[key]) for key, value in trainable_state(model).items())
    if not changed or any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("Expert update or frozen backbone invariant failed")
    write_json(run / "training_summary.json", {
        "completed": True, "optimizer_updates": args.steps, "microsteps": microsteps_seen,
        "samples_seen": samples_seen, "effective_batch_size": effective_batch,
        "resume_step": start_step,
        "checkpoint": checkpoint, "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": perf_counter() - started, "expert_weights_changed": changed,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "backbone", "backbone-manifest", "run-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    if min(args.steps, args.checkpoint_every, args.keep_checkpoints, args.micro_batch, args.accumulation) < 1:
        parser.error("Step, checkpoint and batch counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
