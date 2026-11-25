"""Offline evaluation of a trained P1/Real CosmosWAM checkpoint across all held-out test splits & human-operations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from scripts.train_wam_p1 import _FastBranchReader
from wammimic.data_pipeline.common import write_json
from wammimic.data_pipeline.datasets import HumanWAMDataset, NymeriaWAMDataset, WAMDataset
from wammimic.training.evaluation import autocast
from wammimic.training.model import CosmosWAM, load_backbone


def _rot6d_to_mat(r6: np.ndarray) -> tuple[np.ndarray, float]:
    # r6: [..., 6] -> [..., 3, 3]
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(n1, 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    u2 = a2 - proj
    n2 = np.linalg.norm(u2, axis=-1, keepdims=True)
    degen = float(((n1[..., 0] < 1e-4) | (n2[..., 0] < 1e-4)).mean())
    b2 = u2 / np.maximum(n2, 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1), degen


@torch.inference_mode()
def eval_reader(model, readers, name, device, *, leg_length=0.85, n_samples=128, batch_size=16, sample_steps=16):
    total = sum(len(r) for r in readers)
    indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
    items = []
    for pos in indices:
        p = int(pos)
        for r in readers:
            if p < len(r):
                items.append((r, r.sample(p)))
                break
            p -= len(r)

    pos_errs, pos_02_errs, rot_errs, ee_maes, degen_fracs = [], [], [], [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        r0 = chunk[0][0]
        act_stats = r0.reader.statistics["action"]
        a_mean = np.asarray(act_stats["mean"], dtype=np.float32)
        a_std = np.maximum(np.asarray(act_stats["std"], dtype=np.float32), 1e-6)

        video = torch.stack([s["video_latents"] for _, s in chunk]).to(device)
        text = torch.empty((len(chunk), *chunk[0][1]["text_embeddings"].shape), dtype=chunk[0][1]["text_embeddings"].dtype, device=device)
        for i, (_, s) in enumerate(chunk):
            text[i].copy_(s["text_embeddings"])
        proprio = torch.stack([s["proprio"] for _, s in chunk]).to(device)
        ee_hist = torch.stack([s["ee_hist"] for _, s in chunk]).to(device)
        emb_id = torch.stack([s["embodiment_id"] for _, s in chunk]).to(device)
        gt_norm = np.stack([s["actions"].numpy() for _, s in chunk])

        gen = torch.Generator(device=device).manual_seed(2026 + start)
        noise = torch.randn((len(chunk), 10, 47), device=device, generator=gen)
        with autocast(device):
            ctx = model.encode_history(video, text, proprio=proprio, ee_hist=ee_hist, embodiment_id=emb_id)
            pred_norm = model.sample(ctx, noise, steps=sample_steps, embodiment_id=emb_id).cpu().numpy()

        d_act = len(a_mean)
        pred = pred_norm[..., :d_act] * a_std + a_mean
        gt = gt_norm[..., :d_act] * a_std + a_mean

        # Unscale lower body (first 3 points) by leg_length to meters; wrists (last 2) are already meters
        p_pred = pred[..., :15].reshape(-1, 10, 5, 3).copy()
        p_gt = gt[..., :15].reshape(-1, 10, 5, 3).copy()
        p_pred[:, :, :3] *= leg_length
        p_gt[:, :, :3] *= leg_length

        dist_mm = np.linalg.norm(p_pred - p_gt, axis=-1) * 1000.0  # [B, 10, 5]
        pos_errs.append(dist_mm.mean())
        pos_02_errs.append(dist_mm[:, 1, :].mean())  # t0 + 0.2s (index 1)

        R_pred, deg = _rot6d_to_mat(pred[..., 15:45].reshape(-1, 10, 5, 6))
        R_gt, _ = _rot6d_to_mat(gt[..., 15:45].reshape(-1, 10, 5, 6))
        degen_fracs.append(deg)
        delta = np.einsum("...ji,...jk->...ik", R_pred, R_gt).reshape(-1, 3, 3)
        ang_deg = np.rad2deg(Rotation.from_matrix(delta).magnitude())
        rot_errs.append(float(ang_deg.mean()))

        if d_act == 47:
            ee_maes.append(float(np.abs(pred[..., 45:47] - gt[..., 45:47]).mean()))

        del video, text, proprio, ee_hist, emb_id

    out = {
        "split": name,
        "evaluated_windows": len(items),
        "total_split_windows": total,
        "ade_mm": round(float(np.mean(pos_errs)), 2),
        "err_02s_mm": round(float(np.mean(pos_02_errs)), 2),
        "rot_geodesic_deg": round(float(np.mean(rot_errs)), 2),
        "degenerate_rot_pct": round(float(np.mean(degen_fracs)) * 100.0, 4),
        "ee_mae": round(float(np.mean(ee_maes)), 4) if ee_maes else None,
    }
    print(json.dumps(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-processed-root", type=Path, required=True)
    parser.add_argument("--wam-run", type=Path, required=True)
    parser.add_argument("--wam-checkpoint", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = json.loads((args.wam_run / "run.json").read_text())
    saved = torch.load(args.wam_checkpoint, map_location="cpu", weights_only=True)
    dit_block_ids = {int(k.split(".")[2]) for k in saved["expert"] if k.startswith("backbone.transformer_blocks.")}
    backbone = load_backbone(config["backbone"], config["backbone_manifest"], device)
    model = CosmosWAM(
        backbone,
        width=config["width"],
        layers=config["layers"],
        unfreeze_blocks=len(dit_block_ids),
        use_proprio=True,
        num_embodiments=config["num_embodiments"],
    ).to(device).eval()
    model.load_state_dict(saved["expert"], strict=False)

    root = args.p1_processed_root
    results = []

    # 1. human-protocol test
    proto = json.loads((root / "human-protocol/protocol.json").read_text())
    h_readers = []
    for sid, src in enumerate(proto["sources"]):
        ep_ids = [r["episode_index"] for r in proto["episodes"] if r["source_id"] == sid and r["split"] == "test"]
        if ep_ids:
            src_root = Path(src["root"])
            if not src_root.exists():
                src_root = root / src_root.name
            h_readers.append(_FastBranchReader(
                HumanWAMDataset(src_root, "test", episode_ids=ep_ids, statistics=proto["statistics"]),
                embodiment_id=2, ee_dim=0, ee_hist_dim=0,
            ))
    results.append(eval_reader(model, h_readers, "human_protocol_test", device, leg_length=0.88, n_samples=args.n_samples))

    # 2. human-operations fixed regression holdout
    if (root / "human-operations/wam_lerobot_v2/human_ego").is_dir():
        op_eps = [r["episode_index"] for r in map(json.loads, (root / "human-operations/wam_lerobot_v2/human_ego/meta/episodes.jsonl").read_text().splitlines())]
        op_reader = [_FastBranchReader(
            HumanWAMDataset(root / "human-operations", "test", episode_ids=op_eps, statistics=proto["statistics"]),
            embodiment_id=2, ee_dim=0, ee_hist_dim=0,
        )]
        results.append(eval_reader(model, op_reader, "human_operations_holdout", device, leg_length=0.88, n_samples=args.n_samples))

    # 3. nymeria-40 test (3 held-out participants)
    if (root / "nymeria-40/wam_lerobot_v2/nymeria_ego").is_dir():
        n_reader = [_FastBranchReader(NymeriaWAMDataset(root / "nymeria-40", "test"), embodiment_id=2, ee_dim=0, ee_hist_dim=0)]
        results.append(eval_reader(model, n_reader, "nymeria40_test", device, leg_length=0.88, n_samples=args.n_samples))

    # 4. gripper (OpenHLM G1) test
    if (root / "gripper/wam_lerobot_v2/g1_parallel_gripper").is_dir():
        g_reader = [_FastBranchReader(WAMDataset(root / "gripper", "test", dataset_name="g1_parallel_gripper"), embodiment_id=1, ee_dim=2, ee_hist_dim=4, use_proprio_5pt=True)]
        results.append(eval_reader(model, g_reader, "gripper_openhlm_test", device, leg_length=0.68, n_samples=args.n_samples))

    # 5. hiw500 test
    if (root / "hiw500/wam_lerobot_v2/g1_hiw500").is_dir():
        hw_reader = [_FastBranchReader(WAMDataset(root / "hiw500", "test", dataset_name="g1_hiw500"), embodiment_id=1, ee_dim=2, ee_hist_dim=4, use_proprio_5pt=True)]
        results.append(eval_reader(model, hw_reader, "hiw500_test", device, leg_length=0.68, n_samples=args.n_samples))

    write_json(args.out_json, {"checkpoint": str(args.wam_checkpoint), "step": int(saved["step"]), "unfrozen_dit_blocks": len(dit_block_ids), "splits": results})


if __name__ == "__main__":
    main()
