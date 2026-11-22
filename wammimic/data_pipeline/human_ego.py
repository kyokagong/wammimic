"""Human ego supervision from measured optical skeletons, never robot FK or SMPL."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .common import sha256, write_json
from .geometry import matrices

SCHEMA = "human-ego-optical-five-point-v1"
PROVENANCE = "optical_mocap_solved_skeleton"
DATASET = "wam_lerobot_v2/human_ego"
POINT_NAMES = ["Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand"]
SKELETON_SOURCE = "Nokov solved skeleton (vendor export, BVH/CSV Body0)"
QUALITY_POLICY = {"maximum_point_speed_m_s":[10,15,15,20,20],
    "maximum_angular_speed_rad_s":60,"jump_exclusion_margin_s":.1,
    "required_reference":"every 120 Hz source frame in the complete video window"}
HISTORY = {"shape": [4, 75], "offsets_s": [-.3, -.2, -.1, 0.],
    "layout": "45D poses, 15D linear velocity, 15D angular velocity",
    "frame": "t0 full pelvis 3D frame; lower-body positions/linear velocities scaled by leg length",
    "velocity": "backward difference of actual 120 Hz optical measurements; angular velocity in rad/s"}


def optical_quality(times,poses,source_valid):
    """Exclude gross tracking jumps, without changing any measured pose."""
    dt=np.diff(times)
    speed=np.linalg.norm(np.diff(poses[...,:3],axis=0),axis=-1)/dt[:,None]
    q=poses[...,3:];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    angular=2*np.arccos(np.clip(abs((q[1:]*q[:-1]).sum(-1)),0,1))/dt[:,None]
    jumps=(source_valid[1:] & source_valid[:-1]) & (
        (speed>QUALITY_POLICY["maximum_point_speed_m_s"]).any(-1) |
        (angular>QUALITY_POLICY["maximum_angular_speed_rad_s"]).any(-1))
    quality=source_valid.copy()
    margin=QUALITY_POLICY["jump_exclusion_margin_s"]
    for i in np.flatnonzero(jumps):
        quality[(times>=times[i]-margin) & (times<=times[i+1]+margin)]=False
    return quality,int(jumps.sum())


def measured_windows(valid,times,segments,nearest,reference_valid):
    accepted=valid_windows(valid,times,segments);frames=np.flatnonzero(accepted)
    missing=np.r_[0,np.cumsum(~reference_valid)]
    accepted[frames]=missing[nearest[frames+12]+1]==missing[nearest[frames-3]]
    return accepted


def human_features(future, anchor, leg_length):
    """45D in the current human pelvis 3D frame; no assumed floor or robot scale."""
    if not .3 < leg_length < 1.5:
        raise ValueError("Invalid measured human leg length")
    if not np.isfinite(future).all() or not np.isfinite(anchor).all():
        raise ValueError("Unmeasured pose cannot be a human action target")
    reference = matrices(anchor[...,0,3:])
    positions = np.einsum("...tpj,...jk->...tpk", future[..., :3]-anchor[...,0,:3][...,None,None,:], reference)
    positions[..., :3, :] /= leg_length
    rotations = np.einsum("...ij,...tpjk->...tpik", reference.swapaxes(-1,-2), matrices(future[..., 3:]))
    rot6d = rotations[..., :, :2].swapaxes(-1, -2).reshape(future.shape[:-1] + (6,))
    shape=future.shape[:-2]
    return np.concatenate([positions.reshape(shape+(15,)), rot6d.reshape(shape+(30,))], axis=-1).astype(np.float32)


def human_history(poses, times, indices, leg_length):
    """Four actual poses; each velocity uses only its preceding measured sample."""
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 4 or (indices <= 0).any():
        raise ValueError("History requires four measurements with measured predecessors")
    if (indices >= len(poses)).any() or (np.diff(indices, axis=1) <= 0).any():
        raise ValueError("History indices must be increasing and inside the measured sequence")
    current, previous = poses[indices], poses[indices - 1]
    dt = times[indices] - times[indices - 1]
    if not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError("History requires increasing measured timestamps")
    anchor = poses[indices[:, -1]]
    reference = matrices(anchor[:, 0, 3:])
    linear = np.einsum("btpj,bjk->btpk", (current[..., :3] - previous[..., :3]) / dt[..., None, None], reference)
    linear[..., :3, :] /= leg_length
    change = matrices(current[..., 3:]) @ matrices(previous[..., 3:]).swapaxes(-1, -2)
    angular = Rotation.from_matrix(change.reshape(-1, 3, 3)).as_rotvec().reshape(*indices.shape, 5, 3)
    angular = np.einsum("btpj,bjk->btpk", angular / dt[..., None, None], reference)
    result = np.concatenate((human_features(current, anchor, leg_length),
                             linear.reshape(*indices.shape, 15), angular.reshape(*indices.shape, 15)), axis=-1)
    if not np.isfinite(result).all():raise ValueError("Nonfinite measured history")
    return result.astype(np.float32)


def nearest_measurements(mocap_times, video_times, source_valid, *, causal=False):
    """Select an actual measured frame; never interpolate a missing skeleton."""
    if (np.diff(mocap_times) <= 0).any() or (np.diff(video_times) <= 0).any():
        raise ValueError("Nonmonotone source timestamps")
    upper = np.searchsorted(mocap_times, video_times).clip(0, len(mocap_times)-1)
    lower = (upper-1).clip(0)
    index = np.where(abs(mocap_times[lower]-video_times) <= abs(mocap_times[upper]-video_times), lower, upper)
    if causal:index = np.where(mocap_times[upper] <= video_times, upper, lower)
    error = mocap_times[index]-video_times
    valid = source_valid[index] & (video_times >= mocap_times[0]) & (video_times <= mocap_times[-1]) & (abs(error) <= (1/120 if causal else 1/240)+1e-5)
    return index, error, valid


def valid_windows(valid, times, segment):
    result = np.zeros(len(valid), dtype=bool)
    for i in range(3, len(valid)-12):
        span = slice(i-3, i+13)
        result[i] = (valid[span].all() and segment[i] >= 0 and (segment[span] == segment[i]).all()
                     and np.allclose(np.diff(times[span]), .1, atol=.005, rtol=0))
    return result


def _bvh_reference(path, indices):
    """Read vendor BVH for an independent check of the published NPZ skeleton."""
    header, motion = path.read_text().split("MOTION", 1)
    tokens = iter(re.findall(r"[{}]|[^\s{}]+", header))
    names, parents, offsets, channels = [], [], [], []

    def joint(parent):
        name = next(tokens); assert next(tokens) == "{"
        index = len(names); names.append(name); parents.append(parent)
        assert next(tokens) == "OFFSET"
        offsets.append([float(next(tokens)) for _ in range(3)])
        assert next(tokens) == "CHANNELS"
        channels.append([next(tokens) for _ in range(int(next(tokens)))])
        for token in tokens:
            if token == "}": return
            if token == "JOINT": joint(index)
            elif token == "End":
                assert next(tokens) == "Site" and next(tokens) == "{" and next(tokens) == "OFFSET"
                for _ in range(3): float(next(tokens))
                assert next(tokens) == "}"
            else: raise ValueError("Unexpected BVH hierarchy token")

    assert next(tokens) == "HIERARCHY" and next(tokens) == "ROOT"
    joint(-1)
    lines = motion.strip().splitlines()
    count = int(lines[0].split(":")[1]); period = float(lines[1].split(":")[1])
    if len(lines)-2 != count or not np.isclose(period, 1/120, atol=1e-7):
        raise ValueError("Invalid vendor BVH timebase")
    if min(indices) < 0 or max(indices) >= count: raise ValueError("BVH frame index outside source")
    values = np.stack([np.fromstring(lines[2+int(i)], sep=" ") for i in indices])
    if values.shape[1] != sum(map(len, channels)): raise ValueError("Invalid vendor BVH channels")
    positions = np.zeros((len(indices), len(names), 3)); rotations = np.zeros((len(indices), len(names), 3, 3)); cursor = 0
    for j, (parent, axes) in enumerate(zip(parents, channels)):
        block = values[:, cursor:cursor+len(axes)]; cursor += len(axes)
        rotation_axes = [a[0] for a in axes if a.endswith("rotation")]
        local = Rotation.from_euler("".join(rotation_axes), block[:, [k for k,a in enumerate(axes) if a.endswith("rotation")]], degrees=True).as_matrix()
        if parent < 0:
            positions[:, j] = block[:, [axes.index(a+"position") for a in "XYZ"]]
            rotations[:, j] = local
        else:
            positions[:, j] = positions[:, parent] + np.einsum("nij,j->ni", rotations[:, parent], offsets[j])
            rotations[:, j] = rotations[:, parent] @ local
    return names, positions, rotations


def load_optical_reference(folder, definition):
    if definition.get("source") != SKELETON_SOURCE:
        raise ValueError("Only vendor optical skeleton references are accepted")
    meta = json.loads((folder/"meta.json").read_text())
    with np.load(folder/"motions/skeleton.npz", allow_pickle=False) as data:
        names = data["joint_names"].tolist(); t = data["t"].copy(); source_index = data["frame_index"].copy()
        position = data["positions_m"].copy(); quaternion = data["quaternions_xyzw"].copy()
        valid = data["valid"].copy(); offsets = data["offsets_m"].copy(); canonical = data["R_canon_nokov"].copy()
        if "Z-up" not in str(data["frame"]) or float(data["mocap_rate_hz"]) != 120:
            raise ValueError("Unsupported optical coordinate/time convention")
    if len(names) != len(set(names)) or valid.dtype != np.bool_ or valid.shape != t.shape:
        raise ValueError("Invalid optical skeleton schema")
    if position.shape != (len(t),len(names),3) or quaternion.shape != (len(t),len(names),4):
        raise ValueError("Incomplete optical joint poses")
    if not np.allclose(np.diff(t),1/120,atol=1e-8) or not np.isfinite(position[valid]).all():
        raise ValueError("Invalid optical skeleton samples")
    if not np.isfinite(quaternion[valid]).all() or not np.allclose(np.linalg.norm(quaternion[valid],axis=-1),1,atol=1e-3):
        raise ValueError("Invalid measured segment orientation")
    if not np.allclose(canonical,meta["canonical"]["R_canon_nokov"],atol=1e-8):
        raise ValueError("Skeleton calibration differs from session metadata")
    selected = [names.index(name) for name in POINT_NAMES]
    leg = np.mean([sum(np.linalg.norm(offsets[names.index(name)]) for name in chain)
                   for chain in [("LeftLeg","LeftFoot"),("RightLeg","RightFoot")]])
    if not .3 < leg < 1.5: raise ValueError("Optical skeleton has invalid metric scale")
    measured = np.flatnonzero(valid)
    if not len(measured): raise ValueError("No measured skeleton frames")
    probe = measured[np.unique([0,len(measured)//2,len(measured)-1])]
    bvh_names,bvh_pos,bvh_rot = _bvh_reference(folder/"raw/mocap/Body0.bvh", source_index[probe])
    if bvh_names != names: raise ValueError("Vendor BVH and NPZ joint identities differ")
    # NOKOV's BVH translation/offset export uses centimetres; NPZ is metres.
    bvh_pos = (bvh_pos*.01) @ canonical.T
    bvh_rot = np.einsum("ij,ntjk->ntik",canonical,bvh_rot)
    pos_error = float(np.max(np.linalg.norm(bvh_pos-position[probe],axis=-1)))
    actual_rot = Rotation.from_quat(quaternion[probe].reshape(-1,4)).as_matrix().reshape(bvh_rot.shape)
    rot_error = float(np.max(Rotation.from_matrix((bvh_rot.swapaxes(-1,-2) @ actual_rot).reshape(-1,3,3)).magnitude()))
    if pos_error > .001 or rot_error > .002:
        raise ValueError(f"NPZ does not match vendor optical BVH: {pos_error} m, {rot_error} rad")
    poses = np.concatenate([position[:,selected],quaternion[:,selected][:,:,[3,0,1,2]]],axis=-1)
    return meta,t,source_index,poses,valid,float(leg),{"vendor_bvh_position_error_m":pos_error,"vendor_bvh_rotation_error_rad":rot_error}


def _video(source,dest,frames):
    temporary = dest.with_name(dest.stem+".tmp.mp4"); dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",str(source),"-an","-vf",
        "select=not(mod(n\\,3)),setpts=N/(10*TB),scale=224:224","-r","10","-fps_mode","cfr",
        "-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-threads","4",str(temporary)],check=True)
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames",
        "-show_entries","stream=nb_read_frames","-of","json",str(temporary)]))
    if int(probe["streams"][0]["nb_read_frames"]) != frames: raise ValueError("Video/optical table frame count mismatch")
    os.replace(temporary,dest)


def prepare_session(folder,row,index,output,definition,tasks,global_offset,*,history=False):
    dataset=output/DATASET;receipt=dataset/f"receipts/episode_{index:06d}.json"
    meta,t,source_index,poses,source_valid,leg,qa=load_optical_reference(folder,definition)
    annotation=json.loads((folder/"annotations.json").read_text());raw_manifest=json.loads((folder/"raw/manifest.json").read_text())
    view=meta["streams"]["videos"]["left_front"]
    if meta["participant"]!=row["participant"] or raw_manifest["mocap_capture"]["participant"]!=row["participant"]:
        raise ValueError("Camera wearer and optical source participant differ")
    if not view["calibrated"] or view["pts_gaps"] or view["fps"]!=30 or meta["alignment"]["peak_corr"]<.95:
        raise ValueError("Unverified camera/mocap synchronization")
    if meta["alignment"]["windows"]["lag_scatter_ms"]>5:
        raise ValueError("Camera/mocap alignment uncertainty exceeds policy")
    with np.load(folder/"videos/frame_timestamps.npz",allow_pickle=False) as data:
        video_t=data["head_left_front_t"].copy();video_indices=data["head_left_front_frame_index"].copy()
    if len(video_t)!=view["frames"] or not np.array_equal(video_indices,np.arange(len(video_t))):
        raise ValueError("Video index has gaps or differs from calibration metadata")
    frames=video_indices[::3];times=video_t[::3]
    nearest,error,valid=nearest_measurements(t,times,source_valid,causal=history)
    world=poses[nearest];segments=np.full(len(times),-1,dtype=np.int64);task_ids=np.full(len(times),-1,dtype=np.int64)
    for label,ep in enumerate(annotation["episodes"]):
        if ep["quality"]!="valid":continue
        active=(times>=ep["t_start"]) & (times<ep["t_end"])
        if (segments[active]>=0).any():raise ValueError("Overlapping vendor action annotations")
        text=ep.get("instruction") or annotation["instruction"]
        task=tasks.setdefault(text,len(tasks));segments[active]=label;task_ids[active]=task
    reference_quality,jumps=optical_quality(t,poses,source_valid)
    valid_training=measured_windows(valid,times,segments,nearest,reference_quality)
    if history:
        for i in np.flatnonzero(valid_training):
            past = nearest[i-3:i+1]
            if (past <= 0).any() or not reference_quality[past-1].all():valid_training[i] = False
    if not valid_training.any():raise ValueError("No fully measured synchronized training windows")
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved["source_session"]!=row["release_path"] or saved.get("observation_state") != (HISTORY if history else None):
            raise ValueError("Changed human source/history contract on resume")
        for path,digest in saved["outputs"].items():
            if sha256(output/path)!=digest:raise ValueError("Changed prepared human output")
        return saved
    actions=np.zeros((len(times),10,45),dtype=np.float32)
    for i in np.flatnonzero(valid_training):actions[i]=human_features(world[i+1:i+11],world[i],leg)
    label_path=dataset/f"labels/episode_{index:06d}.npz";label_path.parent.mkdir(parents=True,exist_ok=True)
    with label_path.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream,poses_world=world,source_mocap_frame_index=source_index[nearest],
                 source_mocap_array_index=nearest,source_video_frame_index=frames,video_timestamp=times,
                 mocap_timestamp=t[nearest],measurement_time_error=error,pose_valid=valid,
                 quality_valid=reference_quality[nearest],segment_id=segments,human_leg_length=leg)
    os.replace(label_path.with_suffix(".tmp"),label_path)
    video=dataset/f"videos/chunk-000/observation.images.cam_head/episode_{index:06d}.mp4"
    _video(folder/"videos/head_left_front.mp4",video,len(times))
    table=pa.table({"action":pa.array(actions.tolist(),type=pa.list_(pa.list_(pa.float32(),45),10)),
        "action.is_pad":pa.array(np.repeat((~valid_training)[:,None],10,axis=1).tolist(),type=pa.list_(pa.bool_(),10)),
        "timestamp":pa.array(times-times[0]),"sensor_timestamp":pa.array(times),"frame_index":np.arange(len(times)),
        "episode_index":np.full(len(times),index),"index":np.arange(global_offset,global_offset+len(times)),
        "task_index":task_ids,"valid_for_training":valid_training,
        "source_mocap_frame_index":source_index[nearest],"source_video_frame_index":frames})
    if history:
        state = np.zeros((len(times), 4, 75), dtype=np.float32)
        selected = np.flatnonzero(valid_training)
        state[selected] = human_history(poses, t, nearest[selected[:, None] + np.arange(-3, 1)], leg)
        table = table.append_column("observation.state", pa.array(state.tolist(), type=pa.list_(pa.list_(pa.float32(), 75), 4)))
    parquet=dataset/f"data/chunk-000/episode_{index:06d}.parquet";parquet.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(table,parquet.with_suffix(".tmp"),compression="zstd");os.replace(parquet.with_suffix(".tmp"),parquet)
    paths=[label_path,video,parquet]
    report={"episode_index":index,"source_session":row["release_path"],"participant":row["participant"],"split":row.get("evaluation_split",row["split"]) if history else row["split"],
        "family":row["family"],"primitive":row["primitive"],"length":len(times),"valid_samples":int(valid_training.sum()),
        "task_indices":sorted(set(task_ids[valid_training].tolist())),"human_leg_length":leg,
        "label_provenance":PROVENANCE,"measurement_pairing":"nearest actual optical frame; no interpolation or gap filling",
        "maximum_pairing_error_s":float(abs(error[valid]).max()),"invalid_pose_frames":int((~valid).sum()),
        "quality_policy":QUALITY_POLICY,"tracking_jump_transitions":jumps,
        "excluded_optical_quality_frames":int((~reference_quality).sum()),
        "source_synchronization":meta["alignment"],"source_quality_flags":meta["sanity"],**qa,
        "outputs":{str(p.relative_to(output)):sha256(p) for p in paths}}
    if history:
        report["observation_state"] = HISTORY
        report["original_split"] = row["split"]
        report["measurement_pairing"] = "latest actual optical frame at or before video timestamp; no interpolation"
    write_json(receipt,report);return report


def import_human_ego(raw_root,manifest_path,output,*,history=False):
    from .cache import cache_owner
    raw_root,output=Path(raw_root),Path(output);manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text());records=manifest["siblings"]
    for record in records:
        path=raw_root/record["rfilename"]
        if not path.resolve().is_relative_to(raw_root.resolve()):raise ValueError("Unsafe human source path")
        if not path.is_file() or path.stat().st_size!=record["size"]:raise FileNotFoundError(path)
        digest=record.get("lfs",{}).get("sha256")
        if digest:
            if sha256(path)!=digest:raise ValueError(f"Invalid optical source hash: {path}")
        elif hashlib.sha1(f"blob {record['size']}\0".encode()+path.read_bytes()).hexdigest()!=record["blobId"]:
            raise ValueError(f"Invalid source metadata hash: {path}")
    dataset=output/DATASET
    identity={"schema":SCHEMA,"subject_kind":"human","label_provenance":PROVENANCE,
        "source_repo":manifest["id"],"source_revision":manifest["sha"],"source_manifest_sha256":sha256(manifest_path),
        "point_names":POINT_NAMES,"coordinate_frame":"full human pelvis 3D frame anchored at observation t0",
        "position_scaling":"pelvis/ankles divided by measured human leg length; wrists in metres",
        "orientation":"global optical segment orientations relative to t0 pelvis; source segment axes retained",
        "floor":"not assumed: source world z=0 is not certified physical ground","end_effector_supervision":False,
        "quality_policy":QUALITY_POLICY}
    if history:identity["observation_state"] = HISTORY
    if manifest.get("split_policy"):identity["split_policy"] = manifest["split_policy"]
    cache_owner(dataset/"source.json",identity,[dataset/"data",dataset/"videos"])
    definition=json.loads((raw_root/"metadata/definitions/skeleton.json").read_text());tasks={};accepted=[];rejected=[]
    for row in manifest["sessions"]:
        try:
            entry=prepare_session(raw_root/row["release_path"],row,len(accepted),output,definition,tasks,sum(r["length"] for r in accepted),history=history)
            accepted.append(entry);print(f"Human ego {row['release_path']}: {entry['valid_samples']} measured windows",flush=True)
        except (ValueError,KeyError,AssertionError) as error:
            rejected.append({"source_session":row["release_path"],"reason":str(error)})
            print(f"Quarantined {row['release_path']}: {error}",flush=True)
    if not accepted:raise ValueError(f"No qualified optical human sessions: {rejected}")
    meta=dataset/"meta";meta.mkdir(parents=True,exist_ok=True)
    for name,rows in [("episodes.jsonl",accepted),("tasks.jsonl",[{"task_index":i,"task":t} for t,i in tasks.items()])]:
        (meta/name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    values=[]
    for row in accepted:
        if row["split"]!="train":continue
        table=pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        values.append(np.array(table["action"].to_pylist(),dtype=np.float32)[table["valid_for_training"].to_numpy()].reshape(-1,45))
    if not values:raise ValueError("No qualified human training split")
    values=np.concatenate(values)
    write_json(meta/"stats.json",{"action":{"count":len(values),"mean":values.mean(0).tolist(),"std":values.std(0).tolist(),
        "q01":np.quantile(values,.01,axis=0).tolist(),"q99":np.quantile(values,.99,axis=0).tolist()}})
    if history:
        states = []
        for row in accepted:
            if row["split"] != "train":continue
            table = pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet", columns=["observation.state", "valid_for_training"])
            states.append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[table["valid_for_training"].to_numpy()].reshape(-1, 75))
        states = np.concatenate(states)
        stats = json.loads((meta/"stats.json").read_text())
        stats["observation.state"] = {"count": len(states), "mean": states.mean(0).tolist(), "std": states.std(0).tolist()}
        write_json(meta/"stats.json", stats)
    write_json(meta/"splits.json",{s:[r["episode_index"] for r in accepted if r["split"]==s] for s in ["train","val","test"]})
    write_json(meta/"info.json",{**identity,"codebase_version":"v2.1","format_variant":"human_prechunked_action_10x45",
        "world_model":"Cosmos-Predict2.5-2B","fps":10,"total_episodes":len(accepted),"total_frames":sum(r["length"] for r in accepted),
        "total_tasks":len(tasks),"features":{**({"observation.state":{"dtype":"float32","shape":[4,75]}} if history else {}),"action":{"dtype":"float32","shape":[10,45]},
        "observation.images.cam_head":{"dtype":"video","shape":[3,224,224]}}})
    write_json(meta/"modality.json",{"action":{"five_point":[0,45]},"action_horizon":10,"proprioception":HISTORY if history else "not supplied; no robot joints or privileged mocap inputs",
        "end_effector_targets":"not available; not inferred","ground_contact_targets":"not available; not inferred"})
    if history:write_json(meta/"embodiment.json",{"embodiment_tag":"human_five_point","action_dimension":45,"end_effector_supervision":False,"label_provenance":PROVENANCE})
    report={"schema":SCHEMA,"label_provenance":PROVENANCE,"accepted":len(accepted),"rejected":rejected,
        "valid_windows":sum(r["valid_samples"] for r in accepted),"source_manifest_sha256":sha256(manifest_path),
        "subject_kind":"human","robot_fk_used":False,"smpl_fit_used":False,"vision_estimate_used":False}
    write_json(output/"qa_reports/human_ego.json",report);return report


def validate_human_wam(raw_root,manifest_path,output):
    import torch
    from safetensors import safe_open
    from torch.utils.data import DataLoader
    from .datasets import HumanWAMDataset
    raw_root,output=Path(raw_root),Path(output);dataset=output/DATASET
    source=json.loads((dataset/"source.json").read_text())
    if source.get("subject_kind")!="human" or source.get("label_provenance")!=PROVENANCE:
        raise ValueError("Non-optical labels cannot qualify as human reference data")
    if source["source_manifest_sha256"]!=sha256(manifest_path):raise ValueError("Human source selection changed")
    video=json.loads((output/"qa_reports/cache_video.json").read_text());text=json.loads((output/"prompt_bank.json").read_text())
    if not video["complete"] or not text["complete"] or sha256(output/"prompt_bank_bf16.safetensors")!=text["sha256"]:
        raise ValueError("Human caches incomplete or changed")
    episodes={r["episode_index"]:r for r in map(json.loads,(dataset/"meta/episodes.jsonl").read_text().splitlines())}
    if {r["episode_index"] for r in video["episodes"]}!=set(episodes):raise ValueError("Missing or duplicate human latent episodes")
    splits=json.loads((dataset/"meta/splits.json").read_text());members=[i for values in splits.values() for i in values]
    if len(members)!=len(set(members)) or set(members)!=set(episodes):raise ValueError("Invalid human dataset splits")
    official={}
    for split in ["train","val","test"]:
        for line in (raw_root/f"metadata/splits/{split}.txt").read_text().splitlines():
            if line in official:raise ValueError("Overlapping official splits")
            official[line]=split
    people={};samples=0;maximum_pairing=0.;recordings={};videos={};bvhs={};roundtrip_pos=0.;roundtrip_rot=0.
    for cached in video["episodes"]:
        row=episodes[cached["episode_index"]];episode=row["episode_index"]
        if official.get(row["source_session"])!=row.get("original_split",row["split"]) or episode not in splits[row["split"]]:
            raise ValueError("Changed vendor human split")
        people.setdefault(row["participant"],set()).add(row["split"])
        for path,digest in row["outputs"].items():
            if sha256(output/path)!=digest:raise ValueError("Changed paired human labels/video")
        folder=raw_root/row["source_session"]
        capture=json.loads((folder/"raw/manifest.json").read_text())
        group=str(capture["upload_batch_id"])+"/"+capture["mocap_capture"]["source_directory"]
        video_hash=next(d for p,d in row["outputs"].items() if p.endswith(".mp4"))
        for key,mapping in [(group,recordings),(video_hash,videos),(sha256(folder/"raw/mocap/Body0.bvh"),bvhs)]:
            if key in mapping and mapping[key]!=row["split"]:raise ValueError("Recording/video/BVH split leakage")
            mapping[key]=row["split"]
        with np.load(folder/"motions/skeleton.npz",allow_pickle=False) as original, np.load(dataset/f"labels/episode_{episode:06d}.npz",allow_pickle=False) as labels:
            names=original["joint_names"].tolist();point_index=[names.index(n) for n in POINT_NAMES]
            optical_t=original["t"];video_t=labels["video_timestamp"];valid=labels["pose_valid"];world=labels["poses_world"]
            nearest,error,expected_valid=nearest_measurements(optical_t,video_t,original["valid"],causal="observation_state" in source)
            expected=np.concatenate([original["positions_m"][nearest][:,point_index],
                original["quaternions_xyzw"][nearest][:,point_index][:,:,[3,0,1,2]]],axis=-1)
            if not np.array_equal(world,expected,equal_nan=True) or not np.array_equal(valid,expected_valid):
                raise ValueError("Human labels differ from measured optical source")
            if not np.array_equal(labels["source_mocap_frame_index"],original["frame_index"][nearest]):
                raise ValueError("Human source frame identity changed")
            if not np.allclose(labels["measurement_time_error"],error,atol=1e-12):raise ValueError("Changed video/optical pairing")
            maximum_pairing=max(maximum_pairing,float(abs(error[valid]).max()))
            original_world=np.concatenate([original["positions_m"][:,point_index],
                original["quaternions_xyzw"][:,point_index][:,:,[3,0,1,2]]],axis=-1)
            reference_quality,_=optical_quality(optical_t,original_world,original["valid"])
            if not np.array_equal(labels["quality_valid"],reference_quality[nearest]):raise ValueError("Changed optical quality mask")
            accepted=measured_windows(valid,video_t,labels["segment_id"],nearest,reference_quality)
            if "observation_state" in source:
                for i in np.flatnonzero(accepted):
                    past=nearest[i-3:i+1]
                    if (past<=0).any() or not reference_quality[past-1].all():accepted[i]=False
            leg=float(labels["human_leg_length"])
        table=pq.read_table(dataset/f"data/chunk-000/episode_{episode:06d}.parquet")
        if ("observation.state" in table.column_names) != ("observation_state" in source):
            raise ValueError("Undeclared human history contract")
        if not np.array_equal(accepted,table["valid_for_training"].to_numpy()):raise ValueError("Invalid human measurement window mask")
        frames=np.flatnonzero(accepted);actions=np.array(table["action"].to_pylist(),dtype=np.float32)
        for start in range(0,len(frames),128):
            idx=frames[start:start+128]
            expected_actions=human_features(world[idx[:,None]+np.arange(1,11)],world[idx],leg)
            if not np.allclose(actions[idx],expected_actions,atol=1e-5,rtol=0):
                raise ValueError("Human action targets are not derived from measured five-point poses")
            anchor=world[idx,0];basis=matrices(anchor[:,3:]);future=world[idx[:,None]+np.arange(1,11)]
            position=actions[idx,:,:15].reshape(-1,10,5,3).astype(np.float64);position[:,:,:3]*=leg
            restored=np.einsum('bij,btpj->btpi',basis,position)+anchor[:,None,None,:3]
            columns=actions[idx,:,15:].reshape(-1,10,5,2,3).swapaxes(-1,-2).astype(np.float64)
            relative=np.concatenate((columns,np.cross(columns[...,0],columns[...,1])[...,None]),axis=-1)
            orientation=basis[:,None,None]@relative
            roundtrip_pos=max(roundtrip_pos,float(np.linalg.norm(restored-future[...,:3],axis=-1).max()))
            delta=orientation.swapaxes(-1,-2)@matrices(future[...,3:])
            roundtrip_rot=max(roundtrip_rot,float(np.rad2deg(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude()).max()))
            if roundtrip_pos>=.001 or roundtrip_rot>=.1:raise ValueError('Five-point inverse coordinate roundtrip failed')
        if not np.isfinite(actions[frames]).all():raise ValueError("Nonfinite human supervision")
        if "observation_state" in source:
            state=np.asarray(table["observation.state"].to_pylist(),dtype=np.float32)
            expected=human_history(original_world,optical_t,nearest[frames[:,None]+np.arange(-3,1)],leg)
            if not np.allclose(state[frames],expected,atol=1e-5,rtol=0):raise ValueError("History differs from causal measurements")
        latent_path=output/cached["path"]
        if sha256(latent_path)!=cached["sha256"]:raise ValueError("Changed human latent cache")
        with safe_open(str(latent_path),framework="pt",device="cpu") as latent:
            if not np.array_equal(latent.get_tensor("frame_index").numpy(),frames):raise ValueError("Missing measured human windows in cache")
            if latent.get_slice("latents").get_shape()!=[len(frames),16,5,28,28]:raise ValueError("Wrong human latent shape")
            for start in range(0,len(frames),16):
                if not torch.isfinite(latent.get_slice("latents")[start:start+16]).all():raise ValueError("Nonfinite human latent")
        samples+=len(frames)
    if source.get("split_policy") != "recording_holdout" and any(len(values)!=1 for values in people.values()):
        raise ValueError("Human participant split leakage")
    batch=next(iter(DataLoader(HumanWAMDataset(output),batch_size=1)))
    expected_shapes={"video_latents":[1,16,5,28,28],"text_embeddings":[1,512,100352],"actions":[1,10,45]}
    if "observation_state" in source:expected_shapes["proprio"]=[1,4,75]
    for key,shape in expected_shapes.items():
        if list(batch[key].shape)!=shape or not torch.isfinite(batch[key]).all():raise ValueError("Invalid human training batch")
    if ("proprio" in batch) != ("observation_state" in source) or "embodiment_id" in batch:
        raise ValueError("Undeclared conditioning leaked into human WAM")
    head=torch.nn.Linear(16,450);loss=(head(batch["video_latents"].float().mean((2,3,4)))-batch["actions"].flatten(1)).square().mean();loss.backward()
    if not torch.isfinite(head.weight.grad).all():raise ValueError("Invalid human training gradients")
    preserved=json.loads((output/"qa_reports/mimic_preserved_before.json").read_text())
    if any(sha256(path)!=digest for path,digest in preserved.items()):raise ValueError("Existing Mimic metadata changed")
    mimic_files=0
    for path in preserved:
        manifest=Path(path)
        if manifest.name!="manifest.json":continue
        for clip in json.loads(manifest.read_text())["clips"]:
            for name,digest in clip["output_sha256"].items():
                if sha256(manifest.parent.parent/clip["path"]/name)!=digest:
                    raise ValueError("Existing Mimic trajectory changed")
                mimic_files+=1
    report={"ready":True,"scope":"human ego optical five-point training inputs","backbone":"Cosmos-Predict2.5-2B",
        "label_provenance":PROVENANCE,"robot_fk_used":False,"smpl_fit_used":False,"vision_estimate_used":False,
        "episodes":len(episodes),"participants":len(people),"cached_samples":samples,"text_tasks":text["tasks_cached"],
        "batch_shapes":expected_shapes,"maximum_video_mocap_pairing_error_s":maximum_pairing,"finite_gradient":True,
        "mimic_preserved":True,"mimic_files_verified":mimic_files,"mimic_fingerprints":preserved,"source_manifest_sha256":sha256(manifest_path),
        "video_cache_report_sha256":sha256(output/"qa_reports/cache_video.json"),"prompt_bank_sha256":text["sha256"],
        "coordinate_frame":source["coordinate_frame"],"supervision_limits":"No physical ground/contact, finger/EE, robot motor labels or robot proprioception; optical skeleton is mocap-derived reference, not error-free measurement.",
        "feature_roundtrip":{"maximum_position_error_m":roundtrip_pos,"maximum_rotation_error_degrees":roundtrip_rot,
            "scope":"inverse of declared t0 optical feature frame; robot embodiment bridge remains separate"},
        "full_backbone_training":"not run: loaded-batch and finite-gradient data checks only"}
    write_json(output/"qa_reports/human_wam_training_ready.json",report);return report
