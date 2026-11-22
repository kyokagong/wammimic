"""OpenHLM OpenPI trajectories -> paired WAM/Mimic data.

Layout reference: OpenHLM-project/OpenHLM, openpi/policies/g1_policy.py.
Root translation is kinematic odometry, not a recorded global-pose label.
"""
import io
import json
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation,Slerp

from .common import sha256, split_for, write_json
from .geometry import OPENHLM_BODY, features, matrices

SDK_JOINTS = [f"{side}_{joint}_joint" for side in ["left", "right"]
              for joint in ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]]
SDK_JOINTS += [f"waist_{axis}_joint" for axis in ["yaw", "roll", "pitch"]]
SDK_JOINTS += [f"{side}_{joint}_joint" for side in ["left", "right"]
               for joint in ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
POINTS = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]


class G1FK:
    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in POINTS]
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in SDK_JOINTS]
        if min(self.ids + joint_ids) < 0 or self.model.nq != 36:
            raise ValueError("Expected the G1 29-DoF asset with a floating root")
        self.addresses = self.model.jnt_qposadr[joint_ids]
        self.limits = self.model.jnt_range[joint_ids]
        self.sole_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in self.ids[1:3]
                           and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and self.model.geom_contype[i]]
        if not self.sole_geoms:
            raise ValueError("No calibrated foot contact spheres in asset")

    def sequence(self, states, timestamps, root_poses=None):
        q = states[:, OPENHLM_BODY]
        if ((q < self.limits[:, 0] - 0.05) | (q > self.limits[:, 1] + 0.05)).any():
            raise ValueError("Measured joint angle exceeds asset limits by >0.05 rad")
        yaw = np.r_[0, np.cumsum(states[:-1, 33] * np.diff(timestamps))]
        rotations = (Rotation.from_euler("xyz", np.c_[states[:, 31:33], yaw]) if root_poses is None
                     else Rotation.from_quat(root_poses[:,[4,5,6,3]]))
        result = np.empty((len(states), 5, 7), dtype=np.float32)
        root = np.zeros(3)
        previous = None
        for i, (joints, rotation) in enumerate(zip(q, rotations.as_quat())):
            self.data.qpos[:3] = 0
            self.data.qpos[3:7] = rotation[[3, 0, 1, 2]]
            self.data.qpos[self.addresses] = joints
            mujoco.mj_kinematics(self.model, self.data)
            positions = self.data.xpos[self.ids].copy()
            sole_z = self.data.geom_xpos[self.sole_geoms, 2] - self.model.geom_size[self.sole_geoms, 0]
            if root_poses is not None:
                root=root_poses[i,:3]
            elif previous is not None:
                stance = 1 + int(positions[2, 2] < positions[1, 2])
                root[:2] += previous[stance, :2] - positions[stance, :2]
            if root_poses is None:root[2] = -sole_z.min()
            result[i, :, :3] = positions + root
            result[i, :, 3:] = self.data.xquat[self.ids]
            previous = positions
        return result


def proprioceptive_states(states,commands,timestamps,poses):
    """Only current/past 30 Hz measurements; interpolated targets stay separate."""
    q=states[:,OPENHLM_BODY];velocity=np.zeros_like(q);omega=np.zeros((len(q),3))
    velocity[1:]=np.diff(q,axis=0)/np.diff(timestamps)[:,None]
    rotation=matrices(poses[:,0,3:])
    gravity=np.einsum("nji,j->ni",rotation,np.array([0,0,-1]))
    omega[1:]=Rotation.from_matrix(rotation[:-1].transpose(0,2,1) @ rotation[1:]).as_rotvec()/np.diff(timestamps)[:,None]
    return np.c_[q,velocity,gravity,omega,states[:,[7,15]],commands[:,[7,15]]-states[:,[7,15]]].astype(np.float32)


def action_chunks(poses10, ee, leg_length=0.68, *, anchors=None):
    """Targets t+0.1..t+1.0 all share the observation's t0 reference."""
    count = len(poses10)
    anchors=poses10 if anchors is None else np.asarray(anchors)
    if anchors.shape!=poses10.shape:raise ValueError("Action anchors must match the observation timeline")
    chunks = np.zeros((count, 10, 47), dtype=np.float32)
    pad = np.ones((count, 10), dtype=bool)
    for i in range(count):
        n = min(10, count-i-1)
        if n:
            chunks[i, :n, :45] = features(poses10[i+1:i+1+n], anchors[i], leg_length)
            chunks[i, :n, 45:] = ee[i+1:i+1+n]
            pad[i, :n] = False
    return chunks, pad


def filtered_poses10(poses50):
    # Hemisphere alignment prevents sign cancellation when filtering quaternions.
    values = poses50.astype(np.float64).copy()
    for i in range(1, len(values)):
        flip = np.sum(values[i, :, 3:] * values[i-1, :, 3:], axis=-1) < 0
        values[i, flip, 3:] *= -1
    values = resample_poly(values, 1, 5, axis=0, padtype="line")
    values[..., 3:] /= np.linalg.norm(values[..., 3:], axis=-1, keepdims=True)
    matrices(values[..., 3:])
    return values.astype(np.float32)


def encode_video(images, filename):
    temporary = filename.with_name(filename.stem + ".tmp.mp4")
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "224x224", "-r", "10", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", str(temporary)]
    filename.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    visible = False
    try:
        for image in images:
            if not isinstance(image, dict) or image.get("bytes") is None:
                raise ValueError("Missing embedded head image")
            with Image.open(io.BytesIO(image["bytes"])) as frame:
                rgb = np.asarray(frame.convert("RGB").resize((224,224), Image.Resampling.LANCZOS))
            visible |= bool(rgb.any())
            process.stdin.write(rgb.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait() or not visible:
            raise ValueError(error or "Head video consists only of black placeholders")
        os.replace(temporary, filename)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)


def prepare_episode(source, expected_hash, episode_index, fk, output, task_map, global_offset=0):
    if sha256(source) != expected_hash:
        raise ValueError("Input download SHA256 mismatch")
    dataset = output / "wam_lerobot_v2/g1_parallel_gripper"
    receipt = dataset / "receipts" / f"episode_{episode_index:06d}.json"
    if receipt.exists():
        entry = json.loads(receipt.read_text())
        if entry["input_sha256"] != expected_hash:
            raise ValueError("Changed source on resume")
        for name, digest in entry["outputs"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Output hash mismatch: {name}")
        return entry
    table = pq.read_table(source, columns=["state","actions","timestamp","task_index","head_image_left"])
    states = np.array(table["state"].to_pylist(), dtype=np.float64)
    actions = np.array(table["actions"].to_pylist(), dtype=np.float64)
    timestamps = np.array(table["timestamp"].to_pylist(), dtype=np.float64)
    tasks = table["task_index"].to_numpy()
    if states.shape != (len(timestamps),34) or actions.shape != states.shape:
        raise ValueError("Expected official 34D OpenPI state/action layout")
    if not np.isfinite(np.c_[states,actions,timestamps]).all() or len(states)<40:
        raise ValueError("Nonfinite or too-short episode")
    if not np.allclose(np.diff(timestamps), 1/30, atol=2e-4):
        raise ValueError("Expected continuous 30Hz source timestamps")
    if len(set(tasks)) != 1 or int(tasks[0]) not in task_map:
        raise ValueError("Missing or inconsistent task annotation")
    if (states[:,[7,15]] < 0).any() or (states[:,[7,15]] > 1).any() or (actions[:,[7,15]] < 0).any() or (actions[:,[7,15]] > 1).any():
        raise ValueError("EE scalars outside declared [0,1] range")
    timestamps -= timestamps[0]
    t50 = np.arange(int(np.floor(timestamps[-1]*50+1e-5))+1) / 50
    query50=np.minimum(t50,timestamps[-1])
    poses30=fk.sequence(states,timestamps)
    states50 = PchipInterpolator(timestamps, states)(query50)
    root_position=PchipInterpolator(timestamps,poses30[:,0,:3])(query50)
    root_quat=Slerp(timestamps,Rotation.from_quat(poses30[:,0,3:][:,[1,2,3,0]]))(query50).as_quat()
    poses50 = fk.sequence(states50,t50,root_poses=np.c_[root_position,root_quat[:,[3,0,1,2]]])
    q50 = states50[:,OPENHLM_BODY]
    joints50 = np.stack([q50, np.diff(q50, axis=0, prepend=q50[:1])/0.02], axis=1).astype(np.float32)
    poses10 = filtered_poses10(poses50)
    t10 = np.arange(len(poses10))/10
    source_indices = np.searchsorted(timestamps, t10-1e-5).clip(0,len(states)-1)
    commands = actions[source_indices]
    ee = commands[:,[7,15]].astype(np.float32)
    chunks, pad = action_chunks(poses10,ee,anchors=poses30[source_indices])
    state=proprioceptive_states(states,actions,timestamps,poses30)[source_indices]
    traj_id = f"g1/openhlm_{episode_index:06d}"
    morph = output / "morph_5pt_store" / traj_id
    morph.mkdir(parents=True, exist_ok=True)
    for name, array in [("poses.npy", poses50), ("joints.npy", joints50)]:
        with (morph/(name+".tmp")).open("wb") as stream:
            np.save(stream,array,allow_pickle=False)
        os.replace(morph/(name+".tmp"),morph/name)
    video = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{episode_index:06d}.mp4"
    images = table["head_image_left"].to_pylist()
    encode_video((images[i] for i in source_indices), video)
    count=len(state)
    valid=(np.arange(count)>=3)&~pad.any(axis=1)
    frame_table=pa.table({"observation.state":pa.array(state.tolist(),type=pa.list_(pa.float32(),68)),
        "action":pa.array(chunks.tolist(),type=pa.list_(pa.list_(pa.float32(),47),10)),
        "action.is_pad":pa.array(pad.tolist(),type=pa.list_(pa.bool_(),10)),
        "timestamp":pa.array(t10,type=pa.float32()),"frame_index":np.arange(count),
        "episode_index":np.full(count,episode_index),"index":np.arange(global_offset,global_offset+count),
        "task_index":np.full(count,int(tasks[0])),
        "traj_id":[traj_id]*count,"valid_for_training":valid})
    parquet=dataset/f"data/chunk-000/episode_{episode_index:06d}.parquet"
    parquet.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(frame_table,parquet.with_suffix(".tmp"),compression="zstd")
    os.replace(parquet.with_suffix(".tmp"),parquet)
    paths=[morph/"poses.npy",morph/"joints.npy",video,parquet]
    entry={"episode_index":episode_index,"source_episode":source.stem,"traj_id":traj_id,"task_index":int(tasks[0]),
           "tasks":[task_map[int(tasks[0])]],"length":count,"valid_samples":int(valid.sum()),
           "frames50":len(poses50),"input_sha256":expected_hash,"split":split_for(f"openhlm/{source.stem}"),
           "root_pose_source":"stance_kinematic_odometry; not measured global ground truth",
           "outputs":{str(p.relative_to(output)):sha256(p) for p in paths}}
    write_json(receipt,entry)
    return entry


def import_openhlm(raw_root, download_manifest, output, asset_xml):
    raw_root, output = Path(raw_root), Path(output)
    source_root = raw_root/"openhlm_raw"
    metadata=json.loads(Path(download_manifest).read_text())
    source_files=[f for f in metadata["siblings"] if f["rfilename"].endswith(".parquet")]
    missing=[f["rfilename"] for f in source_files if not (source_root/f["rfilename"]).is_file()
             or (source_root/f["rfilename"]).stat().st_size!=f["size"]]
    if missing:
        report={"accepted":0,"pending":missing,"all_inputs_available":False,"raw_wam_ready":False,"dit_latents_ready":False}
        write_json(output/"qa_reports/openhlm.json",report)
        return report
    tasks={r["task_index"]:r["task"] for r in map(json.loads,(source_root/"g1_HLM-12_full_teleop/meta/tasks.jsonl").read_text().splitlines())}
    fk=G1FK(asset_xml)
    accepted=[];pending=[];rejected=[]
    owner=output/"wam_lerobot_v2/g1_parallel_gripper/source.json"
    identity={"schema":"wammimic-openhlm-v1","geometry":"support_xy_lower_pelvis_wrist_v1",
              "action_anchor":"unfiltered_observation_t0_v1",
              "root_interpolation":"causal_30hz_stance_pchip_slerp_v1",
              "source_manifest_sha256":sha256(download_manifest),"asset_sha256":sha256(asset_xml)}
    if not owner.exists() and owner.parent.exists() and any(owner.parent.iterdir()):
        raise FileExistsError("Refuse nonempty unowned WAM dataset")
    if owner.exists() and json.loads(owner.read_text())!=identity:
        raise ValueError("Output belongs to different source manifest or asset")
    write_json(owner,identity)
    for f in source_files:
        source=source_root/f["rfilename"]
        index=int(source.stem.split("_")[-1])
        if not source.exists() or source.stat().st_size!=f["size"]:
            pending.append(f["rfilename"]);continue
        try:
            accepted.append(prepare_episode(source,f["lfs"]["sha256"],len(accepted),fk,output,tasks,
                                            global_offset=sum(e["length"] for e in accepted)))
            print(f"OpenHLM episode {index}: prepared",flush=True)
        except (ValueError,KeyError) as error:
            rejected.append({"episode_index":index,"reason":str(error)})
    dataset=output/"wam_lerobot_v2/g1_parallel_gripper";meta=dataset/"meta";meta.mkdir(parents=True,exist_ok=True)
    for filename, rows in [("episodes.jsonl",accepted),("tasks.jsonl",[{"task_index":k,"task":v} for k,v in tasks.items()])]:
        (meta/filename).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    statistics={"observation.state":[],"action":[]}
    for e in accepted:
        if e["split"]!="train":continue
        table=pq.read_table(dataset/f"data/chunk-000/episode_{e['episode_index']:06d}.parquet")
        valid=np.array(table["valid_for_training"])
        for key in statistics:
            values=np.array(table[key].to_pylist(),dtype=np.float32)[valid]
            statistics[key].append(values.reshape(-1,values.shape[-1]))
    stats={}
    for key, arrays in statistics.items():
        if not arrays:continue
        x=np.concatenate(arrays)
        stats[key]={"count":len(x),"mean":x.mean(axis=0).tolist(),"std":x.std(axis=0).tolist(),
                    "min":x.min(axis=0).tolist(),"max":x.max(axis=0).tolist(),
                    "q01":np.quantile(x,.01,axis=0).tolist(),"q99":np.quantile(x,.99,axis=0).tolist()}
    write_json(meta/"stats.json",stats)
    write_json(meta/"embodiment.json",{"robot_type":"g1","embodiment_tag":"bimanual_parallel_gripper","id":1})
    write_json(meta/"modality.json",{"state":{"body_q":[0,29],"body_qvel":[29,58],"gravity":[58,61],"angular_velocity":[61,64],"ee_measured":[64,66],"ee_residual":[66,68]},
        "action":{"five_point":[0,45],"ee":[45,47]},"action_horizon":10,"action_reference":"observation t0 virtual pelvis heading", "ee_range":[0,1],"closure_direction":"1=closed"})
    features_schema={"observation.state":{"dtype":"float32","shape":[68]},"action":{"dtype":"float32","shape":[10,47]},
        "action.is_pad":{"dtype":"bool","shape":[10]},"valid_for_training":{"dtype":"bool","shape":[1]},
        "traj_id":{"dtype":"string","shape":[1]},"timestamp":{"dtype":"float32","shape":[1]},
        "observation.images.cam_head":{"dtype":"video","shape":[3,224,224],"names":["channels","height","width"],
                                       "info":{"video.fps":10,"video.height":224,"video.width":224,"video.codec":"h264","video.pix_fmt":"yuv420p","video.is_depth_map":False}}}
    features_schema.update({key:{"dtype":"int64","shape":[1]} for key in ["index","frame_index","episode_index","task_index"]})
    write_json(meta/"info.json",{"codebase_version":"v2.1","format_variant":"prechunked_action_10x47","robot_type":"g1","world_model":"Cosmos-Predict2.5-2B","fps":10,"total_episodes":len(accepted),
        "total_frames":sum(e["length"] for e in accepted),"total_tasks":len(set(e["task_index"] for e in accepted)),
        "total_videos":len(accepted),"total_chunks":1,"chunks_size":1000,"features":features_schema,
        "data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"})
    write_json(meta/"splits.json",{s:[e["episode_index"] for e in accepted if e["split"]==s] for s in ["train","val","test"]})
    morph_entries=[]
    for entry in accepted:
        path=entry["traj_id"]
        morph_entries.append({"traj_id":path,"source_id":"openhlm/"+entry["source_episode"],
            "group_id":"openhlm/"+entry["source_episode"],"dataset":"openhlm","robot":"g1",
            "split":entry["split"],"frames":entry["frames50"],"fps":50,"leg_length":.68,
            "path":path,"joint_names":SDK_JOINTS,"input_sha256":entry["input_sha256"],
            "five_point_source":"robot_fk_with_stance_odometry",
            "reference_floor_offset_m":0.,"reference_floor_valid":True,
            "output_sha256":{name:entry["outputs"][f"morph_5pt_store/{path}/{name}"] for name in ["poses.npy","joints.npy"]}})
    write_json(output/"morph_5pt_store/g1/openhlm_manifest.json",{"fps":50,"window_dim":465,"clips":morph_entries})
    main_path=output/"morph_5pt_store/g1/manifest.json"
    combined=json.loads(main_path.read_text()) if main_path.exists() else {"schema":"wammimic-morph-v1","fps":50,"window_dim":465,"clips":[]}
    combined["clips"]=[r for r in combined["clips"] if r.get("dataset")!="openhlm"]+morph_entries
    write_json(main_path,combined)
    report={"accepted":len(accepted),"pending":pending,"rejected":rejected,"all_inputs_available":not pending,
        "raw_wam_ready":bool(accepted),"dit_latents_ready":False,
        "backbone":"nvidia/Cosmos-Predict2.5-2B",
        "latent_blocker":"Cosmos-Predict2.5 official tokenizer/text-encoder cache has not been generated.",
        "auxiliary_sources":{"egowalk":"5Hz RGB/navigation data; lacks full-body action labels", "oakink2":"hand/object annotations only; images and target-hand retargeting are required"}}
    write_json(output/"qa_reports/openhlm.json",report)
    return report


def upgrade_openhlm_5pt_contract(output):
    """Align existing OpenHLM 406 episodes to unified t0 full-3D pelvis actions and 4x79 5-pt history state.

    Reuses already-cached 224x224 MP4s, Cosmos VAE video latents, and Reason1 text embeddings while
    upgrading Parquet columns, receipts, metadata, and train-only statistics.
    """
    from .geometry import rotation6d
    from .human_ego import human_features, human_history

    output = Path(output)
    dataset = output / "wam_lerobot_v2/g1_parallel_gripper"
    meta = dataset / "meta"
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line.strip()]
    splits = json.loads((meta / "splits.json").read_text())
    train_set = set(splits["train"])
    leg_length = 0.68

    max_pos_err, max_rot_err = 0.0, 0.0
    stats_acc = {"observation.state": [], "observation.state_5pt_hist": [], "action": []}
    updated_episodes = []
    for entry in episodes:
        ep = entry["episode_index"]
        parquet_path = dataset / f"data/chunk-000/episode_{ep:06d}.parquet"
        poses50 = np.load(output / "morph_5pt_store" / entry["traj_id"] / "poses.npy").astype(np.float64)
        t50 = np.arange(len(poses50), dtype=np.float64) / 50.0
        poses10 = filtered_poses10(poses50)
        table = pq.read_table(parquet_path)
        count = len(table)
        if len(poses10) != count:
            raise ValueError(f"Episode {ep} poses10 length {len(poses10)} != parquet rows {count}")
        old_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        old_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        pad = np.asarray(table["action.is_pad"].to_pylist(), dtype=bool)
        valid = table["valid_for_training"].to_numpy()
        # Unfiltered 50Hz t0 anchors corresponding to each 10Hz step (every 5th 50Hz frame)
        idx50 = np.minimum(np.arange(count) * 5, len(poses50) - 1)
        anchors = poses50[idx50]
        anchor_quat = anchors[:, 0, 3:].astype(np.float32)

        # Recompute 10x45 actions in t0 full 3D pelvis frame; preserve EE commands in [0, 1]
        actions = np.zeros((count, 10, 47), dtype=np.float32)
        actions[:, :, 45:] = old_action[:, :, 45:]
        for i in range(count):
            n = min(10, count - i - 1)
            if n:
                actions[i, :n, :45] = human_features(poses10[i + 1:i + 1 + n], anchors[i], leg_length)

        # Causal 4-frame 5-point pose/vel history (4x75) + 4-frame [ee_measured(2), ee_residual(2)] (4x4) -> 4x79
        state_5pt = np.zeros((count, 4, 79), dtype=np.float32)
        selected = np.flatnonzero(valid)
        if len(selected):
            hist_idx10 = np.clip(selected[:, None] + np.arange(-3, 1), 0, count - 1)
            # Frame 0 has no predecessor for finite-difference velocity; clamp earliest history slot to 1 (t=0.02s <= t0=0.30s, strictly causal).
            hist_idx50 = np.maximum(idx50[hist_idx10], 1)
            state_5pt[selected, :, :75] = human_history(poses50, t50, hist_idx50, leg_length)
            state_5pt[selected, :, 75:] = old_state[hist_idx10, 64:68]

        if not np.isfinite(actions).all() or not np.isfinite(state_5pt).all():
            raise ValueError(f"Nonfinite upgraded openhlm episode {ep}")

        # Verify full 3D pelvis roundtrip on first valid frame of first 20 episodes
        if ep < 20 and len(selected):
            i0 = int(selected[0])
            r0 = matrices(anchors[i0:i0 + 1, 0, 3:])[0]
            pos_local = actions[i0, :, :15].reshape(10, 5, 3).copy()
            pos_local[:, :3] *= leg_length
            pos_world = pos_local @ r0.T + anchors[i0, 0, :3]
            rot_local = rotation6d(actions[i0, :, 15:45].reshape(10, 5, 6))
            rot_world = r0[None, None] @ rot_local
            gt_future = poses10[i0 + 1:i0 + 11]
            gt_rot = matrices(gt_future[:, :, 3:].reshape(-1, 4)).reshape(10, 5, 3, 3)
            max_pos_err = max(max_pos_err, float(np.linalg.norm(pos_world - gt_future[:, :, :3], axis=-1).max()))
            tr = np.clip((np.einsum("...ij,...ij->...", rot_world, gt_rot) - 1.0) / 2.0, -1.0, 1.0)
            max_rot_err = max(max_rot_err, float(np.rad2deg(np.arccos(tr)).max()))

        new_table = pa.table({
            "observation.state": table["observation.state"],
            "observation.state_5pt_hist": pa.array(state_5pt.tolist(), type=pa.list_(pa.list_(pa.float32(), 79), 4)),
            "anchor_quaternion_wxyz": pa.array(anchor_quat.tolist(), type=pa.list_(pa.float32(), 4)),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 47), 10)),
            "action.is_pad": table["action.is_pad"],
            "timestamp": table["timestamp"],
            "frame_index": table["frame_index"],
            "episode_index": table["episode_index"],
            "index": table["index"],
            "task_index": table["task_index"],
            "traj_id": table["traj_id"],
            "valid_for_training": table["valid_for_training"],
        })
        pq.write_table(new_table, parquet_path.with_suffix(".tmp"), compression="zstd")
        os.replace(parquet_path.with_suffix(".tmp"), parquet_path)

        rel_pq = str(parquet_path.relative_to(output))
        entry["outputs"][rel_pq] = sha256(parquet_path)
        write_json(dataset / "receipts" / f"episode_{ep:06d}.json", entry)
        updated_episodes.append(entry)

        if ep in train_set and len(selected):
            stats_acc["observation.state"].append(old_state[selected])
            stats_acc["observation.state_5pt_hist"].append(state_5pt[selected].reshape(-1, 79))
            stats_acc["action"].append(actions[selected].reshape(-1, 47))

    if max_pos_err >= 1e-3 or max_rot_err >= 0.1:
        raise ValueError(f"OpenHLM 3D pelvis roundtrip error exceeded gate: {max_pos_err} m, {max_rot_err} deg")

    (meta / "episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated_episodes))
    stats = {}
    for key, arrays in stats_acc.items():
        x = np.concatenate(arrays)
        stats[key] = {
            "count": len(x), "mean": x.mean(axis=0).tolist(), "std": x.std(axis=0).tolist(),
            "min": x.min(axis=0).tolist(), "max": x.max(axis=0).tolist(),
            "q01": np.quantile(x, .01, axis=0).tolist(), "q99": np.quantile(x, .99, axis=0).tolist(),
        }
    write_json(meta / "stats.json", stats)

    source_meta = json.loads((dataset / "source.json").read_text())
    source_meta.update({
        "coordinate_frame": "full t0 pelvis 3D; explicit anchor quaternion in every row",
        "extra_state_columns": ["observation.state_5pt_hist"],
        "history_frames": 4,
        "leg_length_m": leg_length,
    })
    write_json(dataset / "source.json", source_meta)

    modality = json.loads((meta / "modality.json").read_text())
    modality.update({
        "state_5pt_hist": {"pose": [0, 45], "linear_velocity": [45, 60], "angular_velocity": [60, 75],
                           "ee_measured": [75, 77], "ee_residual": [77, 79]},
        "history_frames": 4,
        "action_reference": "full t0 pelvis 3D; explicit anchor quaternion in every row",
    })
    write_json(meta / "modality.json", modality)

    info = json.loads((meta / "info.json").read_text())
    info["features"]["observation.state_5pt_hist"] = {"dtype": "float32", "shape": [4, 79]}
    info["features"]["anchor_quaternion_wxyz"] = {"dtype": "float32", "shape": [4]}
    info["coordinate_frame"] = "full t0 pelvis 3D; explicit anchor quaternion in every row"
    write_json(meta / "info.json", info)

    report = {
        "upgraded_episodes": len(updated_episodes),
        "coordinate_frame": "full t0 pelvis 3D",
        "state_5pt_hist_shape": [4, 79],
        "maximum_position_roundtrip_m": max_pos_err,
        "maximum_rotation_roundtrip_degrees": max_rot_err,
    }
    write_json(output / "qa_reports/openhlm_5pt_upgrade.json", report)
    return report
