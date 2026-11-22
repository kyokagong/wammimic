import json
from pathlib import Path
import tempfile
import subprocess
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
from safetensors.torch import save_file

from wammimic.data_pipeline.human_ego import (human_features,human_history,nearest_measurements,valid_windows,
    load_optical_reference,_bvh_reference,_video,optical_quality,measured_windows,SCHEMA,PROVENANCE,DATASET,HISTORY)
from wammimic.data_pipeline.datasets import HumanWAMDataset


class HumanEgoTest(unittest.TestCase):
    def test_history_is_measured_causal_and_has_metric_velocities(self):
        times=np.arange(80)/120
        poses=np.zeros((80,5,7));poses[...,3]=1
        poses[...,:3]=times[:,None,None]*np.array([.3,.2,.1])
        poses[...,3:]=Rotation.from_rotvec(times[:,None]*np.array([0.,0.,.6])).as_quat()[:,[3,0,1,2]][:,None]
        indices=np.array([[12,24,36,48]])
        result=human_history(poses,times,indices,.8)
        self.assertEqual(result.shape,(1,4,75))
        heading=Rotation.from_euler('z',times[48]*.6).as_matrix()
        velocity=np.array([.3,.2,.1])@heading
        np.testing.assert_allclose(result[0,:,45:54].reshape(4,3,3),np.broadcast_to(velocity/.8,(4,3,3)),atol=1e-6)
        np.testing.assert_allclose(result[0,:,54:60].reshape(4,2,3),np.broadcast_to(velocity,(4,2,3)),atol=1e-6)
        np.testing.assert_allclose(result[0,:,60:].reshape(4,5,3),np.broadcast_to([0,0,.6],(4,5,3)),atol=1e-6)
        changed=poses.copy();changed[49:]=np.nan
        np.testing.assert_array_equal(result,human_history(changed,times,indices,.8))
        with self.assertRaises(ValueError):human_history(poses,times,np.array([[0,12,24,36]]),.8)
        video=np.array([.101,.203,.304,.405])
        nearest,error,valid=nearest_measurements(times,video,np.ones(80,dtype=bool),causal=True)
        self.assertTrue(valid.all());self.assertTrue((times[nearest]<=video).all());self.assertTrue((error<=0).all())

    def test_optical_quality_excludes_tracking_jump_without_repair(self):
        times=np.arange(360)/120;poses=self.poses(360);poses[150,3,0]+=1
        original=poses.copy();quality,jumps=optical_quality(times,poses,np.ones(360,dtype=bool))
        self.assertEqual(jumps,2);self.assertFalse(quality[150]);self.assertTrue(quality[100])
        np.testing.assert_array_equal(poses,original)
        poses=self.poses(360);poses[150,3,3:]=[0,1,0,0]
        quality,jumps=optical_quality(times,poses,np.ones(360,dtype=bool))
        self.assertEqual(jumps,2);self.assertFalse(quality[150])

    def test_window_rejects_missing_raw_frame_between_sampled_video_frames(self):
        times=np.arange(40)/10;nearest=np.arange(40)*12;quality=np.ones(480,dtype=bool)
        valid=np.ones(40,dtype=bool);segments=np.zeros(40,dtype=int)
        quality[125]=False
        self.assertFalse(measured_windows(valid,times,segments,nearest,quality)[10])
        self.assertTrue(measured_windows(valid,times,segments,nearest,quality)[24])

    def test_video_keeps_every_third_source_frame_at_ten_hz(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/"source.mp4";dest=Path(folder)/"result.mp4"
            subprocess.run(["ffmpeg","-nostdin","-v","error","-f","lavfi","-i",
                "testsrc2=size=224x224:rate=30","-frames:v","18","-threads","1",str(source)],check=True)
            _video(source,dest,6)
            result=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                "-show_entries","stream=avg_frame_rate","-of","json",str(dest)]))
            self.assertEqual(result["streams"][0]["avg_frame_rate"],"10/1")

    def poses(self,n=20):
        value=np.zeros((n,5,7));value[...,3]=1
        value[:,0,2]=1.;value[:,1:3,2]=.1;value[:,3:5,0]=.4;value[:,3:5,2]=1.2
        return value

    def test_human_canonical_frame_is_invariant_to_world_rigid_transform(self):
        original=self.poses();rotation=Rotation.from_euler("zyx",[.7,.2,-.3])
        changed=original.copy();changed[...,:3]=rotation.apply(original[...,:3].reshape(-1,3)).reshape(original[...,:3].shape)+[3,-2,8]
        changed[...,3:]=rotation.as_quat()[[3,0,1,2]]
        np.testing.assert_allclose(human_features(original[1:11],original[0],.8),human_features(changed[1:11],changed[0],.8),atol=1e-6)

    def test_leg_scale_is_human_specific_and_does_not_scale_wrists(self):
        poses=self.poses();a=human_features(poses[1:11],poses[0],1.);b=human_features(poses[1:11],poses[0],.5)
        np.testing.assert_allclose(b[:,:9],a[:,:9]*2)
        np.testing.assert_allclose(b[:,9:],a[:,9:])
        self.assertAlmostEqual(a[0,2],0.)
        np.testing.assert_allclose(human_features(poses[1:11][None].repeat(2,0),poses[:2],.8)[0],human_features(poses[1:11],poses[0],.8))

    def test_missing_mocap_is_not_filled_by_another_frame(self):
        times=np.arange(5)/120;valid=np.array([True,False,True,True,True])
        index,error,accepted=nearest_measurements(times,np.array([.002,.0084,.0165,.1]),valid)
        np.testing.assert_array_equal(index,[0,1,2,4])
        np.testing.assert_array_equal(accepted,[True,False,True,False])
        self.assertTrue((abs(error[accepted])<=1/240+1e-5).all())

    def test_windows_reject_invalid_measurements_annotation_boundaries_and_time_gaps(self):
        valid=np.ones(40,dtype=bool);times=np.arange(40)/10;segments=np.zeros(40,dtype=int)
        self.assertEqual(valid_windows(valid,times,segments).sum(),25)
        valid[15]=False;self.assertFalse(valid_windows(valid,times,segments)[10])
        valid[:]=True;segments[20:]=1;self.assertFalse(valid_windows(valid,times,segments)[18])
        self.assertTrue(valid_windows(valid,times,segments)[24])
        times[22:]+=.1;self.assertFalse(valid_windows(valid,times,segments)[24])

    def test_estimated_or_fitted_skeleton_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError,"Only vendor optical"):
                load_optical_reference(Path(path),{"source":"SMPL-X fitting / vision estimator"})

    def test_bvh_uses_vendor_joint_rotations_and_does_not_double_root_offset(self):
        text="""HIERARCHY
ROOT Hips
{
 OFFSET 100 200 300
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftFoot
 {
  OFFSET 0 -100 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site { OFFSET 0 0 10 }
 }
}
MOTION
Frames: 2
Frame Time: 0.00833333
100 200 300 90 0 0 0 0 0
100 200 300 0 0 0 0 0 0
"""
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"Body0.bvh";path.write_text(text)
            names,positions,_=_bvh_reference(path,[0,1])
            self.assertEqual(names,["Hips","LeftFoot"])
            np.testing.assert_allclose(positions[:,0],[[100,200,300],[100,200,300]])
            np.testing.assert_allclose(positions[:,1],[[200,200,300],[100,100,300]],atol=1e-10)

    def test_human_reader_uses_frame_prompt_and_has_no_robot_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/DATASET;meta=dataset/"meta";meta.mkdir(parents=True)
            (dataset/"source.json").write_text(json.dumps({"schema":SCHEMA,"subject_kind":"human","label_provenance":PROVENANCE}))
            (meta/"episodes.jsonl").write_text(json.dumps({"episode_index":0,"task_indices":[0,1]})+"\n")
            (meta/"splits.json").write_text(json.dumps({"train":[0],"val":[],"test":[]}))
            (meta/"stats.json").write_text(json.dumps({"action":{"mean":[0]*45,"std":[1]*45}}))
            data=dataset/"data/chunk-000";latent=dataset/"latents/chunk-000";data.mkdir(parents=True);latent.mkdir(parents=True)
            tasks=np.zeros(20,dtype=np.int64);tasks[5]=1
            pq.write_table(pa.table({"task_index":tasks,"action":np.zeros((20,10,45)).tolist()}),data/"episode_000000.parquet")
            save_file({"frame_index":torch.tensor([5]),"latents":torch.zeros((1,16,5,28,28),dtype=torch.bfloat16)},str(latent/"episode_000000.safetensors"))
            save_file({"task_0":torch.tensor([1.,2.]),"task_1":torch.tensor([3.,4.])},str(root/"prompt_bank_bf16.safetensors"))
            batch=HumanWAMDataset(root)[0]
            torch.testing.assert_close(batch["text_embeddings"],torch.tensor([3.,4.]))
            self.assertEqual(batch["actions"].shape,(10,45));self.assertNotIn("proprio",batch);self.assertNotIn("embodiment_id",batch)
            override={"action":{"mean":[1]*45,"std":[2]*45}}
            torch.testing.assert_close(HumanWAMDataset(root,episode_ids=[0],statistics=override)[0]["actions"],torch.full((10,45),-.5))
            with self.assertRaises(ValueError):HumanWAMDataset(root,episode_ids=[0,0])
            source=json.loads((dataset/"source.json").read_text());source["observation_state"]=HISTORY
            (dataset/"source.json").write_text(json.dumps(source))
            table=pq.read_table(data/"episode_000000.parquet")
            table=table.append_column("observation.state",pa.array(np.full((20,4,75),3.).tolist()))
            pq.write_table(table,data/"episode_000000.parquet")
            stats={"action":{"mean":[0]*45,"std":[1]*45},"observation.state":{"mean":[1]*75,"std":[2]*75}}
            (meta/"stats.json").write_text(json.dumps(stats))
            torch.testing.assert_close(HumanWAMDataset(root)[0]["proprio"],torch.ones((4,75)))
            source=json.loads((dataset/"source.json").read_text());source["label_provenance"]="robot_fk"
            (dataset/"source.json").write_text(json.dumps(source))
            with self.assertRaises(ValueError):HumanWAMDataset(root)


if __name__=="__main__":unittest.main()
