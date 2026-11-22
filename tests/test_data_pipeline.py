import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation

from wammimic.data_pipeline.geometry import OPENHLM_BODY,features,mimic_window
from wammimic.data_pipeline.openhlm import action_chunks,filtered_poses10,proprioceptive_states
from wammimic.data_pipeline.sim import import_sim


class GeometryTest(unittest.TestCase):
    def test_simulation_physics_success_cannot_bypass_visual_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);episode=root/'raw/episodes/seed_000000/episode.json';episode.parent.mkdir(parents=True)
            episode.write_text(json.dumps({'accepted':True,'physical_button_success':True}))
            with self.assertRaisesRegex(ValueError,'framing gate'):import_sim(root/'raw',root/'processed')

    def poses(self,n=20):
        p=np.zeros((n,5,7));p[...,3]=1
        p[:,0,2]=.68
        p[:,1:3,2]=.04
        p[:,3:5,0]=.4
        p[:,3:5,2]=.8
        return p

    def test_mixed_scale_and_rotation_columns(self):
        p=self.poses()
        x=features(p,p[0],.68)
        np.testing.assert_allclose(x[:,2],1)
        np.testing.assert_allclose(x[:,9],.4)
        np.testing.assert_allclose(x[0,15:21],[1,0,0,0,1,0])

    def test_translation_and_heading_invariance(self):
        p=self.poses();p[:,:,0]+=np.arange(len(p))[:,None]*.02
        rotation=Rotation.from_euler('z',1.2)
        changed=p.copy();changed[...,:3]=rotation.apply(p[...,:3].reshape(-1,3)).reshape(p[...,:3].shape)+[8,-3,0]
        q=rotation.as_quat()[[3,0,1,2]];changed[...,3:]=q
        np.testing.assert_allclose(features(p,p[0],.68),features(changed,changed[0],.68),atol=1e-6)

    def test_lower_body_uses_support_origin_but_wrists_use_pelvis(self):
        p=self.poses();p[:,1:3,0]=.2
        x=features(p,p[0],.68)
        np.testing.assert_allclose(x[:,0],-.2/.68)
        np.testing.assert_allclose(x[:,9],.4)

    def test_calibrated_floor_preserves_height_and_metric_wrist_offsets(self):
        p=self.poses();changed=p.copy();changed[...,2]+=.08
        np.testing.assert_allclose(features(p,p[0],.68),features(changed,changed[0],.68,floor_height=.08),atol=1e-6)
        np.testing.assert_allclose(mimic_window(p,2,.68),mimic_window(changed,2,.68,floor_height=.08),atol=1e-6)

    def test_tail_and_point_masks(self):
        p=self.poses()
        x=mimic_window(p,18,.68,[1,1,1,0,1])
        self.assertEqual(x.shape,(465,))
        np.testing.assert_array_equal(x[-10:],[1,1,0,0,0,0,0,0,0,0])
        b=x[:450].reshape(10,45)
        self.assertFalse(b[2:].any());self.assertFalse(b[:,9:12].any());self.assertFalse(b[:,33:39].any())

    def test_chunk_targets_share_observation_reference(self):
        p=self.poses();p[:,:,0]+=np.arange(len(p))[:,None]*.1
        chunks,pad=action_chunks(p,np.zeros((len(p),2)))
        np.testing.assert_allclose(chunks[5,:,0],np.arange(1,11)*.1/.68,rtol=1e-6)
        self.assertFalse(pad[5].any());self.assertTrue(pad[-1].all())

    def test_observation_anchor_is_independent_of_target_low_pass_filter(self):
        targets=self.poses();anchors=targets.copy();anchors[...,0]+=.1
        chunks,_=action_chunks(targets,np.zeros((len(targets),2)),anchors=anchors)
        np.testing.assert_allclose(chunks[0,:,0],-.1/.68)
        np.testing.assert_allclose(chunks[0,:,9],.3)

    def test_proprioception_does_not_use_future_interpolation_slopes(self):
        timestamps=np.arange(20)/30;states=np.zeros((20,34));commands=states.copy();poses=self.poses()
        states[:,16]=timestamps*.5
        expected=proprioceptive_states(states,commands,timestamps,poses)
        states[11:,16:28]+=1;commands[11:]+=1;poses[11:,0,3:]=Rotation.from_euler('x',.5).as_quat()[[3,0,1,2]]
        actual=proprioceptive_states(states,commands,timestamps,poses)
        np.testing.assert_allclose(actual[:11],expected[:11])
        np.testing.assert_allclose(expected[1:,29],.5)

    def test_openhlm_layout_is_not_legacy_body_first(self):
        values=np.arange(34)
        np.testing.assert_array_equal(values[OPENHLM_BODY][:6],np.arange(16,22))
        np.testing.assert_array_equal(values[OPENHLM_BODY][15:22],np.arange(7))
        self.assertNotIn(7,OPENHLM_BODY);self.assertNotIn(15,OPENHLM_BODY)

    def test_quaternion_signs_survive_downsampling(self):
        p=self.poses(100);p[1::2,:,3:]*=-1
        q=filtered_poses10(p)
        self.assertEqual(q.shape,(20,5,7))
        np.testing.assert_allclose(np.linalg.norm(q[...,3:],axis=-1),1,atol=1e-6)

    def test_reject_invalid_orientation(self):
        p=self.poses();p[0,0,3:]=0
        with self.assertRaises(ValueError):features(p,p[0],.68)


if __name__=='__main__':unittest.main()
