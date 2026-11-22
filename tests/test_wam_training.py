import unittest
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from wammimic.training.evaluation import pose_errors,rotation6d,world_actions,temporal_errors
from wammimic.training.model import CosmosWAM,LinearBaseline,action_loss
from wammimic.training.data import build_protocol
from wammimic.data_pipeline.common import sha256,write_json
from wammimic.data_pipeline.human_ego import DATASET,SCHEMA,PROVENANCE,human_features


class WAMTrainingTest(unittest.TestCase):
    def poses(self):
        value=np.zeros((10,45));value[:,15:]=np.tile([1,0,0,0,1,0],5)
        return value

    def test_metric_restores_leg_scale_and_does_not_recenter_future(self):
        truth=self.poses();prediction=truth.copy();prediction[:,0]=1;prediction[:,9]=1
        errors=pose_errors(prediction,truth,.8)
        np.testing.assert_allclose(errors['position_mm'][:,0],800)
        np.testing.assert_allclose(errors['position_mm'][:,3],1000)
        np.testing.assert_allclose(errors['rotation_deg'],0)

    def test_rotations_use_geodesic_angle_and_penalize_degenerate_output(self):
        truth=self.poses();prediction=truth.copy();prediction[:,15:21]=[0,1,0,-1,0,0]
        np.testing.assert_allclose(pose_errors(prediction,truth,1.)['rotation_deg'][:,0],90)
        prediction[:,15:21]=0;errors=pose_errors(prediction,truth,1.)
        self.assertTrue(errors['invalid_rotation'][:,0].all())
        np.testing.assert_allclose(errors['rotation_deg'][:,0],180)
        prediction[0,0]=np.nan
        with self.assertRaises(ValueError):pose_errors(prediction,truth,1.)

    def test_world_conversion_uses_one_fixed_anchor_for_all_future_frames(self):
        from scipy.spatial.transform import Rotation
        anchor=np.zeros((5,7));q=Rotation.from_euler('zyx',[.5,.2,-.3]).as_quat()[[3,0,1,2]]
        anchor[:,:3]=[2,-3,1];anchor[:,3:]=q
        future=np.repeat(anchor[None],10,axis=0);future[...,:3]+=np.arange(10)[:,None,None]*np.array([.1,.05,-.02])
        action=human_features(future,anchor,.8);world,_,valid=world_actions(action,anchor,.8)
        np.testing.assert_allclose(world,future[...,:3],atol=1e-6);self.assertTrue(valid.all())

    def test_temporal_diagnostics_use_physical_units_and_real_time_intervals(self):
        truth=self.poses();prediction=truth.copy();times=np.arange(1,11)/10
        prediction[:,:15].reshape(10,5,3)[...,0]=times[:,None]
        result=temporal_errors(prediction,truth,1.,times)
        self.assertAlmostEqual(result['velocity_rmse_m_s'],1.)
        self.assertLess(result['acceleration_rmse_m_s2'],1e-10);self.assertLess(result['jerk_rmse_m_s3'],1e-9)
        self.assertAlmostEqual(result['angular_speed_rmse_rad_s'],0.)
        with self.assertRaises(ValueError):temporal_errors(prediction,truth,1.,np.zeros(10))

    def test_whole_recording_splits_refit_statistics_and_reject_duplicate_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);roots=[base/'base',base/'operation'];raws=[base/'raw0',base/'raw1'];assignments=[]
            for source,root in enumerate(roots):
                dataset=root/DATASET;write_json(dataset/'source.json',{'schema':SCHEMA,'label_provenance':PROVENANCE})
                (root/'prompt_bank_bf16.safetensors').write_bytes(f'encoded text {source}'.encode())
                write_json(root/'qa_reports/cache_video.json',{'complete':True,'fixture':source})
                write_json(root/'qa_reports/human_wam_training_ready.json',{'ready':True,
                    'prompt_bank_sha256':sha256(root/'prompt_bank_bf16.safetensors'),
                    'video_cache_report_sha256':sha256(root/'qa_reports/cache_video.json')});rows=[]
                splits=['train','val','test'] if source==0 else ['val','test']
                for episode,split in enumerate(splits):
                    release=f'actions/task{source}/{episode}';raw=raws[source]/release
                    write_json(raw/'raw/manifest.json',{'upload_batch_id':'20261118','mocap_capture':{'source_directory':f'capture{source}_{episode}'}})
                    (raw/'raw/mocap').mkdir();(raw/'raw/mocap/Body0.bvh').write_text(f'vendor source {source} {episode}')
                    parquet=dataset/f'data/chunk-000/episode_{episode:06d}.parquet';parquet.parent.mkdir(parents=True,exist_ok=True)
                    value=2 if split=='train' else 1000
                    pq.write_table(pa.table({'action':np.full((2,10,45),value).tolist(),'valid_for_training':[True,True]}),parquet)
                    video=dataset/f'videos/{episode}.mp4';video.parent.mkdir(exist_ok=True);video.write_bytes(f'camera {source} {episode}'.encode())
                    row={'episode_index':episode,'source_session':release,'participant':'actor','family':'static_manipulation' if source==1 or split=='train' else 'locomotion_primitives',
                        'primitive':'SM1','length':32,'valid_samples':2,'human_leg_length':.8,'split':split,
                        'outputs':{str(parquet.relative_to(root)):sha256(parquet),str(video.relative_to(root)):sha256(video)}}
                    rows.append(row)
                    if source==1:assignments.append({'release_path':release,'evaluation_split':split})
                (dataset/'meta').mkdir();(dataset/'meta/episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest=base/'manifest.json';write_json(manifest,{'sessions':assignments})
            report=build_protocol(*roots,manifest,base/'protocol',raws)
            np.testing.assert_allclose(report['statistics']['action']['mean'],2)
            self.assertEqual(report['counts']['train']['episodes'],1)
            for root in roots:
                dataset=root/DATASET;source=json.loads((dataset/'source.json').read_text());source['observation_state']={'shape':[4,75]};write_json(dataset/'source.json',source)
                rows=list(map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines()))
                for row in rows:
                    parquet=dataset/f'data/chunk-000/episode_{row["episode_index"]:06d}.parquet';table=pq.read_table(parquet)
                    table=table.append_column('observation.state',pa.array(np.full((2,4,75),3 if row['split']=='train' else 3000).tolist()));pq.write_table(table,parquet)
                    row['outputs'][str(parquet.relative_to(root))]=sha256(parquet)
                (dataset/'meta/episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            history_report=build_protocol(*roots,manifest,base/'history-protocol',raws)
            np.testing.assert_allclose(history_report['statistics']['observation.state']['mean'],3)
            changed=raws[1]/assignments[0]['release_path']/'raw/manifest.json'
            write_json(changed,{'upload_batch_id':'20261118','mocap_capture':{'source_directory':'capture0_0'}})
            with self.assertRaisesRegex(ValueError,'leakage'):build_protocol(*roots,manifest,base/'protocol2',raws)

    @unittest.skipUnless(importlib.util.find_spec('diffusers'),'training image includes Diffusers')
    def test_real_cosmos_modules_preserve_last_block_gradients_and_exclude_future(self):
        from diffusers import CosmosTransformer3DModel
        torch.manual_seed(7)
        backbone=CosmosTransformer3DModel(in_channels=17,out_channels=16,num_attention_heads=2,
            attention_head_dim=32,num_layers=2,text_embed_dim=32,adaln_lora_dim=16,
            max_size=(8,8,8),patch_size=(1,2,2),extra_pos_embed_type=None)
        model=CosmosWAM(backbone,width=64,layers=1).eval();video=torch.randn(1,16,5,4,4);text=torch.randn(1,8,32)
        context=model.encode_history(video,text);changed=video.clone();changed[:,:,2:]=1e4
        torch.testing.assert_close(context,model.encode_history(changed,text),rtol=0,atol=0)
        noise=torch.randn(1,10,45);time=torch.tensor([.5]);prediction=model.action_velocity(context,noise,time)
        loss,_=action_loss(prediction,torch.randn_like(prediction));loss.backward()
        grads=[p.grad for p in backbone.transformer_blocks[-1].parameters() if p.grad is not None]
        self.assertTrue(grads and any(g.abs().sum()>0 for g in grads));self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(p.grad is None for p in backbone.transformer_blocks[0].parameters()))
        model.zero_grad();velocity=model.video_velocity(video,text,torch.tensor([.5]));self.assertEqual(velocity.shape,video.shape)
        velocity[:,:,2:].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in backbone.transformer_blocks[-1].parameters()))
        baseline=LinearBaseline(text_dim=32).eval()
        torch.testing.assert_close(baseline(video,text),baseline(changed,text),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
