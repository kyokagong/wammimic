import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch

import numpy as np
import torch

from scripts.download_cache_models import verify,dataset_records,download_ranges
from scripts.run_phase0 import download_verified,file_hash,available_gpu
from wammimic.data_pipeline.cache import clip_frame_indices,concatenate_reason_layers,prompt_ids,check_causal_prefix,cache_owner
from wammimic.data_pipeline.datasets import MimicDataset
from wammimic.data_pipeline.validate import validate_mimic
from wammimic.data_pipeline.common import sha256
from wammimic.data_pipeline.morph import reference_floors


class CacheTest(unittest.TestCase):
    def test_gpu_cache_waits_when_no_card_has_enough_free_memory(self):
        with patch("scripts.run_phase0.subprocess.check_output",return_value="0, 17000\n1, 22000\n2, 23000\n"):
            self.assertEqual(available_gpu(20000),2)
            self.assertIsNone(available_gpu(24000))

    def test_causal_alignment_has_no_future_in_first_five_pixels(self):
        indices=clip_frame_indices(10)
        np.testing.assert_array_equal(indices[:5],[7,7,8,9,10])
        self.assertEqual(len(indices),17)
        self.assertEqual(indices[-1],22)
        self.assertTrue((indices[:5]<=10).all())
        with self.assertRaises(ValueError):clip_frame_indices(2)

    def test_reason_layers_skip_token_embeddings_and_keep_layer_order(self):
        embeddings=torch.full((1,2,3),99.)
        a=torch.tensor([[[1.,2.,3.],[3.,2.,1.]]])
        b=torch.tensor([[[3.,1.,2.],[2.,1.,3.]]])
        out=concatenate_reason_layers((embeddings,a,b))
        self.assertEqual(out.shape,(1,2,6))
        torch.testing.assert_close(out[0,0],torch.tensor([-1.,0.,1.,1.,-1.,0.]))

    def test_chat_encoding_uses_input_ids_not_batch_encoding_keys(self):
        class Tokenizer:
            pad_token_id=0
            def apply_chat_template(self,*args,**kwargs):
                self.kwargs=kwargs
                return {"input_ids":[10,20,30],"attention_mask":[1,1,1]}
        tokenizer=Tokenizer();ids=prompt_ids(tokenizer,"walk")
        self.assertEqual(ids[:4],[10,20,30,0]);self.assertEqual(len(ids),512)
        self.assertTrue(tokenizer.kwargs["return_dict"])
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])

    def test_future_leak_is_detected(self):
        class Leaky:
            def encode(self,x):return x.mean((1,2,3,4))[:,None,None,None,None].expand(-1,16,5,2,2)
        with self.assertRaises(ValueError):check_causal_prefix(Leaky(),torch.ones(1,3,17,2,2))

    def test_model_hash_and_size_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"config";p.write_bytes(b"{}")
            record={"size":2,"sha256":hashlib.sha256(b"{}").hexdigest()}
            self.assertTrue(verify(p,record));p.write_bytes(b"[]")
            self.assertFalse(verify(p,record))
            record={"size":2,"blob_sha1":hashlib.sha1(b"blob 2\0[]").hexdigest()}
            self.assertTrue(verify(p,record))

    def test_cache_resume_preserves_unowned_or_mismatched_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/"source.json";bank=root/"bank"
            bank.touch()
            with self.assertRaises(FileExistsError):cache_owner(marker,{"recipe":"a"},[bank])
            bank.unlink();cache_owner(marker,{"recipe":"a"},[bank])
            cache_owner(marker,{"recipe":"a"},[bank])
            with self.assertRaises(ValueError):cache_owner(marker,{"recipe":"b"},[bank])

    def test_dataset_download_keeps_pinned_revision_hash_and_budget(self):
        root=Path("/tmp/selected-test")
        source={"id":"owner/data","sha":"pinned","siblings":[{"rfilename":"a/b","size":3,"blobId":"blob","lfs":{"sha256":"digest"}}]}
        budget={"current_subset_plans":[{"root":str(root),"cap_bytes":10}]}
        plan=dataset_records(source,budget,root)
        self.assertEqual(plan["total_bytes"],3);self.assertEqual(plan["cap_bytes"],10)
        self.assertEqual(plan["files"][0]["revision"],"pinned")
        self.assertEqual(plan["files"][0]["sha256"],"digest")

    def test_download_completion_is_bound_to_the_selected_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/"manifest.json";manifest.write_text("first")
            self.assertFalse(download_verified(root,manifest))
            receipt=root/"wammimic_100g_verification.json"
            receipt.write_text(json.dumps({"complete":True,"manifest_sha256":file_hash(manifest)}))
            self.assertTrue(download_verified(root,manifest));manifest.write_text("changed")
            self.assertFalse(download_verified(root,manifest))

    def test_floor_calibration_must_match_the_verified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/"manifest.json"
            document={"clips":[{"robot":"g1","name":"clip","output":"original.npz","output_sha256":"expected"}]}
            manifest.write_text(json.dumps(document))
            row={"robot":"g1","name":"clip","path":"original.npz","sha256":"expected","offset_m":.08,"valid":True}
            receipt={"prepared_manifest_sha256":sha256(manifest),"clips":[row]}
            path=root/"contact-floor.json";path.write_text(json.dumps(receipt))
            floors,_=reference_floors(manifest,document);self.assertEqual(floors[("g1","clip")]["offset_m"],.08)
            row["valid"]=False;path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):reference_floors(manifest,document)
            row["valid"]=True;row["sha256"]="changed";path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):reference_floors(manifest,document)

    def test_bounded_ranges_resume_a_prefix_and_verify_final_file(self):
        payload=b"0123456789abcdefghijklmnopqrstuvwxyz"
        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                start,end=map(int,self.headers["Range"].removeprefix("bytes=").split("-"))
                data=payload[start:end+1]
                self.send_response(200 if len(data)==len(payload) else 206)
                if len(data)!=len(payload):self.send_header("Content-Range",f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(("127.0.0.1",0),Server)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory,patch("scripts.download_cache_models.RANGE_BYTES",8):
                target=Path(directory)/"model";prefix=target.with_name("model.download.part");prefix.write_bytes(payload[:3])
                record={"size":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}
                download_ranges(f"http://127.0.0.1:{server.server_port}/model",target,record)
                self.assertTrue(verify(target,record));self.assertFalse(prefix.exists())
                self.assertFalse(target.with_name("model.ranges").exists())
                small=Path(directory)/"config"
                git_record={"size":len(payload),"blob_sha1":hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()}
                download_ranges(f"http://127.0.0.1:{server.server_port}/config",small,git_record)
                self.assertTrue(verify(small,git_record))
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_mimic_reader_does_not_cross_clip_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for clip in range(2):
                folder=root/"g1"/str(clip);folder.mkdir(parents=True)
                poses=np.zeros((10,5,7),dtype=np.float32);poses[...,3]=1;poses[:,0,2]=.68
                np.save(folder/"poses.npy",poses)
                joints=np.zeros((10,2,1),dtype=np.float32);joints[:,0,0]=clip
                np.save(folder/"joints.npy",joints)
                rows.append({"frames":10,"path":"g1/"+str(clip),"leg_length":.68,"split":"train","joint_names":["j"],
                             "reference_floor_offset_m":.08 if clip==0 else 0.,"reference_floor_valid":clip==0})
            (root/"g1/manifest.json").write_text(json.dumps({"clips":rows}))
            reader=MimicDataset(root,"g1")
            self.assertEqual(len(reader),20)
            self.assertEqual(reader[9]["five_point_window"][-10:].sum(),1)
            self.assertEqual(reader[10]["five_point_window"][-10:].sum(),10)
            self.assertEqual(reader[10]["joint_pos"].item(),1)
            self.assertAlmostEqual(reader[0]["root_pose"][2].item(),.6,places=6)
            self.assertAlmostEqual(reader[0]["five_point_window"][2].item(),.6/.68,places=6)
            self.assertTrue(reader[0]["reference_floor_valid"]);self.assertFalse(reader[10]["reference_floor_valid"])
            with self.assertRaises(IndexError):reader[20]

    def test_shared_robot_training_rejects_cross_robot_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"morph_5pt_store"
            for robot in ["q1","bumi","g1"]:
                folder=root/robot/"clip";folder.mkdir(parents=True)
                poses=np.zeros((10,5,7),dtype=np.float32);poses[...,3]=1
                np.save(folder/"poses.npy",poses);np.save(folder/"joints.npy",np.zeros((10,2,1),dtype=np.float32))
                row={"frames":10,"path":robot+"/clip","leg_length":.68,"split":"val" if robot=="g1" else "train",
                     "joint_names":["j"],"group_id":"shared-source","reference_floor_offset_m":0.,"reference_floor_valid":True,
                     "output_sha256":{f:sha256(folder/f) for f in ["poses.npy","joints.npy"]}}
                (root/robot/"manifest.json").write_text(json.dumps({"clips":[row]}))
            with self.assertRaisesRegex(ValueError,"split leakage"):validate_mimic(directory)


if __name__=="__main__":unittest.main()
