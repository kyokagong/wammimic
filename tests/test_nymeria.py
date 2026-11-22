import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wammimic.data_pipeline.nymeria import annotation_labels, causal_indices, valid_blocks, verify_manifest, DATASET
from wammimic.data_pipeline.common import sha256
from scripts.run_p1_pipeline import wait_download
from wammimic.data_pipeline.datasets import NymeriaWAMDataset


class NymeriaTest(unittest.TestCase):
    def test_manifest_paths_cannot_escape_raw_root(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Unsafe source path'):
                verify_manifest(Path(folder),{'files':[{'path':'../outside','size':0,'sha1':'bad'}]})

    def test_wait_checks_nymeria_file_bytes_as_well_as_manifest_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/'manifest.json';receipt=Path(folder)/'receipt.json'
            manifest.write_text(json.dumps({'files':[{'path':'data','size':9}]}))
            value={'complete':True,'manifest_sha256':sha256(manifest),'total_bytes':8}
            receipt.write_text(json.dumps(value))
            def finish(_):
                value['total_bytes']=9;receipt.write_text(json.dumps(value))
            with patch('scripts.run_p1_pipeline.time.sleep',side_effect=finish) as sleep:
                wait_download(manifest,receipt)
                sleep.assert_called_once()

    def test_sampling_never_reads_future_or_stale_images(self):
        times=np.arange(20)/15
        query=np.array([-.01,.1,.2,2.])
        index,valid=causal_indices(times,query,1/15+.002)
        np.testing.assert_array_equal(index,[0,1,3,19])
        np.testing.assert_array_equal(valid,[False,True,True,False])
        self.assertTrue((times[index[valid]]<=query[valid]).all())
        with self.assertRaises(ValueError):causal_indices(np.array([1.,1.]),query,.1)

    def test_overlapping_narration_is_excluded_regardless_of_order(self):
        grid=np.arange(50)/10
        tasks={};segments,ids,count=annotation_labels(grid,[(0,3,'a'),(2,5,'b')],tasks)
        self.assertEqual(count,10)
        self.assertTrue((segments[20:30]==-1).all())
        self.assertTrue((ids[20:30]==-1).all())
        reverse=annotation_labels(grid,[(2,5,'b'),(0,3,'a')],{})[0]
        np.testing.assert_array_equal(segments<0,reverse<0)
        with self.assertRaises(ValueError):annotation_labels(grid,[(3,2,'a')],{})

    def test_windows_exclude_unsampled_bad_frames_and_clock_gaps(self):
        times=np.arange(1200)/240;grid=np.arange(45)/10+.1
        index,paired=causal_indices(times,grid,1/240+.001)
        quality=np.ones(len(times),bool);segments=np.zeros(len(grid),int)
        baseline=valid_blocks(grid,times,index,quality,paired,segments)
        self.assertTrue(baseline[10]);self.assertTrue(baseline[30])
        quality[index[10]+5]=False
        broken=valid_blocks(grid,times,index,quality,paired,segments)
        self.assertFalse(broken[10]);self.assertTrue(broken[30])
        quality[:]=True;times[index[10]+5:]+=.01
        broken=valid_blocks(grid,times,index,quality,paired,segments)
        self.assertFalse(broken[10]);self.assertTrue(broken[30])
        segments[15:]=1
        self.assertFalse(valid_blocks(grid,times,index,quality,paired,segments)[10])

    def test_reader_requires_explicit_inertial_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/DATASET/'source.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'schema':'nymeria-inertial-five-point-v1','label_provenance':'optical_mocap'}))
            with self.assertRaisesRegex(ValueError,'inertial provenance'):NymeriaWAMDataset(folder)
