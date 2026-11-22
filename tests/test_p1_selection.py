import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.select_p1_robomocap import balanced
from scripts.run_p1_pipeline import wait_download
from wammimic.data_pipeline.common import sha256


class P1SelectionTest(unittest.TestCase):
    def test_processing_waits_for_expanded_download_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/'manifest.json';receipt=Path(folder)/'receipt.json'
            manifest.write_text(json.dumps({'siblings':[{'size':20}]}))
            receipt.write_text(json.dumps({'complete':True,'total_bytes':10,'manifest_sha256':sha256(manifest)}))
            with patch('scripts.run_p1_pipeline.time.sleep',side_effect=lambda _:receipt.write_text(json.dumps({'complete':True,'total_bytes':20,'manifest_sha256':sha256(manifest)}))) as wait:
                wait_download(manifest,receipt)
            wait.assert_called_once_with(30)

    def test_balance_people_before_choosing_more_short_sessions(self):
        rows=[{'participant':p,'primitive':f'p{i%2}','duration_s':10+i,
               'release_path':f'{p}/{i}'} for p in ['a','b','c'] for i in range(5)]
        chosen=balanced(rows,6)
        self.assertEqual({p:sum(r['participant']==p for r in chosen) for p in ['a','b','c']},dict(a=2,b=2,c=2))
        self.assertEqual(len({r['release_path'] for r in chosen}),6)
        self.assertEqual(len(rows),15)


if __name__=='__main__':unittest.main()
