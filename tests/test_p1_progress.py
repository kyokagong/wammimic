import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_p1_progress import audit,file_size


class P1ProgressTest(unittest.TestCase):
    def test_selected_files_and_completed_downloads_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);control=root/'control';control.mkdir()
            (control/'budget_100g.json').write_text(json.dumps({'limit_bytes':100,'projected_reserved_bytes':90}))
            (control/'robomocap-p1.json').write_text(json.dumps({'siblings':[{'rfilename':'video','size':20},{'rfilename':'empty','size':0}]}))
            report=audit(root)
            self.assertEqual(report['downloads']['robomocap']['selected_files'],2)
            self.assertEqual(report['downloads']['robomocap']['size_complete_files'],0)
            self.assertFalse(report['downloads']['robomocap']['all_hashes_verified'])
            self.assertFalse(report['p1_training_ready'])
            ready=root/'processed/nymeria/qa_reports/nymeria_wam_training_ready.json'
            ready.parent.mkdir(parents=True);ready.write_text(json.dumps({'ready':True,'cached_samples':42}))
            report=audit(root)
            self.assertTrue(report['nymeria_training_data']['ready'])
            self.assertFalse(report['p1_training_ready'])
            fragment=root/'fragment';fragment.write_bytes(b'part');present=fragment.stat()
            with patch.object(Path,'stat',side_effect=[present,FileNotFoundError()]):
                self.assertIsNone(file_size(fragment))


if __name__=='__main__':unittest.main()
