import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_cache_models as download


class DownloadAuthTest(unittest.TestCase):
    def test_nymeria_standard_sha1_is_distinct_from_git_blob_sha1(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recording';path.write_bytes(b'abcd')
            self.assertTrue(download.verify(path,{'size':4,'sha1':hashlib.sha1(b'abcd').hexdigest()}))
            self.assertFalse(download.verify(path,{'size':4,'blob_sha1':hashlib.sha1(b'abcd').hexdigest()}))

    def test_authorized_bounded_download_supports_older_http_reader_without_saving_token(self):
        class Reader:
            def __init__(self):self.data=io.BytesIO(b'abcd')
            def read(self,size,decode_content=True):return self.data.read(size)
        class Response:
            status_code=206
            headers={'Content-Range':'bytes 0-3/4'}
            raw=Reader()
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def raise_for_status(self):pass
        with tempfile.TemporaryDirectory() as folder,patch.object(download.requests.Session,'get',return_value=Response()) as request:
            path=Path(folder)/'weights';record={'size':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}
            download.download_ranges('https://huggingface.co/example/weights',path,record,{'Authorization':'Bearer test-only'})
            self.assertEqual(path.read_bytes(),b'abcd')
            self.assertEqual(request.call_args.kwargs['headers']['Range'],'bytes=0-3')
            self.assertEqual(request.call_args.kwargs['headers']['Authorization'],'Bearer test-only')
            self.assertEqual(list(Path(folder).iterdir()),[path])


if __name__=='__main__':unittest.main()
