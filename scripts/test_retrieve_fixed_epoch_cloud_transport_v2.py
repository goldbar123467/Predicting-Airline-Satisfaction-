"""Tiny fake GET/header/page fixtures. No network, model or competition reads."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import retrieve_fixed_epoch_cloud_transport_v2 as transport


class Response:
    url = "https://provider.invalid/get?signature=PRIVATE"
    def __init__(self, chunks=(b"abc",), length="3", encoding=None):
        self.chunks = chunks
        self.headers = {}
        if length is not None:
            self.headers["Content-Length"] = length
        if encoding is not None:
            self.headers["Content-Encoding"] = encoding
        self.iterated = False
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        self.iterated = True
        assert chunk_size == 1024 * 1024
        yield from self.chunks


class TransportTests(unittest.TestCase):
    def test_original_verified_helper_remains_pinned(self):
        transport.require_original()

    def test_fullpath_listing_requires_no_ambiguous_basename_size_join(self):
        calls = []
        def fetch(token):
            calls.append(token)
            names = ["fixed_epoch_return/cloud_status.json", "other/cloud_status.json"] if not token else [
                "fixed_epoch_return/output-manifest.json", "fixed_epoch_return/results.zip"]
            return SimpleNamespace(files=[SimpleNamespace(file_name=name, url=Response.url) for name in names],
                                   next_page_token="next" if not token else "")
        selected, pages = transport.base.selected_pages(fetch, field="file_name")
        self.assertEqual(set(selected), transport.base.SELECTED)
        self.assertEqual(calls, ["", "next"])
        self.assertNotIn("PRIVATE", json.dumps(pages))

    def test_same_get_length_and_hash_are_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            response = Response((b"a", b"bc"))
            def get(url, **kwargs):
                self.assertEqual(kwargs["headers"], {"Accept-Encoding": "identity"})
                self.assertTrue(kwargs["stream"])
                return response
            result = transport.stream_http_length(Response.url, Path(temporary) / "results.zip", remaining_bytes=10, get=get)
            self.assertEqual(result["bytes"], 3)
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_missing_invalid_or_excess_length_rejects_before_file_allocation(self):
        for length in (None, "-1", "3.0", "", "99999999999999"):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as temporary:
                response = Response(length=length)
                path = Path(temporary) / "results.zip"
                with self.assertRaises(RuntimeError) as caught:
                    transport.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: response)
                self.assertFalse(response.iterated)
                self.assertFalse(path.exists())
                self.assertNotIn("PRIVATE", str(caught.exception))

    def test_short_long_and_encoded_body_fail_without_overwrite(self):
        for response in (Response((b"a",)), Response((b"abcd",)), Response(encoding="gzip")):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "results.zip"
                with self.assertRaises(RuntimeError):
                    transport.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: response)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.zip"
            path.write_bytes(b"original")
            with self.assertRaises(RuntimeError):
                transport.stream_http_length(Response.url, path, remaining_bytes=10, get=lambda *a, **k: Response())
            self.assertEqual(path.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
