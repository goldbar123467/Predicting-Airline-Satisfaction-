"""Reserved production-window dispatch tests; no real API calls or model work."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import kaggle_private_job as dispatch


class FixedDateTime(datetime):
    fixed = datetime(2026, 10, 2, 16, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed if tz is None else cls.fixed.astimezone(tz)


class ProductionDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="production-dispatch-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(dispatch, "ROOT", self.root))
        self.stack.enter_context(patch.object(dispatch, "datetime", FixedDateTime))
        self.stack.enter_context(patch.object(FixedDateTime, "fixed", datetime(2026, 10, 2, 16, tzinfo=timezone.utc)))
        self.folder = self.root / "cloud/reviewed-production"
        (self.folder / "payload").mkdir(parents=True)
        (self.folder / "job.py").write_text("# SYNTHETIC: no execution or imports\n", encoding="utf-8")
        self.meta = {"id": "clarkkitchen/s6e10-cloud-production-synthetic", "is_private": True,
                     "code_file": "job.py", "enable_gpu": True, "machine_shape": "NvidiaTeslaP100",
                     "competition_sources": ["playground-series-s6e10"]}
        self.freeze_path = self.root / "artifacts/third_pass_batch03/blend/frozen.json"
        self.freeze = {"weights": {"v3_cloud_selected": .1},
                       "source_result_hashes": {"v3_cloud_selected": "a" * 64}}
        self.spec = {"format": "third_pass_cloud_refit", "run_id": "v3_cloud_selected",
                     "cv_result_sha256": "a" * 64}
        self.prepared = {"freeze_file": str(self.freeze_path),
                         "runtime_sha256": self.sha(self.folder / "job.py")}
        self.bind_freeze()
        response = SimpleNamespace(error=None, version_number=1,
                                   to_dict=lambda **kwargs: {"versionNumber": 1, "error": None})
        quota = SimpleNamespace(total_time_allowed=timedelta(hours=2), time_used=timedelta(),
                                time_reserved=timedelta())
        self.api = SimpleNamespace(kernels_push=Mock(return_value=response),
                                   quota_view=Mock(return_value=SimpleNamespace(gpu_quota=quota)))
        self.stack.enter_context(patch.object(dispatch, "api_client", return_value=self.api))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def bind_freeze(self):
        self.write(self.freeze_path, self.freeze)
        digest = self.sha(self.freeze_path)
        self.spec["selection_frozen_sha256"] = digest
        self.prepared.update(freeze_file=str(self.freeze_path), frozen_selection_hash=digest)
        self.save()

    def save(self):
        for path, value in [("kernel-metadata.json", self.meta), ("preparation_manifest.json", self.prepared),
                            ("payload/refit_spec.json", self.spec)]:
            self.write(self.folder / path, value)

    def invoke(self, production=True, timeout=1500):
        args = ["kaggle_private_job.py", "push", str(self.folder), "--timeout", str(timeout)]
        if production:
            args.append("--production")
        with patch.object(sys, "argv", args):
            dispatch.main()

    def configure_lgb(self, production=False):
        self.protocol = {'protocol_id': 'lgb_rating_probability_v1',
                         'candidate_id': 'v3_cloud_lgb_rating_probability',
                         'execution_image': 'gcr.io/synthetic/cpu@sha256:' + 'b' * 64}
        self.protocol_path = self.root / 'configs/third_pass_lgb_cloud.json'
        self.write(self.protocol_path, self.protocol)
        self.protocol_hash = self.sha(self.protocol_path)
        self.stack.enter_context(patch.object(dispatch, 'LGB_PROTOCOL_SHA', self.protocol_hash))
        self.meta.update(enable_gpu=False, docker_image=self.protocol['execution_image'])
        self.meta.pop('machine_shape')
        self.prepared['protocol_sha256'] = self.protocol_hash
        if production:
            name = self.protocol['candidate_id']
            self.freeze = {'weights': {name: .1}, 'source_result_hashes': {name: 'a' * 64}}
            self.spec.update(run_id=name, protocol_id=self.protocol['protocol_id'],
                             protocol_sha256=self.protocol_hash, timeout_seconds=600,
                             execution_image=self.protocol['execution_image'])
            self.bind_freeze()
        else:
            self.folder = self.root / 'cloud/third_pass_lgb'
            self.folder.mkdir(parents=True)
            (self.folder / 'job.py').write_text('# SYNTHETIC CPU CV\n', encoding='utf-8')
            self.meta.update(id='clarkkitchen/s6e10-cloud-lgb-synthetic', competition_sources=[])
            self.prepared['runtime_sha256'] = self.sha(self.folder / 'job.py')
            self.save()

    def assert_no_dispatch(self):
        self.api.kernels_push.assert_not_called()
        self.api.quota_view.assert_not_called()
        self.assertFalse((self.folder / 'push_intent.json').exists())

    def test_cpu_cv_latest_start_1510_and_end_1530_without_gpu_quota(self):
        self.configure_lgb()
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 1200, False),
                         datetime(2026, 10, 2, 15, 30, tzinfo=timezone.utc))
        with patch.object(FixedDateTime, 'fixed', datetime(2026, 10, 2, 15, 10, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, 'cutoff'):
                self.invoke(production=False, timeout=1200)
        self.assert_no_dispatch()
        with patch.object(FixedDateTime, 'fixed', datetime(2026, 10, 2, 15, 10, tzinfo=timezone.utc)):
            self.invoke(production=False, timeout=1200)
        self.api.kernels_push.assert_called_once_with(str(self.folder), timeout='1200', acc=None)
        self.api.quota_view.assert_not_called()
        intent = json.loads((self.folder / 'push_intent.json').read_text())
        self.assertEqual(intent['timeout_seconds'], 1200)

    def test_cpu_cv_rejects_wrong_timeout_protocol_preparation_image_gpu_and_competition(self):
        self.configure_lgb()
        for timeout in [600, 1199, 1201, 1500]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'CPU CV'):
                dispatch.dispatch_deadline(self.folder, self.meta, timeout, False)
        for mapping, key, value in [
                (self.prepared, 'protocol_sha256', 'wrong'), (self.prepared, 'runtime_sha256', 'wrong'),
                (self.meta, 'docker_image', 'different-image'), (self.meta, 'enable_gpu', True),
                (self.meta, 'enable_gpu', None), (self.meta, 'competition_sources', ['playground-series-s6e10'])]:
            original = copy.deepcopy(mapping[key])
            mapping[key] = value
            self.save()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'CPU CV'):
                self.invoke(production=False, timeout=1200)
            mapping[key] = original
        self.assert_no_dispatch()

    def test_cpu_cv_detects_protocol_and_executed_source_byte_changes(self):
        self.configure_lgb()
        original = self.protocol_path.read_bytes()
        self.protocol_path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            self.invoke(production=False, timeout=1200)
        self.protocol_path.write_bytes(original)
        (self.folder / 'job.py').write_text('# different source')
        with self.assertRaisesRegex(ValueError, 'CPU CV'):
            self.invoke(production=False, timeout=1200)
        self.assert_no_dispatch()

    def test_cpu_production_latest_start_1550_and_end_1600_without_gpu_quota(self):
        self.configure_lgb(production=True)
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 600, True),
                         datetime(2026, 10, 2, 16, tzinfo=timezone.utc))
        with patch.object(FixedDateTime, 'fixed', datetime(2026, 10, 2, 15, 50, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, 'cutoff'):
                self.invoke(timeout=600)
        self.assert_no_dispatch()
        with patch.object(FixedDateTime, 'fixed', datetime(2026, 10, 2, 15, 50, tzinfo=timezone.utc)):
            self.invoke(timeout=600)
        self.api.kernels_push.assert_called_once_with(str(self.folder), timeout='600', acc=None)
        self.api.quota_view.assert_not_called()

    def test_cpu_production_binds_protocol_image_gpu_timeout_and_preparation(self):
        self.configure_lgb(production=True)
        for timeout in [599, 601, 1200, 1500]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'contract differs'):
                dispatch.dispatch_deadline(self.folder, self.meta, timeout, True)
        for mapping, key, value in [
                (self.spec, 'protocol_sha256', 'wrong'), (self.prepared, 'protocol_sha256', 'wrong'),
                (self.spec, 'execution_image', 'wrong-image'), (self.meta, 'docker_image', 'wrong-image'),
                (self.meta, 'enable_gpu', True), (self.meta, 'enable_gpu', None),
                (self.spec, 'timeout_seconds', 1500), (self.prepared, 'runtime_sha256', 'wrong'),
                (self.meta, 'competition_sources', []), (self.spec, 'cv_result_sha256', 'wrong')]:
            original = copy.deepcopy(mapping[key])
            mapping[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract differs'):
                self.invoke(timeout=600)
            mapping[key] = original
        self.assert_no_dispatch()

    def test_cpu_production_requires_exact_positive_registered_candidate_and_freeze(self):
        self.configure_lgb(production=True)
        name = self.protocol['candidate_id']
        for weight in [0, -.1, float('nan'), float('inf')]:
            self.freeze['weights'][name] = weight
            self.bind_freeze()
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'contract differs'):
                dispatch.dispatch_deadline(self.folder, self.meta, 600, True)
        self.spec['run_id'] = 'v3_different_positive_candidate'
        self.freeze = {'weights': {self.spec['run_id']: .2},
                       'source_result_hashes': {self.spec['run_id']: self.spec['cv_result_sha256']}}
        self.bind_freeze()
        with self.assertRaisesRegex(ValueError, 'CPU production'):
            dispatch.dispatch_deadline(self.folder, self.meta, 600, True)
        self.spec['run_id'] = name
        self.freeze = {'weights': {name: .1}, 'source_result_hashes': {name: self.spec['cv_result_sha256']}}
        self.bind_freeze()
        for mapping, key in [(self.spec, 'selection_frozen_sha256'), (self.prepared, 'frozen_selection_hash')]:
            original = mapping[key]
            mapping[key] = 'wrong'
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract differs'):
                dispatch.dispatch_deadline(self.folder, self.meta, 600, True)
            mapping[key] = original
        self.save()
        self.protocol_path.write_bytes(self.protocol_path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            dispatch.dispatch_deadline(self.folder, self.meta, 600, True)
        self.assert_no_dispatch()

    def test_cpu_production_cannot_downgrade_to_generic_gpu_protocol(self):
        self.configure_lgb(production=True)
        for protocol in [None, 'unregistered_gpu_protocol']:
            if protocol is None:
                self.spec.pop('protocol_id', None)
            else:
                self.spec['protocol_id'] = protocol
            self.save()
            for timeout in [600, 1500]:
                with self.subTest(protocol=protocol, timeout=timeout), \
                        self.assertRaisesRegex(ValueError, 'contract differs'):
                    dispatch.dispatch_deadline(self.folder, self.meta, timeout, True)
        self.assert_no_dispatch()

    def test_cpu_production_folder_identity_prevents_candidate_rename_downgrade(self):
        self.configure_lgb(production=True)
        self.folder = self.root / 'cloud/third_pass_lgb/production-fixture'
        self.folder.mkdir(parents=True)
        (self.folder / 'job.py').write_text('# SYNTHETIC renamed protocol attempt')
        self.prepared['runtime_sha256'] = self.sha(self.folder / 'job.py')
        self.spec.pop('protocol_id')
        self.spec['run_id'] = 'v3_renamed_cloud_candidate'
        self.freeze = {'weights': {self.spec['run_id']: .2},
                       'source_result_hashes': {self.spec['run_id']: self.spec['cv_result_sha256']}}
        self.bind_freeze()
        for timeout in [600, 1500]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, 'contract differs'):
                dispatch.dispatch_deadline(self.folder, self.meta, timeout, True)
        self.assert_no_dispatch()

    def test_valid_production_dispatch_uses_exact_timeout_and_durable_intent(self):
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 1500, True),
                         datetime(2026, 10, 2, 16, 35, tzinfo=timezone.utc))

        def check_push(folder, **kwargs):
            intent = json.loads((Path(folder) / "push_intent.json").read_text())
            self.assertEqual(intent["timeout_seconds"], 1500)
            self.assertEqual(intent["source_sha256"], self.sha(self.folder / "job.py"))
            self.assertIs(intent["metadata"]["is_private"], True)
            self.assertEqual(kwargs["timeout"], "1500")
            return SimpleNamespace(error=None, version_number=1,
                                   to_dict=lambda **kwargs: {"versionNumber": 1})

        self.api.kernels_push.side_effect = check_push
        self.invoke()
        self.api.kernels_push.assert_called_once()
        self.assertTrue((self.folder / "push_receipt.json").exists())

    def test_without_flag_original_cv_cutoff_remains_1600(self):
        self.assertEqual(dispatch.dispatch_deadline(self.folder, self.meta, 1500, False),
                         datetime(2026, 10, 2, 16, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.invoke(production=False)
        self.api.kernels_push.assert_not_called()
        self.assertFalse((self.folder / "push_intent.json").exists())

    def test_exact_latest_start_allowed_but_one_second_later_rejected(self):
        with patch.object(FixedDateTime, "fixed", datetime(2026, 10, 2, 16, 10, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "cutoff"):
                self.invoke()
        self.api.kernels_push.assert_not_called()
        self.assertFalse((self.folder / "push_intent.json").exists())
        with patch.object(FixedDateTime, "fixed", datetime(2026, 10, 2, 16, 10, tzinfo=timezone.utc)):
            self.invoke()
        self.api.kernels_push.assert_called_once()

    def test_production_requires_exactly_1500_seconds_and_positive_finite_weight(self):
        for timeout in [60, 900, 1499, 1501, 3600]:
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "contract differs"):
                dispatch.dispatch_deadline(self.folder, self.meta, timeout, True)
        for weight in [0, -.1, float("nan"), float("inf")]:
            self.freeze["weights"]["v3_cloud_selected"] = weight
            self.bind_freeze()
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "contract differs"):
                dispatch.dispatch_deadline(self.folder, self.meta, 1500, True)
        self.freeze["weights"] = {}
        self.bind_freeze()
        with self.assertRaisesRegex(ValueError, "contract differs"):
            dispatch.dispatch_deadline(self.folder, self.meta, 1500, True)

    def test_both_frozen_hash_bindings_and_source_result_hash_are_required(self):
        for mapping, key in [(self.spec, "selection_frozen_sha256"),
                             (self.prepared, "frozen_selection_hash"), (self.spec, "cv_result_sha256")]:
            original = mapping[key]
            mapping[key] = "0" * 64
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract differs"):
                dispatch.dispatch_deadline(self.folder, self.meta, 1500, True)
            mapping[key] = original
        self.save()
        self.freeze_path.write_bytes(self.freeze_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "contract differs"):
            dispatch.dispatch_deadline(self.folder, self.meta, 1500, True)

    def test_only_owned_production_slug_and_third_pass_freeze_namespace_are_accepted(self):
        invalid = {**self.meta, "id": "clarkkitchen/s6e10-cloud-cv-synthetic"}
        with self.assertRaisesRegex(ValueError, "production namespace"):
            dispatch.dispatch_deadline(self.folder, invalid, 1500, True)
        for path in ["artifacts/second_pass/blend/frozen.json", "artifacts/third_pass_bad/blend/frozen.json",
                     "elsewhere/third_pass/blend/frozen.json", "artifacts/third_pass/final/frozen.json",
                     "artifacts/third_pass/blend/current.json"]:
            self.prepared["freeze_file"] = str(self.root / path)
            self.save()
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "isolated third-pass"):
                dispatch.dispatch_deadline(self.folder, self.meta, 1500, True)

    def test_runtime_metadata_and_refit_format_tampering_fail_before_push(self):
        for mapping, key, value in [(self.prepared, "runtime_sha256", "0" * 64),
                                     (self.meta, "competition_sources", ["different-competition"]),
                                     (self.spec, "format", "not_production")]:
            original = copy.deepcopy(mapping[key])
            mapping[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract differs"):
                self.invoke()
            mapping[key] = original
        self.save()
        (self.folder / "job.py").write_text("# changed reviewed runtime\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract differs"):
            self.invoke()
        self.api.kernels_push.assert_not_called()
        self.assertFalse((self.folder / "push_intent.json").exists())

    def test_private_and_quota_checks_still_apply_and_existing_intent_blocks_retry(self):
        self.meta["is_private"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "private"):
            self.invoke()
        self.meta["is_private"] = True
        self.save()
        self.api.quota_view.return_value.gpu_quota.time_reserved = timedelta(hours=2)
        with self.assertRaisesRegex(ValueError, "GPU allowance"):
            self.invoke()
        self.write(self.folder / "push_intent.json", {"synthetic": "uncertain prior push"})
        with self.assertRaisesRegex(RuntimeError, "Existing dispatch intent"):
            self.invoke()
        self.api.kernels_push.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
