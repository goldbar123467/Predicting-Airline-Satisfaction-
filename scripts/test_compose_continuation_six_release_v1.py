"""Generated full-size composition contracts; no real rows, fits, GPUs or APIs.

Controller context/remote-readback/ZIP primitives are mocked because their
separate generated suite exercises those boundaries. This suite exercises the
compositor's own byte/role/scope checks, real CSV/Parquet I/O and submitter intake.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import compose_continuation_six_release_v1 as composer
import submit_continuation_six_v1 as submitter
from continuation_six_policy_v1 import policy_for, policy_for_stage

PROJECT = Path(__file__).resolve().parents[1]
STAGE = "fixed_epoch_continuation_01"
CID = STAGE + "_release"
COUNT = 299844
STRICT = {"deterministic_algorithms":True,"deterministic_warn_only":False,"cudnn_benchmark":False,
          "cudnn_deterministic":True,"cuda_matmul_allow_tf32":False,"cublas_workspace_config":":4096:8",
          "startup_cuda_initialized":False}
RUNTIME_MODULES = ("pytabkit.models.optim.optimizers", "pytabkit.models.training.lightning_modules",
    "pytabkit.models.training.nn_creator", "pytabkit.models.training.scheduling", "pytabkit.models.training.coord",
    "pytabkit.models.training.lightning_callbacks", "pytabkit.models.data.data", "pytabkit.models.torch_utils",
    "pytabkit.models.nn_models.nn", "pytabkit.models.nn_models.base", "pytabkit.models.nn_models.categorical",
    "pytabkit.models.nn_models.activations")


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def mapping(root, paths):
    return {p.relative_to(root).as_posix():{"sha256":composer.sha(p),"bytes":p.stat().st_size} for p in paths}


def build_generated(root):
    """All rows and evidence below are generated, including the fake native file."""
    ids=np.arange(9000000,9000000+COUNT,dtype=np.int64)
    incumbent=.15+.7*((np.arange(COUNT)%997)/996)
    candidate=.03+.94*((np.arange(COUNT)*17%991)/990)
    (root/"data").mkdir()
    pd.DataFrame({"id":ids,"satisfaction":np.zeros(COUNT)}).to_csv(root/"data/sample_submission.csv",index=False)
    parent=root/"artifacts/third_pass/final";parent.mkdir(parents=True)
    parent_submission=parent/"submission.csv"
    pd.DataFrame({"id":ids,"satisfaction":incumbent}).to_csv(parent_submission,index=False)
    baseline=root/"artifacts/third_pass/blend/frozen.json"
    dump(baseline,{"oof_auc":.8,"fold_auc":[.8]*3})
    scripts=root/"scripts";scripts.mkdir()
    for name in ("compose_continuation_six_release_v1.py","continuation_six_selection_v1.py",
                 "continuation_six_policy_v1.py","continuation_six_cloud_v1.py",
                 "continuation_six_release_v1.py","submit_continuation_six_v1.py"):
        shutil.copyfile(PROJECT/"scripts"/name,scripts/name)
    (scripts/"generated_parent.py").write_text("# generated parent source",encoding="utf-8")
    (parent/"generated_native.bin").write_bytes(b"generated placeholder; never loaded as model")
    native={"artifacts/third_pass/final/generated_native.bin":composer.sha(parent/"generated_native.bin")}
    sources={"scripts/generated_parent.py":composer.sha(scripts/"generated_parent.py")}
    dump(parent/"release_provenance.json",{"native_hashes":native,"source_hashes":sources})
    parent_verification=root/"artifacts/third_pass/verification.json"
    dump(parent_verification,{"rows":COUNT,"all_row_raw_inference":True,"independent_blend_recomputation":True,
                             "audit_evaluated":False,"sha256":composer.sha(parent_submission)})
    dump(root/"state/continuation_six_v1/incumbent_release_binding.json",{"status":"generated",
         "files":{p.relative_to(root).as_posix():composer.sha(p) for p in (baseline,parent_submission,parent_verification,parent/"release_provenance.json")}})
    plan={"id":"continuation_six_v1","stages":[{"id":name,"index":i+1,"policy":policy_for_stage(name)} for i,name in enumerate(submitter.STAGES)],
          "budget":{"maximum_new_submissions":6},"incumbent_selection_path":baseline.relative_to(root).as_posix(),
          "incumbent_selection_sha256":composer.sha(baseline)}
    dump(root/"configs/continuation_six_v1.json",plan)
    now=datetime.now(timezone.utc)
    dump(root/"state/continuation_six_v1/authorization.json",{"id":"continuation_six_v1","status":"authorized",
        "maximum_new_submissions":6,"plan_sha256":composer.sha(root/"configs/continuation_six_v1.json"),
        "requested_utc":(now-timedelta(minutes=1)).isoformat(),"expires_utc":(now+timedelta(hours=23)).isoformat()})
    selection_path=root/f"artifacts/{STAGE}/release_selection.json"
    selection={"status":"frozen_for_release","stage":STAGE,"plan_sha256":composer.sha(root/"configs/continuation_six_v1.json"),
               "fixed_alpha":.1,"policy":policy_for_stage(STAGE),"gain_gate_passed":True,"audit_evaluated":False,
               "original_incumbent_submission_sha256":composer.sha(parent_submission),"oof_auc":.81,"fold_auc":[.81]*3}
    for name,path in (("evaluation",root/f"artifacts/{STAGE}/evaluation.json"),("protocol",root/f"artifacts/{STAGE}/protocol.json"),
                      ("original_incumbent_selection",baseline),("best_prior_selection",baseline)):
        if name in {"evaluation","protocol"}:dump(path,{"generated":True})
        selection[name+"_path"]=path.relative_to(root).as_posix();selection[name+"_sha256"]=composer.sha(path)
    dump(selection_path,selection)
    folder=root/f"cloud/continuation_six_v1/{STAGE}/release"
    workspace=folder/"assessment_workspace";payload=folder/"bundle/payload"
    out=workspace/f"artifacts/{CID}";out.mkdir(parents=True)
    registry={"id":CID,"stage":STAGE,"selection_sha256":composer.sha(selection_path),
              "trajectory_policies":{"A":policy_for("control"),"C":policy_for_stage(STAGE)},
              "expected_versions":{name:"generated-1" for name in ("torch","numpy","pandas","scikit-learn","pytabkit","pytorch-lightning")},
              "installed_runtime_source_sha256":{name:hashlib.sha256(("generated:"+name).encode()).hexdigest() for name in RUNTIME_MODULES}}
    dump(workspace/"registry.json",registry)
    dump(workspace/f"configs/{CID}.json",{"id":CID,"registry_sha256":composer.sha(workspace/"registry.json")})
    candidate_source=workspace/"scripts/generated_candidate.py"
    candidate_source.parent.mkdir();candidate_source.write_text("# generated candidate source",encoding="utf-8")
    candidate_native=out/"trajectory_C/epoch_016/graph.pt"
    candidate_native.parent.mkdir(parents=True);candidate_native.write_bytes(b"generated fake graph; never interpreted")
    execution={"source_sha256":composer.sha(candidate_source),"generated":True}
    dump(workspace/"provenance/execution_policy.json",execution)
    predictions=out/"test_predictions.parquet"
    pd.DataFrame({"id":ids,"prediction":candidate}).to_parquet(predictions,index=False)
    (workspace/"data").mkdir()
    shutil.copyfile(root/"data/sample_submission.csv",workspace/"data/sample_submission.csv")
    (workspace/"data/test.csv").write_text("generated raw-feature proof placeholder; no real data\n",encoding="utf-8")
    shutil.copyfile(workspace/"data/test.csv",root/"data/test.csv")
    dump(root/"state/continuation_six_v1/full_refit_input_binding.json",{"files":{
         name:composer.sha(root/name) for name in ("data/test.csv","data/sample_submission.csv")}})
    prefix=out/"trajectory_C/prefix_match.json"
    components=("network","learned_static_preprocessing","gradients","optimizer","cpu_rng","cuda_rng",
                "numpy_rng","python_rng","progress","schedule","sampler","batch_order","schema",
                "input_train","probe","modes","dropout_scopes","parameter_groups","input_monitor")
    prefix_state={}
    for kind,arm in (("control","A"),("treatment","C")):
        state_path=out/f"trajectory_{arm}/prefix_state.json";dump(state_path,{"generated":True,"epoch":4})
        prefix_state[kind+"_prefix_state_path"]=state_path.relative_to(workspace).as_posix()
        prefix_state[kind+"_prefix_state_sha256"]=composer.sha(state_path)
    dump(prefix,{"status":"passed","epoch":4,"gate_completed_before_epoch5":True,
                 "components_equal":{name:True for name in components},"native_probabilities_exact":True,
                 "update_schedule_prefix_exact":True,"observation_live_state_unchanged":True,**prefix_state})
    for role in ("release_controller","release_worker_A","release_worker_C"):
        dump(out/f"runtime_amendment/processes/{role}.json",{"role":role,"status":"passed","flags":STRICT,
             "registry_sha256":composer.sha(workspace/"registry.json"),"execution_policy_sha256":composer.sha(workspace/"provenance/execution_policy.json"),
             "source_sha256":execution["source_sha256"]})
        dump(out/f"runtime_compatibility/{role}.json",{"role":role,"status":"passed","before_full_fit":True,
             "registry_sha256":composer.sha(workspace/"registry.json"),"versions":registry["expected_versions"],
             "installed_sources":{name:{"installed_source_path":"/generated/"+name.replace(".","/")+".py","sha256":digest}
                                  for name,digest in registry["installed_runtime_source_sha256"].items()}})
    dump(out/"verification.json",{"id":CID,"stage":STAGE,"status":"passed","rows":COUNT,"train_rows":699635,"class_order":[0,1],
        "full_test_native_verified":True,"full_test_raw_inference":True,"audit_evaluated":False,
        "selection_sha256":composer.sha(selection_path),"registry_sha256":composer.sha(workspace/"registry.json"),
        "keyed_schema_verified":True,"native_reload_verified":True,"atol":2e-6,"rtol":1e-5,"chunks":[32768,8191],
        "full_test_max_abs_error":1e-7,"expected_test_ids_sha256":hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "raw_test_sha256":composer.sha(workspace/"data/test.csv"),"sample_submission_sha256":composer.sha(workspace/"data/sample_submission.csv"),
        "prefix_match_sha256":composer.sha(prefix),"heldout_quality_metrics_computed":False,"training_probe_metrics_recorded":True,
        "test_predictions_path":predictions.relative_to(workspace).as_posix(),"test_predictions_sha256":composer.sha(predictions),
        "source_hashes":{candidate_source.relative_to(workspace).as_posix():composer.sha(candidate_source)},
        "native_hashes":{candidate_native.relative_to(workspace).as_posix():composer.sha(candidate_native)}})
    dump(out/"completion_receipt.json",{"id":CID,"status":"completed","completed_fit_count":2,"completed_prefix_count":1,
         "full_test_native_verified":True,"release_ready":True,"verification_sha256":composer.sha(out/"verification.json"),
         "selection_sha256":composer.sha(selection_path),"registry_sha256":composer.sha(workspace/"registry.json"),"completed_startup_count":3})
    return_files=mapping(workspace,[p for p in out.rglob("*") if p.is_file()]+[workspace/"registry.json",workspace/f"configs/{CID}.json"])
    dump(folder/"generated_return_manifest.json",{"id":CID,"files":return_files})
    bundle_files=mapping(workspace,[workspace/"registry.json",workspace/f"configs/{CID}.json",candidate_source,workspace/"provenance/execution_policy.json",
                                  workspace/"data/test.csv",workspace/"data/sample_submission.csv"])
    for name in bundle_files:
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(workspace/name,target)
    dump(payload/"bundle-manifest.json",{"id":CID,"files":bundle_files})
    remote_source=folder/"remote_source_api.py";remote_source.write_text("# generated remote entry",encoding="utf-8")
    dump(folder/"push_intent.json",{"generated":True});dump(folder/"remote_metadata.json",{"generated":True})
    registration=root/f"state/continuation_six_v1/registrations/{STAGE}.release.json";dump(registration,{"generated":True})
    identity={"actual_kernel":"clarkkitchen/generated-fixture","kernel_id":123,"status":"verified","private_verified":True,
              "push_intent_sha256":composer.sha(folder/"push_intent.json"),"registration_sha256":composer.sha(registration),
              "remote_metadata_sha256":composer.sha(folder/"remote_metadata.json"),"remote_source_sha256":composer.sha(remote_source),
              "runtime_sha256":composer.sha(remote_source)}
    dump(folder/"provider_identity.json",identity)
    download=folder/"download_01"
    dump(download/"fixed_epoch_return/cloud_status.json",{"id":CID,"status":"release_complete"})
    dump(download/"retrieval.json",{"status":"retrieved_success_unscored","version":1,"ref":identity["actual_kernel"],
         "kernel_id":123,"identity_sha256":composer.sha(folder/"provider_identity.json"),
         "files":{"fixed_epoch_return/cloud_status.json":{"sha256":composer.sha(download/"fixed_epoch_return/cloud_status.json")}}})


class ComposerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_temp=tempfile.TemporaryDirectory(prefix="generated_composer_base_")
        cls.base=Path(cls.base_temp.name);build_generated(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.base_temp.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="generated_composer_");self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"workspace";shutil.copytree(self.base,self.root)
        self.folder=self.root/f"cloud/continuation_six_v1/{STAGE}/release"
        self.workspace=self.folder/"assessment_workspace";self.out=self.workspace/f"artifacts/{CID}"
        self.manifest=composer.read(self.folder/"generated_return_manifest.json")
        self.ctx={"folder":self.folder,"registration_path":self.root/f"state/continuation_six_v1/registrations/{STAGE}.release.json",
                  "prep":{"runtime_sha256":composer.sha(self.folder/"remote_source_api.py"),"registry_sha256":composer.sha(self.workspace/"registry.json")}}
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(composer,"plan_context",return_value=self.ctx))
        self.stack.enter_context(patch.object(composer,"validate_archive",side_effect=lambda *args:self.manifest))
        self.stack.enter_context(patch.object(composer,"completed_contract",return_value=True))
        self.stack.enter_context(patch.object(composer,"validate_remote",return_value=None))
        self.stack.enter_context(patch.object(composer,"FULL_REFIT_INPUT_BINDING_SHA",
             composer.sha(self.root/"state/continuation_six_v1/full_refit_input_binding.json")))
        # Real source files are copied into the generated workspace. The
        # compositor records those bytes; no production path is modified.
        self.stack.enter_context(patch.multiple(submitter,
            AUTHORITY_SHA=composer.sha(self.root/"state/continuation_six_v1/authorization.json"),
            INCUMBENT_SHA=composer.sha(self.root/"artifacts/third_pass/blend/frozen.json"),
            INCUMBENT_SUBMISSION_SHA=composer.sha(self.root/"artifacts/third_pass/final/submission.csv")))

    def refresh_return_member(self,path):
        self.manifest["files"][path.relative_to(self.workspace).as_posix()]={"sha256":composer.sha(path),"bytes":path.stat().st_size}

    def assert_no_release(self):
        self.assertFalse((self.root/f"artifacts/{STAGE}/release").exists())

    def rebind_candidate(self,value):
        path=self.out/"verification.json";dump(path,value);self.refresh_return_member(path)
        completion_path=self.out/"completion_receipt.json"
        completion=composer.read(completion_path);completion["verification_sha256"]=composer.sha(path)
        dump(completion_path,completion);self.refresh_return_member(completion_path)

    def test_full_generated_composition_passes_submission_contract(self):
        result=composer.compose(self.root,STAGE)
        self.assertEqual(result["rows"],COUNT)
        self.assertTrue(result["parent_native_verification_reused"])
        ctx=submitter.verification_contract(self.root,STAGE,mutation=True)
        self.assertEqual(ctx["sha256"],result["sha256"])
        self.assertTrue(all(gate["passed"] for gate in ctx["gates"].values()))
        frame=pd.read_csv(ctx["submission_path"])
        original=pd.read_csv(self.root/"artifacts/third_pass/final/submission.csv").satisfaction.to_numpy()
        candidate=pd.read_parquet(self.out/"test_predictions.parquet").prediction.to_numpy()
        # Different algebraic form provides an independent mixture oracle.
        np.testing.assert_allclose(frame.satisfaction.to_numpy(),original+(candidate-original)/10,atol=3e-16,rtol=0)
        expected_ids=hashlib.sha256(np.asarray(frame.id,dtype="<i8").tobytes()).hexdigest()
        self.assertEqual(result["expected_test_ids_sha256"],expected_ids)
        manifest=composer.read(self.root/f"artifacts/{STAGE}/release/manifest.json")
        self.assertEqual(manifest["candidate_verification_sha256"],composer.sha(self.out/"verification.json"))
        self.assertIn("scripts/submit_continuation_six_v1.py",result["source_hashes"])
        with self.assertRaises(FileExistsError):composer.compose(self.root,STAGE)

    def test_candidate_prediction_tamper_fails_before_blending(self):
        path=self.out/"test_predictions.parquet"
        with path.open("ab") as stream:stream.write(b"changed")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_candidate_source_tamper_fails_before_blending(self):
        (self.workspace/"scripts/generated_candidate.py").write_text("# changed",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_strict_flags_fail_even_if_return_member_digest_is_updated(self):
        path=self.out/"runtime_amendment/processes/release_worker_C.json"
        value=composer.read(path);value["flags"]["deterministic_warn_only"]=True;dump(path,value)
        self.refresh_return_member(path)
        with self.assertRaisesRegex(ValueError,"Strict release startup"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_missing_or_duplicate_startup_role_fails(self):
        path=self.out/"runtime_amendment/processes/release_worker_C.json"
        value=composer.read(path);value["role"]="release_worker_A";dump(path,value);self.refresh_return_member(path)
        with self.assertRaisesRegex(ValueError,"startup roles"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_runtime_compatibility_semantics_fail_after_member_rehash(self):
        path=self.out/"runtime_compatibility/release_worker_C.json"
        original=composer.read(path)
        mutations=("version","source","missing_source","extra_source","before_fit","role","registry")
        for change in mutations:
            with self.subTest(change=change):
                value=json.loads(json.dumps(original))
                if change=="version":value["versions"]["torch"]="generated-2"
                elif change=="source":value["installed_sources"][RUNTIME_MODULES[0]]["sha256"]="0"*64
                elif change=="missing_source":value["installed_sources"].pop(RUNTIME_MODULES[0])
                elif change=="extra_source":value["installed_sources"]["unexpected.module"]={"sha256":"0"*64}
                elif change=="before_fit":value["before_full_fit"]=False
                elif change=="role":value["role"]="release_worker_A"
                else:value["registry_sha256"]="0"*64
                dump(path,value);self.refresh_return_member(path)
                with self.assertRaisesRegex(ValueError,"installed runtime differs"):
                    composer.compose(self.root,STAGE)
                self.assert_no_release()

    def test_drifted_registry_rejected_against_prepared_payload(self):
        path=self.workspace/"registry.json"
        value=composer.read(path);value["changed"]=True;dump(path,value);self.refresh_return_member(path)
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_rebound_prediction_id_order_still_rejected(self):
        path=self.out/"test_predictions.parquet"
        frame=pd.read_parquet(path);frame.iloc[[0,1]]=frame.iloc[[1,0]].to_numpy();frame.to_parquet(path,index=False)
        proof=self.out/"verification.json";value=composer.read(proof);value["test_predictions_sha256"]=composer.sha(path)
        self.refresh_return_member(path);self.rebind_candidate(value)
        with self.assertRaisesRegex(ValueError,"identifiers or order"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_candidate_metadata_rebound_but_semantically_wrong_rejected(self):
        original=composer.read(self.out/"verification.json")
        for key,value in (("id","wrong"),("stage","wrong"),("registry_sha256","0"*64),("selection_sha256","0"*64),
                          ("native_reload_verified",False),("atol",1e-3),("chunks",[32768,32768])):
            with self.subTest(field=key):
                self.rebind_candidate({**original,key:value})
                with self.assertRaisesRegex(ValueError,"candidate native verification"):
                    composer.compose(self.root,STAGE)
                self.assert_no_release()

    def test_completion_hash_binding_rejected(self):
        path=self.out/"completion_receipt.json";value=composer.read(path);value["verification_sha256"]="0"*64
        dump(path,value);self.refresh_return_member(path)
        with self.assertRaisesRegex(ValueError,"Paired full-data release"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_prefix_failure_rebound_but_rejected(self):
        path=self.out/"trajectory_C/prefix_match.json";value=composer.read(path);value["components_equal"]["optimizer"]=False
        dump(path,value);self.refresh_return_member(path)
        candidate=composer.read(self.out/"verification.json");candidate["prefix_match_sha256"]=composer.sha(path);self.rebind_candidate(candidate)
        with self.assertRaisesRegex(ValueError,"training-state prefix"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()

    def test_raw_binding_and_parity_error_rejected(self):
        original=composer.read(self.out/"verification.json")
        for key,value,pattern in (("raw_test_sha256","0"*64,"raw test/sample"),
                                  ("full_test_max_abs_error",1e-3,"parity exceeds")):
            with self.subTest(field=key):
                self.rebind_candidate({**original,key:value})
                with self.assertRaisesRegex(ValueError,pattern):composer.compose(self.root,STAGE)
                self.assert_no_release()

    def test_wrong_prediction_id_digest_never_gets_verified_receipt(self):
        candidate=composer.read(self.out/"verification.json");candidate["expected_test_ids_sha256"]="0"*64;self.rebind_candidate(candidate)
        with self.assertRaisesRegex(ValueError,"ID digest differs"):
            composer.compose(self.root,STAGE)
        output=self.root/f"artifacts/{STAGE}/release"
        self.assertFalse((output/"verification.json").exists())
        self.assertFalse((output/"release_provenance.json").exists())

    def test_wrong_alpha_or_no_incremental_gain_fails(self):
        path=self.root/f"artifacts/{STAGE}/release_selection.json"
        selection=composer.read(path);selection["fixed_alpha"]=.2;dump(path,selection)
        with self.assertRaisesRegex(ValueError,"no longer qualifies"):
            composer.compose(self.root,STAGE)
        selection["fixed_alpha"]=.1;selection["oof_auc"]=.8;dump(path,selection)
        with self.assertRaisesRegex(ValueError,"no longer qualifies"):
            composer.compose(self.root,STAGE)
        self.assert_no_release()


if __name__=="__main__":
    unittest.main()
