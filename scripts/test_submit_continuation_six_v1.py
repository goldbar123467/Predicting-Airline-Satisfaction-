"""Bounded generated artifacts and fake-provider tests; never calls Kaggle."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import submit_continuation_six_v1 as submitter


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def fixture(root, stage=None, *, auc=.81, prior=None):
    stage = stage or submitter.STAGES[0]
    now = datetime.now(timezone.utc)
    baseline = root/"artifacts/third_pass/blend/frozen.json"
    if not baseline.exists():
        dump(baseline,{"oof_auc":.8,"fold_auc":[.8]*3})
    source = root/"scripts/submit_continuation_six_v1.py"
    source.parent.mkdir(parents=True,exist_ok=True);source.write_text("# generated source fixture",encoding="utf-8")
    native = root/"artifacts/native.bin"
    native.write_bytes(b"generated fake native evidence; no model")
    plan_path = root/f"configs/{submitter.SEQUENCE}.json"
    if not plan_path.exists():
        dump(plan_path,{"id":submitter.SEQUENCE,"stages":[{"id":name,"policy":{"id":name}} for name in submitter.STAGES],
            "budget":{"maximum_new_submissions":6},"incumbent_selection_path":baseline.relative_to(root).as_posix(),
            "incumbent_selection_sha256":submitter.sha(baseline)})
    authority_path = root/f"state/{submitter.SEQUENCE}/authorization.json"
    if not authority_path.exists():
        dump(authority_path,{"id":submitter.SEQUENCE,"status":"authorized","maximum_new_submissions":6,
            "plan_sha256":submitter.sha(plan_path),"requested_utc":(now-timedelta(minutes=1)).isoformat(),
            "expires_utc":(now+timedelta(hours=23)).isoformat()})
    selection_path = root/f"artifacts/{stage}/release_selection.json"
    evaluation,protocol = root/f"artifacts/{stage}/evaluation.json",root/f"artifacts/{stage}/protocol.json"
    dump(evaluation,{"generated":True});dump(protocol,{"generated":True})
    prior = prior or baseline
    selection = {"status":"frozen_for_release","stage":stage,"plan_sha256":submitter.sha(plan_path),
        "fixed_alpha":.1,"gain_gate_passed":True,"policy":{"id":stage},"audit_evaluated":False,
        "evaluation_path":evaluation.relative_to(root).as_posix(),"evaluation_sha256":submitter.sha(evaluation),
        "protocol_path":protocol.relative_to(root).as_posix(),"protocol_sha256":submitter.sha(protocol),
        "original_incumbent_selection_path":baseline.relative_to(root).as_posix(),"original_incumbent_selection_sha256":submitter.sha(baseline),
        "original_incumbent_submission_sha256":submitter.INCUMBENT_SUBMISSION_SHA,
        "best_prior_selection_path":prior.relative_to(root).as_posix(),"best_prior_selection_sha256":submitter.sha(prior),
        "oof_auc":auc,"fold_auc":[auc]*3}
    dump(selection_path,selection)
    out = root/f"artifacts/{stage}/release";out.mkdir(parents=True,exist_ok=True)
    csv = out/"submission.csv";csv.write_text("id,satisfaction\n1,0.1\n2,0.7\n3,0.9\n",encoding="utf-8")
    ids_sha=hashlib.sha256(b"".join(struct.pack("<q",x) for x in [1,2,3])).hexdigest()
    maps={"source_hashes":{source.relative_to(root).as_posix():submitter.sha(source)},
          "native_hashes":{native.relative_to(root).as_posix():submitter.sha(native)}}
    workspace=root/f"cloud/{submitter.SEQUENCE}/{stage}/release/assessment_workspace"
    candidate=workspace/f"artifacts/{stage}_release/verification.json"
    dump(candidate,{"generated":True});dump(workspace/"registry.json",{"generated":True})
    verification={"status":"verified","stage":stage,"rows":3,"sha256":submitter.sha(csv),
        "all_row_raw_inference":True,"independent_blend_recomputation":True,"audit_evaluated":False,"class_order":[0,1],
        "keyed_schema_verified":True,"native_reload_verified":True,"fixed_alpha":.1,
        "selection_sha256":submitter.sha(selection_path),"ids_sha256":ids_sha,"expected_test_ids_sha256":ids_sha,
        "registry_sha256":submitter.sha(workspace/"registry.json"),**maps}
    dump(out/"verification.json",verification)
    dump(out/"manifest.json",{"stage":stage,"id":stage+"_release","fixed_alpha":.1,"submission_sha256":submitter.sha(csv),
        "selection_sha256":submitter.sha(selection_path),"candidate_verification_path":candidate.relative_to(root).as_posix(),
        "candidate_verification_sha256":submitter.sha(candidate),"registry_sha256":submitter.sha(workspace/"registry.json"),**maps})
    provenance={"status":"verified","stage":stage,"fixed_alpha":.1,"submission_sha256":submitter.sha(csv),"verification_sha256":submitter.sha(out/"verification.json"),
        "manifest_sha256":submitter.sha(out/"manifest.json"),"selection_sha256":submitter.sha(selection_path),**maps}
    dump(out/"release_provenance.json",provenance)
    return authority_path,baseline,selection_path,out


class GeneratedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.authority,self.baseline,self.selection,self.out=fixture(self.root)
        self.patch=patch.multiple(submitter,AUTHORITY_SHA=submitter.sha(self.authority),INCUMBENT_SHA=submitter.sha(self.baseline),ROWS=3)
        self.patch.start()
        self.ctx=submitter.verification_contract(self.root,submitter.STAGES[0],mutation=True)

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def api(self):
        api=NS(history=[],competition_get_submission_limits=Mock(return_value=NS(num_allowed_now=2,num_today=8,num_total=31,limited_by_total=False)))
        def upload(path,description,competition,quiet):
            row={"ref":100+len(api.history),"description":description,"file_name":Path(path).name,
                 "total_bytes":Path(path).stat().st_size,"status":"PENDING","public_score":"","date":None}
            api.history.append(row)
            return NS(ref=row["ref"])
        api.competition_submit=Mock(side_effect=upload)
        return api

    @staticmethod
    def history(api):
        return list(api.history),[{"index":0,"count":len(api.history),"has_next_page":False}]

    def refresh_provenance(self):
        value=submitter.read(self.out/"release_provenance.json")
        value["verification_sha256"]=submitter.sha(self.out/"verification.json")
        value["selection_sha256"]=submitter.sha(self.selection)
        dump(self.out/"release_provenance.json",value)

    def test_import_is_stdlib_only(self):
        result=subprocess.run([sys.executable,"-c","import sys; import submit_continuation_six_v1; assert not any(x in sys.modules for x in ['torch','numpy','pandas','kaggle'])"],cwd=Path(submitter.__file__).parent,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_verified_contract_and_original_gate(self):
        self.assertTrue(all(value["passed"] for value in self.ctx["gates"].values()))
        self.assertEqual(self.ctx["submission_path"],self.out/"submission.csv")

    def test_gates_pooled_macro_and_worst_fold(self):
        base={"oof_auc":.8,"fold_auc":[.8]*3}
        self.assertTrue(submitter.gain_gate({"oof_auc":.8001,"fold_auc":[.8001]*3},base)["passed"])
        for value in ({"oof_auc":.8,"fold_auc":[.8001]*3},
                      {"oof_auc":.8001,"fold_auc":[.8]*3},
                      {"oof_auc":.8001,"fold_auc":[.79997,.801,.801]}):
            self.assertFalse(submitter.gain_gate(value,base)["passed"])
        for value in ({"oof_auc":float("nan"),"fold_auc":[.8]*3},{"oof_auc":True,"fold_auc":[.8]*3},{"oof_auc":.8,"fold_auc":[.8]*2}):
            with self.assertRaises(ValueError):submitter.metric(value)

    def test_csv_invalid_schema_duplicate_nonfinite_and_order(self):
        expected=submitter.read(self.out/"verification.json")["ids_sha256"]
        for content in ["satisfaction,id\n.1,1\n.7,2\n.9,3\n","id,satisfaction\n1,.1\n1,.7\n3,.9\n", "id,satisfaction\n1,nan\n2,.7\n3,.9\n", "id,satisfaction\n2,.1\n1,.7\n3,.9\n", "id,satisfaction\n1,1.1\n2,.7\n3,.9\n"]:
            path=self.root/"bad.csv";path.write_text(content,encoding="utf-8")
            with self.assertRaises(ValueError):submitter.csv_contract(path,expected,expected_rows=3)

    def test_no_audit_or_missing_native_certification(self):
        original=submitter.read(self.out/"verification.json")
        for key,value in [("audit_evaluated",True),("all_row_raw_inference",False),("native_reload_verified",False),("independent_blend_recomputation",False)]:
            dump(self.out/"verification.json",{**original,key:value});self.refresh_provenance()
            with self.assertRaisesRegex(ValueError,"verification"):
                submitter.verification_contract(self.root,submitter.STAGES[0])

    def test_changed_source_model_or_selection_rejected(self):
        for path in [self.root/"artifacts/native.bin",self.root/"scripts/submit_continuation_six_v1.py",self.selection]:
            before=path.read_bytes();path.write_bytes(before+b"x")
            with self.assertRaises(ValueError):submitter.verification_contract(self.root,submitter.STAGES[0])
            path.write_bytes(before)

    def test_best_prior_must_be_incrementally_beaten(self):
        second=submitter.STAGES[1]
        fixture(self.root,second,auc=.811,prior=self.selection)
        self.assertTrue(submitter.verification_contract(self.root,second)["gates"]["best_prior"]["passed"])
        fixture(self.root,second,auc=.810001,prior=self.selection)
        with self.assertRaisesRegex(ValueError,"incremental"):
            submitter.verification_contract(self.root,second)
        fixture(self.root,second,auc=.82,prior=self.baseline)
        with self.assertRaisesRegex(ValueError,"best prior"):
            submitter.verification_contract(self.root,second)

    def test_expired_window_rejected_without_network(self):
        with self.assertRaisesRegex(ValueError,"closed"):
            submitter.verification_contract(self.root,submitter.STAGES[0],now=self.ctx["end"]+timedelta(seconds=1),mutation=True)
        submitter.authority_context(self.root,now=self.ctx["end"]+timedelta(days=1),mutation=False)

    def test_submit_records_intent_and_reconciles_no_duplicate(self):
        api=self.api()
        def checked_upload(*args,**kwargs):
            intent=self.root/f"state/{submitter.SEQUENCE}/submissions/{self.ctx['stage']}/intent.json"
            self.assertTrue(intent.exists())
            self.assertEqual(submitter.read(intent)["sha256"],self.ctx["sha256"])
            row={"ref":123,"description":args[1],"file_name":"submission.csv","total_bytes":self.ctx["submission_bytes"],"status":"PENDING","public_score":"","date":None}
            api.history.append(row);return NS(ref=123)
        api.competition_submit.side_effect=checked_upload
        result=submitter.submit(self.ctx,api,history_loader=self.history)
        self.assertEqual(result["submission"]["ref"],123)
        again=submitter.submit(self.ctx,api,history_loader=self.history)
        self.assertEqual(again["status"],"reconciled_remote")
        self.assertEqual(api.competition_submit.call_count,1)

    def test_unknown_upload_blocks_retry_and_later_stage(self):
        api=self.api();api.competition_submit.side_effect=RuntimeError("secret-signed-url")
        with self.assertRaisesRegex(RuntimeError,"unknown") as error:
            submitter.submit(self.ctx,api,history_loader=self.history)
        self.assertNotIn("secret",str(error.exception))
        for ctx in [self.ctx,{**self.ctx,"stage":submitter.STAGES[1]}]:
            with self.assertRaisesRegex(RuntimeError,"unknown"):
                submitter.submit(ctx,api,history_loader=self.history)
        self.assertEqual(api.competition_submit.call_count,1)

    def test_readonly_reconcile_unknown_response(self):
        api=self.api();api.competition_submit.side_effect=RuntimeError("unknown")
        with self.assertRaises(RuntimeError):submitter.submit(self.ctx,api,history_loader=self.history)
        path=self.root/f"state/{submitter.SEQUENCE}/submissions/{self.ctx['stage']}/intent.json"
        intent=submitter.read(path)
        row={"ref":42,"description":intent["description"],"file_name":intent["file_name"],"total_bytes":intent["submission_bytes"],"status":"COMPLETE","public_score":"0.9","date":None}
        result=submitter.reconcile_intent(path,self.ctx,[row],[])
        self.assertEqual(result["cloud_mutations_performed"],0)
        self.assertEqual(api.competition_submit.call_count,1)
        with self.assertRaises(RuntimeError):submitter.reconcile_intent(path,self.ctx,[row,row],[])

    def test_allowance_and_mutation_boundary_hash_check(self):
        api=self.api();api.competition_get_submission_limits.return_value.num_allowed_now=0
        with self.assertRaisesRegex(RuntimeError,"allowance"):
            submitter.submit(self.ctx,api,history_loader=self.history)
        self.assertFalse(submitter.intent_paths(self.root));api.competition_submit.assert_not_called()
        def mutate(_):
            (self.root/"artifacts/native.bin").write_bytes(b"changed during remote preflight")
            return NS(num_allowed_now=1,num_today=9,num_total=40,limited_by_total=False)
        api.competition_get_submission_limits.side_effect=mutate
        with self.assertRaisesRegex(ValueError,"changed"):
            submitter.submit(self.ctx,api,history_loader=self.history)
        self.assertFalse(submitter.intent_paths(self.root));api.competition_submit.assert_not_called()

    def test_hash_deduplication_and_remote_size_check(self):
        api=self.api();api.history=[{"description":"prior sha256:"+self.ctx["sha256"]}]
        with self.assertRaisesRegex(RuntimeError,"already remote"):
            submitter.submit(self.ctx,api,history_loader=self.history)
        api=self.api();submitter.submit(self.ctx,api,history_loader=self.history)
        api.history[0]["total_bytes"]+=1
        with self.assertRaisesRegex(ValueError,"size/name"):
            submitter.submit(self.ctx,api,history_loader=self.history)

    def test_path_escape_and_seventh_intent_rejected(self):
        for name in ["../x","a/../../x","/tmp/x","C:/x","a\\b"]:
            with self.assertRaises(ValueError):submitter.checked(self.root,name)
        for index in range(7):
            dump(self.root/f"state/{submitter.SEQUENCE}/submissions/generated{index}/intent.json",{})
        with self.assertRaisesRegex(ValueError,"count/stage"):
            submitter.intent_paths(self.root)


class PaginationTests(unittest.TestCase):
    @staticmethod
    def row(ref):
        return NS(ref=ref,description="fixture",file_name="x.csv",total_bytes=12,status="COMPLETE",public_score=".9",date=None)

    def test_underlying_pagination_keeps_all_pages(self):
        calls=[]
        def fetch(token):
            calls.append(token)
            return NS(submissions=[self.row(1 if token is None else 2)],next_page_token="next" if token is None else "")
        values,pages=submitter.collect_submissions(fetch)
        self.assertEqual([x["ref"] for x in values],[1,2]);self.assertEqual(calls,[None,"next"])
        self.assertEqual(len(pages),2)

    def test_repeated_token_or_duplicate_identity_stops(self):
        count=0
        def token(_):
            nonlocal count
            count+=1
            return NS(submissions=[self.row(count)],next_page_token="same")
        with self.assertRaisesRegex(ValueError,"Repeated"):
            submitter.collect_submissions(token)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            submitter.collect_submissions(lambda _:NS(submissions=[self.row(1)],next_page_token="again"))


if __name__=="__main__":
    unittest.main()
