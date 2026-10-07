"""Prepare, but never upload or execute, a private Kaggle RealMLP GPU probe."""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
FILES = ("realmlp.py", "realmlp_categorical.py", "categorical_transform.py")
REQUIREMENTS = "pytabkit==1.7.3\npytorch-lightning==2.6.6\ntorchmetrics==1.9.0\n"

RUNTIME = r'''
import base64, contextlib, gc, hashlib, importlib, importlib.metadata as md
import json, os, platform, subprocess, sys, time, traceback, zlib
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def version(name):
    try:return md.version(name)
    except md.PackageNotFoundError:return None

def install_minimal(out, report):
    # Keep the working Kaggle binary stack exactly as provisioned.
    protected = ['torch','torchvision','torchaudio','numpy','pandas','scikit-learn','scipy']
    protected += [d.metadata['Name'] for d in md.distributions()
                  if d.metadata['Name'].lower().startswith(('nvidia-','cuda-','triton'))]
    before = {name:version(name) for name in protected if version(name) is not None}
    if before.get('torch') != '2.10.0+cu128':
        raise RuntimeError('Expected probed Kaggle torch 2.10.0+cu128; re-inspect a changed image')
    constraints = out/'protected-stack.txt'
    constraints.write_text(''.join(f'{n}=={v}\n' for n,v in before.items()))
    requirements = out/'requirements-realmlp.txt'
    requirements.write_text(REQUIREMENTS)
    plan = out/'pip-dry-run.json'
    cmd = [sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input',
           '--dry-run','--report',str(plan),'--constraint',str(constraints),
           '--requirement',str(requirements),'--upgrade-strategy','only-if-needed']
    with (out/'pip-dry-run.log').open('w') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    allowed = {'pytabkit','pytorch-lightning','torchmetrics','lightning-utilities','psutil',
               'packaging','typing-extensions','pyyaml','tqdm','fsspec','aiohttp',
               'aiohappyeyeballs','aiosignal','attrs','frozenlist','multidict','propcache',
               'yarl','idna','async-timeout','setuptools'}
    records = json.loads(plan.read_text())['install']
    wheels=[]
    for item in records:
        name=item['metadata']['name'].lower().replace('_','-')
        if name not in allowed:raise RuntimeError(f'Unreviewed dependency change: {name}')
        info=item['download_info']; url=info['url']; sha=info['archive_info']['hashes']['sha256']
        if not url.startswith('https://files.pythonhosted.org/') or not url.endswith('.whl'):
            raise RuntimeError(f'Only official PyPI wheels permitted: {name}')
        if name=='pytabkit' and sha!='1589f281e99d4a6965a83f954d9f78b038fc633a4246d450a1a3599952d4d841':
            raise RuntimeError('PyTabKit wheel differs from inspected official 1.7.3 wheel')
        wheels.append(url+'#sha256='+sha)
    if wheels:
        with (out/'pip-install.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                            '--no-input','--no-deps',*wheels],check=True,stdout=log,
                           stderr=subprocess.STDOUT,timeout=180)
    importlib.invalidate_caches()
    after={name:version(name) for name in before}
    if before!=after:raise RuntimeError('Protected binary/CUDA stack changed')
    # Check required base distributions without imposing unrelated notebook extras.
    from packaging.requirements import Requirement
    for name in ['pytabkit','pytorch-lightning','torchmetrics']:
        for text in md.metadata(name).get_all('Requires-Dist',[]):
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            installed=version(requirement.name)
            if installed is None or installed not in requirement.specifier:
                raise RuntimeError(f'Unsatisfied minimal dependency: {text}, found {installed}')
    report['dependency_install']={'requested':REQUIREMENTS.splitlines(),'changed':[
        {'name':i['metadata']['name'],'version':i['metadata']['version']} for i in records],
        'protected_before':before,'protected_after':after,'pytabkit_version':version('pytabkit'),
        'pip_dry_run_sha256':digest(plan)}

def synthetic(out, report):
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from realmlp_categorical import fit_realmlp_categorical, load_realmlp_categorical, _fit_schema, _InputSplit
    from categorical_transform import CategoricalTransform
    if not torch.cuda.is_available():raise RuntimeError('GPU probe requires CUDA')
    torch.set_num_threads(4); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)
    report['hardware']={'cpu_count':os.cpu_count(),'ram_bytes':psutil.virtual_memory().total,
        'cuda_build':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'device_used':'cuda:0','multi_gpu_training':False}
    # Fold-local preprocessing check: arbitrary indices, unseen category, reload.
    frame=pd.DataFrame({'numeric':[1.,2.,np.nan,4.],'category':['a','b','a','a']},index=[4,8,3,1])
    held=pd.DataFrame({'numeric':[99.],'category':['unseen']})
    transform=CategoricalTransform('realmlp_cat').fit(frame)
    assert transform.transform(held)[0,-1]==-1 and transform.mean['numeric']<4
    transform.save(out/'synthetic_transform.json')
    np.testing.assert_array_equal(transform.transform(frame),CategoricalTransform.load(out/'synthetic_transform.json').transform(frame))
    rng=np.random.default_rng(3201)
    X=rng.normal(size=(900,9)).astype(np.float32)
    X[:,4]=rng.integers(0,3,900); X[:,5]=rng.integers(0,100,900)
    X[:,6]=-1; X[:,7]=rng.integers(0,2,900); X[:,8]=4
    X[650:700,5]=555
    y=(X[:,0]+X[:,1]+.5*(X[:,4]==2)>0).astype(np.int64)
    config={'device':'cuda','seed':3201,'threads':4,'epochs':2,'n_ens':8,
        'hidden_sizes':[512,256,128],'batch_size':256,'eval_batch_size':37,'ls_eps':0.,
        'categorical_indices':[4,5,6,7,8]}
    schema=_fit_schema(X[:600],config['categorical_indices'])
    assert schema['cardinalities']==[4,101,1,3,2]
    split=_InputSplit(schema)(torch.from_numpy(X[:17]))
    assert tuple(split['x_cont'].shape)==(17,4) and tuple(split['x_cat'].shape)==(17,5)
    assert split['x_cat'].dtype==torch.int64
    for bad in [[4,4],[9]]:
        try:_fit_schema(X[:600],bad)
        except ValueError:pass
        else:raise AssertionError('Invalid categorical index contract accepted')
    torch.cuda.synchronize(); started=time.perf_counter()
    inner,selected=fit_realmlp_categorical(X[:600],y[:600],X[600:],y[600:],config,out/'synthetic_inner')
    torch.cuda.synchronize(); inner_seconds=time.perf_counter()-started
    assert 555 not in inner.metadata['input_schema']['vocabularies'][1]
    del inner;gc.collect();torch.cuda.empty_cache()
    started=time.perf_counter()
    fixed,epoch=fit_realmlp_categorical(X[:600],y[:600],None,None,
        dict(config,epochs=selected),out/'synthetic_fixed')
    torch.cuda.synchronize(); fixed_seconds=time.perf_counter()-started
    restored=load_realmlp_categorical(out/'synthetic_fixed',device='cuda')
    cpu=load_realmlp_categorical(out/'synthetic_fixed',device='cpu')
    reload_error=cpu_error=0.
    for n in [1,17,257]:
        a=fixed.predict_proba(X[-n:]);b=restored.predict_proba(X[-n:]);c=cpu.predict_proba(X[-n:])
        assert a.shape==(n,2) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=2e-6)
        np.testing.assert_allclose(a,c,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(a.sum(1),1.,rtol=1e-5,atol=2e-6)
        reload_error=max(reload_error,float(np.max(np.abs(a-b))))
        cpu_error=max(cpu_error,float(np.max(np.abs(a-c))))
    a=X[:17].copy();b=a.copy();a[:,config['categorical_indices']]=-1;b[:,config['categorical_indices']]=555
    np.testing.assert_array_equal(restored.predict_proba(a),restored.predict_proba(b))
    assert restored.predict_proba(X[:0]).shape==(0,2) and epoch==selected
    report['synthetic']={'passed':True,'config':config,'selected_epoch':selected,'fixed_epoch':epoch,
        'train_rows':600,'inner_validation_rows':300,'inner_fit_export_seconds':inner_seconds,
        'fixed_fit_export_seconds':fixed_seconds,'batches_checked':[1,17,257],
        'validation_only_category_excluded':True,'unknown_category_equivalence':True,
        'reload_max_abs_error':reload_error,'cpu_cuda_max_abs_error':cpu_error,
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated(0)/2**20,
        'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved(0)/2**20,
        'process_rss_mib':psutil.Process().memory_info().rss/2**20,
        'native_export_max_abs_error':fixed.metadata['native_export_max_abs_error']}

def main():
    out=Path('/kaggle/working')
    if not out.is_dir() or not Path('/kaggle/input').is_dir():
        raise RuntimeError('This prepared probe is Kaggle-only; no local training permitted')
    out=out/'realmlp_probe';out.mkdir(exist_ok=True)
    report={'status':'running','scope':'synthetic compatibility only; no real labels or model selection',
        'source_manifest':SOURCE_MANIFEST,'python':platform.python_version(),'started_unix':time.time()}
    try:
        install_minimal(out,report)
        source=out/'source';source.mkdir(exist_ok=True)
        for name,encoded in EMBEDDED.items():
            data=zlib.decompress(base64.b64decode(encoded));path=source/name;path.write_bytes(data)
            if digest(path)!=SOURCE_MANIFEST[name]:raise RuntimeError('Embedded source hash mismatch')
        sys.path.insert(0,str(source))
        synthetic(out,report)
        report['versions']={n:version(n) for n in ['pytabkit','torch','pytorch-lightning','lightning','torchmetrics','numpy','pandas','scikit-learn']}
        report['status']='passed'
    except Exception:
        report['status']='failed';report['traceback']=traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds']=time.time()-report['started_unix']
        (out/'realmlp_probe_report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
'''


def prepare(destination: Path, owner: str, slug: str) -> dict:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a fresh preparation directory; source snapshots are immutable")
    destination.mkdir(parents=True, exist_ok=True)
    sources = {name: (ROOT / "scripts" / name).read_bytes() for name in FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    embedded = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in sources.items()}
    program = ("# Self-contained private RealMLP compatibility probe.\n"
               f"SOURCE_MANIFEST = {hashes!r}\nEMBEDDED = {embedded!r}\nREQUIREMENTS = {REQUIREMENTS!r}\n" + RUNTIME)
    ast.parse(program)
    (destination / "probe.py").write_text(program, encoding="utf-8", newline="\n")
    (destination / "requirements-realmlp.txt").write_text(REQUIREMENTS, encoding="utf-8")
    metadata = {"id": f"{owner}/{slug}", "title": slug.replace("-", " "),
                "code_file": "probe.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "competition_sources": [], "dataset_sources": [],
                "kernel_sources": [], "model_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": hashes,
                "probe_sha256": hashlib.sha256((destination / "probe.py").read_bytes()).hexdigest(),
                "requirements": REQUIREMENTS.splitlines(), "kaggle_ref": metadata["id"],
                "recommended_timeout_seconds": 600, "prepared_only": True, "launched": False}
    (destination / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "cloud/realmlp_probe")
    parser.add_argument("--owner", default="clarkkitchen")
    parser.add_argument("--slug", default="s6e10-realmlp-compatibility-20261002")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.output, arguments.owner, arguments.slug), indent=2))
