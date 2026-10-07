"""Shared data contracts, fold-local preprocessing, and native model IO."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'satisfaction'
CAT = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.csv')
    frame.to_csv(temp,index=False,float_format='%.12g')
    os.replace(temp,path)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_parquet(ROOT/'data/train.parquet')
    te = pd.read_parquet(ROOT/'data/test.parquet')
    split = pd.read_parquet(ROOT/'data/splits.parquet')
    assert np.array_equal(tr.id.to_numpy(), split.id.to_numpy()), 'Split identity mismatch'
    return tr, te, split

def features(df: pd.DataFrame, run: dict) -> pd.DataFrame:
    x = df.drop(columns=['id', TARGET], errors='ignore').copy()
    if run.get('categorical_twins', False):
        # Preserve each original numeric measurement as well as its exact-value
        # category. No vocabulary is estimated here; that happens within fit.
        for c in list(x.columns):
            if pd.api.types.is_numeric_dtype(x[c]):
                x[c+'_category'] = x[c].astype('float32').astype(str)
        x['class_travel_gender'] = (x['Class'].astype(str)+'|'+
            x['Type of Travel'].astype(str)+'|'+x['Gender'].astype(str))
    if run.get('route', False):
        x['route_category'] = x['Flight Distance'].astype(str)
    if run.get('features', False):
        values = x[RATINGS].astype('float32')
        x['rating_mean'] = values.mean(axis=1)
        x['rating_std'] = values.std(axis=1)
        x['rating_zero_count'] = (values == 0).sum(axis=1)
        x['rating_low_count'] = ((values > 0) & (values <= 2)).sum(axis=1)
        x['rating_high_count'] = (values >= 4).sum(axis=1)
        x['digital_mean'] = x[['Inflight wifi service','Ease of Online booking','Online boarding']].mean(axis=1)
        x['comfort_mean'] = x[['Seat comfort','Inflight entertainment','Food and drink','Cleanliness']].mean(axis=1)
        x['delay_difference'] = x['Arrival Delay in Minutes'] - x['Departure Delay in Minutes']
        for c in ['Departure Delay in Minutes','Arrival Delay in Minutes','Flight Distance']:
            x[c+'_log1p'] = np.log1p(x[c].clip(lower=0))
    if run.get('teacher', False):
        teacher = pd.read_parquet(ROOT/'data/teacher_predictions.parquet')
        assert teacher.id.is_unique
        mapped = df[['id']].merge(teacher, on='id', how='left', validate='one_to_one', sort=False)
        assert np.array_equal(mapped.id.to_numpy(), df.id.to_numpy())
        assert mapped.teacher_probability.notna().all(), 'Missing teacher row'
        x['teacher_probability'] = mapped.teacher_probability.to_numpy()
    for flag, filename, expected_columns in [
        ('original_aux', 'original_aux_predictions.parquet', [f'orig_aux_{i:02d}' for i in range(13)]),
        ('original_aux_probability', 'original_aux_probability_predictions.parquet', [f'orig_aux_prob_{i:02d}' for i in range(13)] + ['orig_aux_log_score']),
        ('original_realmlp_teacher', 'original_realmlp_teacher_predictions.parquet', ['original_realmlp_teacher_logit']),
        ('original_lgb_teacher', 'original_lgb_teacher_predictions.parquet', ['original_lgb_teacher_probability']),
    ]:
        if not run.get(flag, False):
            continue
        cached = pd.read_parquet(ROOT/'data'/filename)
        if list(cached.columns) != ['id'] + expected_columns or not cached.id.is_unique:
            raise ValueError(f'Invalid {flag} feature cache schema or identifiers')
        mapped = df[['id']].merge(cached, on='id', how='left', validate='one_to_one', sort=False)
        if not np.array_equal(mapped.id.to_numpy(), df.id.to_numpy()):
            raise ValueError(f'{flag} feature join changed row identity')
        values = mapped[expected_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Missing or nonfinite {flag} feature rows')
        for j, column in enumerate(expected_columns):
            x[column] = values[:, j]
    return x

class Transform:
    def __init__(self, family: str):
        self.family = family

    def fit(self, x: pd.DataFrame) -> 'Transform':
        self.columns = list(x.columns)
        self.cats = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        self.vocab = {c: sorted(x[c].fillna('__MISSING__').astype(str).unique().tolist()) for c in self.cats}
        if self.family in {'tabm','realmlp'} and sum(map(len,self.vocab.values()))>256:
            raise ValueError('High-cardinality neural one-hot exceeds this machine budget; use cross-fitted route encoding or native categorical embeddings')
        self.nums = [c for c in x if c not in self.cats]
        self.median = {c: float(x[c].median()) if x[c].notna().any() else 0.0 for c in self.nums}
        self.mean = {c: float(x[c].fillna(self.median[c]).mean()) for c in self.nums}
        self.std = {c: max(float(x[c].fillna(self.median[c]).std()), 1e-6) for c in self.nums}
        return self

    def transform(self, x: pd.DataFrame):
        assert list(x.columns) == self.columns
        if self.family in {'tabm','realmlp'}:
            pieces = [np.column_stack([((x[c].fillna(self.median[c]).to_numpy(dtype=np.float32)-self.mean[c])/self.std[c]) for c in self.nums])]
            for c in self.cats:
                codes = pd.Categorical(x[c].fillna('__MISSING__').astype(str), categories=self.vocab[c]).codes
                onehot = np.zeros((len(x),len(self.vocab[c])), dtype=np.float32)
                known = codes >= 0
                onehot[np.flatnonzero(known), codes[known]] = 1
                pieces.append(onehot)
            return np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)
        out = pd.DataFrame(index=x.index)
        for c in self.columns:
            if c in self.cats:
                if self.family == 'catboost':
                    out[c] = x[c].fillna('__MISSING__').astype(str)
                elif self.family == 'lightgbm':
                    out[c] = pd.Categorical(x[c].fillna('__MISSING__').astype(str), categories=self.vocab[c])
                else:
                    out[c] = pd.Categorical(x[c].fillna('__MISSING__').astype(str), categories=self.vocab[c]).codes.astype(np.float32)
            else:
                out[c] = x[c].astype('float32').fillna(-999.0) if self.family == 'catboost' else x[c].astype('float32')
        # Native boosters sanitize names to avoid forbidden JSON punctuation.
        out.columns = [f'f{i}' for i in range(len(out.columns))]
        return out

    def save(self, path: Path) -> None:
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path: Path) -> 'Transform':
        values = load_config(path)
        obj = cls(values['family'])
        obj.__dict__.update(values)
        return obj

class EncodedTransform:
    """Apply a persisted feature stage before downstream model preprocessing."""
    def __init__(self, base: Transform, encoder):
        self.base=base
        self.encoder=encoder
        self.family=base.family

    def transform(self,x:pd.DataFrame):
        return self.base.transform(self.encoder.transform(x))

def fit_model(x: pd.DataFrame, y: np.ndarray, valid: pd.DataFrame | None, yv: np.ndarray | None, run: dict, out: Path, rounds: int | None = None):
    out.mkdir(parents=True, exist_ok=True)
    family = run['family']
    profiles=None
    if run.get('route_profiles'):
        from route_profiles import RouteProfiles
        profile_config={} if run['route_profiles'] is True else run['route_profiles']
        profiles=RouteProfiles(profile_config)
        x=profiles.fit_transform(x)
        if valid is not None:valid=profiles.transform(valid)
    encoder=None
    if run.get('encoding'):
        from encoding import RouteEncoder
        encoder=RouteEncoder(run['encoding'])
        x=encoder.fit_transform(x,y)
        if valid is not None:valid=encoder.transform(valid)
    if family in {'tabm_cat','realmlp_cat'}:
        from categorical_transform import CategoricalTransform
        transform = CategoricalTransform(family).fit(x)
    else:
        transform = Transform(family).fit(x)
    xt = transform.transform(x)
    xv = transform.transform(valid) if valid is not None else None
    p = dict(run.get('params', {}))
    seed = int(run.get('seed', 20261001))
    limit = int(rounds or run.get('max_rounds', 1800))
    patience = int(run.get('patience', 120))
    if family == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=limit, objective='binary', n_jobs=8, random_state=seed, verbosity=-1, **p)
        kwargs = {} if xv is None else dict(eval_set=[(xv, yv)], eval_metric='auc', callbacks=[lgb.early_stopping(patience, first_metric_only=True), lgb.log_evaluation(100)])
        model.fit(xt, y, **kwargs)
        best = int(model.best_iteration_ or limit)
        model.booster_.save_model(str(out/'model.txt'))
    elif family == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=limit, loss_function='Logloss', eval_metric='AUC', task_type='GPU', devices='0', thread_count=8, random_seed=seed, allow_writing_files=False, verbose=200, **p)
        kwargs = {} if xv is None else dict(eval_set=(xv,yv), early_stopping_rounds=patience, use_best_model=True)
        model.fit(xt,y,cat_features=[i for i,c in enumerate(transform.columns) if c in transform.cats],**kwargs)
        best = int(model.tree_count_)
        model.save_model(str(out/'model.cbm'))
    elif family == 'xgboost':
        from xgboost import XGBClassifier
        extra = {} if xv is None else {'early_stopping_rounds':patience}
        model = XGBClassifier(n_estimators=limit, objective='binary:logistic', eval_metric='auc', device='cuda', tree_method='hist', n_jobs=8, random_state=seed, **p, **extra)
        model.fit(xt,y,eval_set=None if xv is None else [(xv,yv)],verbose=100)
        best = int(model.best_iteration+1) if xv is not None else limit
        model.save_model(out/'model.ubj')
    elif family == 'tabm':
        from neural import fit_neural
        p.update(seed=seed,epochs=limit)
        model,best = fit_neural(xt,y,xv,yv,p,out)
        model.save(out)
    elif family == 'realmlp':
        from realmlp import fit_realmlp
        p.update(seed=seed,epochs=limit)
        model,best=fit_realmlp(xt,y,xv,yv,p,out)
        model.save(out)
    elif family == 'realmlp_cat':
        from realmlp_categorical import fit_realmlp_categorical
        p.update(seed=seed,epochs=limit,categorical_indices=transform.categorical_indices)
        model,best=fit_realmlp_categorical(xt,y,xv,yv,p,out)
        model.save(out)
    elif family == 'tabm_cat':
        from neural_categorical import fit_neural
        p.update(seed=seed,epochs=limit,categorical_indices=transform.categorical_indices)
        model,best=fit_neural(xt,y,xv,yv,p,out)
        model.save(out)
    else:
        raise ValueError(f'Unknown family {family}')
    transform.save(out/'transform.json')
    if encoder is not None:
        encoder.save(out/'encoding')
        transform=EncodedTransform(transform,encoder)
    if profiles is not None:
        profiles.save(out/'route_profiles')
        transform=EncodedTransform(transform,profiles)
    atomic_json(out/'model_metadata.json', {'family':family, 'rounds':best, 'run':run})
    del xt,xv
    gc.collect()
    return model,transform,best

def load_model(path: Path):
    meta = load_config(path/'model_metadata.json')
    family = meta['family']
    if family == 'lightgbm':
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path/'model.txt'))
    elif family == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier().load_model(str(path/'model.cbm'))
    elif family == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(path/'model.ubj')
    elif family == 'tabm':
        from neural import load_neural
        model = load_neural(path)
    elif family == 'realmlp':
        from realmlp import load_realmlp
        model=load_realmlp(path)
    elif family == 'realmlp_cat':
        from realmlp_categorical import load_realmlp_categorical
        model=load_realmlp_categorical(path)
    elif family == 'tabm_cat':
        from neural_categorical import load_neural
        model=load_neural(path)
    else:
        raise ValueError(f'Unknown family {family}')
    if family in {'tabm_cat','realmlp_cat'}:
        from categorical_transform import CategoricalTransform
        transform=CategoricalTransform.load(path/'transform.json')
    else:
        transform=Transform.load(path/'transform.json')
    if (path/'encoding').is_dir():
        from encoding import RouteEncoder
        transform=EncodedTransform(transform,RouteEncoder.load(path/'encoding'))
    if (path/'route_profiles').is_dir():
        from route_profiles import RouteProfiles
        transform=EncodedTransform(transform,RouteProfiles.load(path/'route_profiles'))
    return model,transform

def predict(model, transform: Transform, x: pd.DataFrame) -> np.ndarray:
    result=[]
    for start in range(0,len(x),32768):
        z=transform.transform(x.iloc[start:start+32768])
        p=model.predict_proba(z)[:,1] if hasattr(model,'predict_proba') else model.predict(z)
        result.append(np.asarray(p,dtype=np.float64))
    p=np.concatenate(result)
    assert p.shape == (len(x),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p
