"""Fold-local compact numeric/category matrices for native categorical networks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


class CategoricalTransform:
    def __init__(self,family:str):
        self.family=family

    def fit(self,x:pd.DataFrame):
        self.columns=list(x.columns)
        self.cats=[c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        self.nums=[c for c in x if c not in self.cats]
        self.vocab={c:sorted(x[c].fillna('__MISSING__').astype(str).unique().tolist()) for c in self.cats}
        self.median={c:float(x[c].median()) if x[c].notna().any() else 0. for c in self.nums}
        self.mean={c:float(x[c].fillna(self.median[c]).mean()) for c in self.nums}
        self.std={}
        for c in self.nums:
            value=float(x[c].fillna(self.median[c]).std())
            self.std[c]=value if np.isfinite(value) and value>1e-6 else 1.
        self.categorical_indices=list(range(len(self.nums),len(self.nums)+len(self.cats)))
        return self

    def transform(self,x:pd.DataFrame)->np.ndarray:
        if list(x.columns)!=self.columns:raise ValueError('Categorical network feature schema/order changed')
        columns=[(x[c].fillna(self.median[c]).to_numpy(dtype=np.float32)-self.mean[c])/self.std[c] for c in self.nums]
        for c in self.cats:
            # Unknown strings stay -1; the network's train-only mapper reserves native index0.
            codes=pd.Index(self.vocab[c]).get_indexer(x[c].fillna('__MISSING__').astype(str))
            columns.append(codes.astype(np.float32))
        result=np.ascontiguousarray(np.column_stack(columns),dtype=np.float32)
        if not np.isfinite(result).all():raise ValueError('Non-finite categorical network matrix')
        return result

    def save(self,path:Path)->None:
        payload={**self.__dict__,'transform_type':'compact_categorical','schema_version':1}
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temp,path)

    @classmethod
    def load(cls,path:Path):
        values=json.loads(path.read_text(encoding='utf-8'))
        if values.pop('transform_type')!='compact_categorical' or values.pop('schema_version')!=1:
            raise ValueError('Unsupported categorical transform')
        result=cls(values['family']);result.__dict__.update(values)
        return result


if __name__=='__main__':
    import tempfile
    frame=pd.DataFrame({'number':[1.,2.,np.nan,4.],'category':['a','b','a','a']})
    held=pd.DataFrame({'number':[999.],'category':['unseen']})
    transform=CategoricalTransform('tabm_cat').fit(frame)
    assert transform.transform(held)[0,-1]==-1
    assert transform.mean['number']<4 and transform.categorical_indices==[1]
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'transform.json';transform.save(path)
        assert np.array_equal(transform.transform(frame),CategoricalTransform.load(path).transform(frame))
    print('PASS compact categorical train-only transform, unknown category and persistence')
