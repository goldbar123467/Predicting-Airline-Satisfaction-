"""Bounded real-data integration checks for compact categorical neural models."""
from pathlib import Path
import argparse
import gc
import tempfile
import numpy as np
import pandas as pd
from common import ROOT,features,fit_model,load_model,predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['realmlp_cat','tabm_cat'])
    args=parser.parse_args()
    all_rows=pd.read_parquet(ROOT/'data/train.parquet')
    splits=pd.read_parquet(ROOT/'data/splits.parquet')
    assert np.array_equal(all_rows.id,splits.id)
    train=all_rows.loc[splits.fold>=0].sample(1024,random_state=42)
    del all_rows,splits
    test=pd.read_parquet(ROOT/'data/test.parquet').iloc[:129]
    raw=pd.read_csv(ROOT/'data/test.csv',nrows=129)
    run={'id':'integration','categorical_twins':True,'teacher':True,
         'encoding':{'keys':['Flight Distance','Age'],'smoothing':20,
                     'n_splits':3,'seed':42,'counts':True},
         'seed':42,'max_rounds':1}
    pd.testing.assert_frame_equal(features(test,run),features(raw,run),check_dtype=False)
    x=features(train,run); xt=features(test,run)
    y=train.satisfaction.to_numpy()
    for family,params in [
        ('realmlp_cat',{'device':'cpu','n_ens':2,'hidden_sizes':[32,16],
                        'batch_size':128,'eval_batch_size':128,'threads':2}),
        ('tabm_cat',{'device':'cpu','k':2,'d_block':32,'n_blocks':1,
                     'batch_size':128,'eval_batch_size':128,'threads':2,
                     'num_embeddings':True,'n_bins':8,'d_embedding':4})]:
        if args.family and family!=args.family:continue
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            model,transform,rounds=fit_model(x.iloc[:900],y[:900],x.iloc[900:],y[900:],
                                           {**run,'family':family,'params':params},out)
            assert rounds==1
            before=predict(model,transform,xt)
            loaded,restored=load_model(out)
            after=predict(loaded,restored,xt)
            np.testing.assert_allclose(before,after,rtol=1e-5,atol=2e-6)
            # A validation-only category cannot expand the saved training vocabulary.
            unknown=xt.copy(); unknown['Gender']='NEW_CATEGORY'
            z=restored.transform(unknown)
            base=restored.base
            position=len(base.nums)+base.cats.index('Gender')
            assert (z[:,position]==-1).all()
            assert np.isfinite(predict(loaded,restored,unknown)).all()
            print(f'PASS {family}: real-data features, nested encoding, unknowns, native reload',flush=True)
            del model,transform,loaded,restored
            gc.collect()


if __name__=='__main__':main()
