"""Bounded, generated-data-only native reload diagnosis of preserved smoke graph."""
import argparse
import faulthandler
import json
import time
from pathlib import Path

import torch

from smoke_fixed_epoch_v1 import build_features, generated_frames
from categorical_transform import CategoricalTransform
from realmlp_categorical import load_realmlp_categorical
from run_fixed_epoch_v1 import ROOT, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unoptimized', action='store_true')
    args = parser.parse_args()
    faulthandler.dump_traceback_later(20, repeat=True)
    torch.set_num_threads(4)
    started = time.monotonic()

    def mark(stage):
        print(json.dumps({'stage': stage, 'seconds': time.monotonic()-started,
                          'threads': torch.get_num_threads()}), flush=True)

    recipe = next(x for x in read(ROOT/'configs/second_pass.json')['runs']
                  if x['id'] == 'v2_realmlp_cat_raw_aux')
    frame, bank = generated_frames()
    features = build_features(frame, bank, recipe)
    folder = ROOT/'artifacts/fixed_epoch_v1/smoke_cuda_01/inner_C'
    transform = CategoricalTransform.load(folder/'transform.json')
    values = transform.transform(features.iloc[384:])
    mark('features_ready')
    model = load_realmlp_categorical(folder/'epoch_004', device='cuda')
    mark('loaded')
    with torch.jit.optimized_execution(not args.unoptimized):
        for length in (37,37,17,128):
            prediction = model.predict_proba(values[:length])
            mark(f'predicted_{length}')
            print(json.dumps({'range':[float(prediction.min()),float(prediction.max())]}),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
