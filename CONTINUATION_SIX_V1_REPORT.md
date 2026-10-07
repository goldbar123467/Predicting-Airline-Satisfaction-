# Six continuation experiments: completed

All six preregistered tests completed and failed the unchanged release gate. The original blend and release bytes remain unchanged. No full-data refit, release promotion or new submission was performed.

| Treatment | E16 candidate AUC | Fixed 10% blend AUC |
|---|---:|---:|
| Linear LR cooling | 0.954950663400 | 0.961359425156 |
| Decay correction | 0.956581534886 | 0.961368192946 |
| EMA export | 0.956852903351 | 0.961393572703 |
| Label smoothing | 0.955330500515 | 0.961375549400 |
| Frozen embeddings | 0.955592833519 | 0.961343525143 |
| Head-only continuation | 0.960728931595 | 0.961402118889 |

Incumbent development AUC: 0.961407500920. Shared epoch-4 AUC: 0.960546132226. Newly fitted E16 controls: 0.955469094940.

Head-only continuation was the only intervention to improve on the epoch-4 endpoint: +0.000182799369 pooled AUC. Its fixed blend remained below the incumbent by 0.000005382031, and its mean-fold gain over the epoch-4 mixture was below the required 0.00001. This supports further investigation of representation drift, but does not establish its cause or authorize another experiment.

Each stage passed complete generated CUDA smoke, 12 new paired fits/192 epochs, nine outer endpoints, six exact full-state epoch-4 prefix gates, three full-fold native-prefix checks and fourteen deterministic startup roles. Total real-data fitting: 72 fits/1152 epochs. Six isolated frozen assessments ran once, after source/input/archive/member/native/partition/order and actual optimizer schedule verification. Three independent agent reviews and 158 generated checks preceded the sequence.

Each experiment used a private immutable Kaggle version1 with one cloud GPU job at a time. Local work was artifact verification and CPU assessment; no local GPU fits. Original returned status, manifest, archive and logs are preserved in each stage experiment/download_01. Exact hashes and exclusive dispositions are bound in state/continuation_six_v1/final_verification.json.

Sequence wall interval: October4 21:21UTC first dispatch through 2026-10-05T08:13:52.852589+00:00 final closure, approximately 10 hours 52 minutes. This includes provider execution, scheduled monitoring gaps, retrieval and assessment; it is not a claim of continuous human or agent research time. All work closed before the October5 21:02:37UTC deadline.

These are exploratory comparisons on historically reused development OOF data, not independent confirmation or evidence of statistical significance. No audit rescoring, alpha search, public-score tuning or historical control reuse was performed. The fixed mixture was always 90% original incumbent plus 10% candidate.

Reproducible commands: continuation_six_cloud_v1.py status/retrieve --stage fixed_epoch_continuation_0N --phase experiment; evaluate_continuation_six_v1.py with the corresponding isolated assessment_workspace and already-frozen local_evaluation_protocol.json; continuation_six_selection_v1.py --stage fixed_epoch_continuation_0N. Assessments are complete and must not be rerun.

No further training is pending. Pause the same research heartbeat; older monitors remain paused. Requested GPT-6.1 Sol/Low setting was not verified because the heartbeat tool does not accept model settings.
