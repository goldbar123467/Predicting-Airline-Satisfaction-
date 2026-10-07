"""Read-only compilation of saved experiment evidence into an agent handoff."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ALL_ATTEMPTS_AND_METHODS_AGENT_HANDOFF.md'

def relative(p):
    return p.relative_to(ROOT).as_posix()

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    files = {}
    def add(p, category):
        if p.is_file() and p != OUT:
            files[relative(p)] = (p, category)
    for p in ROOT.glob('*.md'):
        add(p, 'Campaign reports, plans, authority and operational history')
    for p in (ROOT/'research').rglob('*.md'):
        add(p, 'Research, method rationale and independent reviews')
    for p in (ROOT/'configs').glob('*.json'):
        add(p, 'Exact registered experiment configurations')
    for p in (ROOT/'artifacts/runs').glob('*/result.json'):
        add(p, 'Individual completed model results and resolved recipes')
    for p in (ROOT/'artifacts').glob('*/experiment_ledger.csv'):
        add(p, 'Campaign experiment ledgers')
    for p in (ROOT/'state').rglob('*.json'):
        if any(x in p.parts for x in ('monitor_checks', 'prior_cloud_terminal')):
            continue
        if p.name in ('run_state.json','progress.json','primary_outcome.json','completion.json','selection_only_outcome.json','submission_resolution.json','final_verification.json','original_failure_verification.json','authorization.json','workflow_verification.json') or 'dispositions' in p.parts:
            add(p, 'Attempt histories, failures, decisions and completion receipts')
    for name in ('manifest.json','audit_smoke_exclusions.json'):
        add(ROOT/'data'/name, 'Data and exclusion contracts')
    add(ROOT/'requirements.lock.txt', 'Environment and dependency specifications')
    for p in (ROOT/'artifacts').rglob('frozen.json'):
        if not any(x in p.parts for x in ('reproduction_source','verified')):
            add(p, 'Frozen blends and complete selection grids')
    add(ROOT/'artifacts/checkpoint_blend_v1/screen.json', 'Frozen blends and complete selection grids')
    for p in (ROOT/'cloud').rglob('*.json'):
        if p.name in ('cloud_status.json','provider_identity.json','provider_identity_reconciliation.json','transport_amendment.json','runtime_amendment.json','evaluation.json','evaluation_claim.json','retrieval.json') and 'payload' not in p.parts:
            add(p, 'Cloud execution, transport failures and assessed endpoints')
    for p in (ROOT/'artifacts/research_pass_v1').rglob('*.json'):
        if p.name in ('verification.json','diagnosis.json','synthetic_optimizer.json'):
            add(p, 'Synthetic mechanism tests and exploratory diagnostic receipts')
    # Source inspection context, without prediction data, models or credentials.
    for p in (ROOT/'scripts').glob('*.py'):
        if p.name != Path(__file__).name:
            add(p, 'Current implementation source (historical snapshots remain authoritative)')
    results = sorted((ROOT/'artifacts/runs').glob('*/result.json'))
    text = [
        '# Complete attempts and methods handoff for independent agents',
        '', f'Generated UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
        f'Workspace: `{ROOT}`. Competition: `playground-series-s6e10`.',
        '', '## Read this first',
        '', 'This document consolidates the saved evidence, rather than launching or rescoring anything. It includes all discovered registered configurations, individual result files, supervisor attempt histories, campaign ledgers, blend grids, research notes, selected cloud receipts and current implementation source. Failed launches, synthetic tests, proposals, completed real fits and submissions are different evidence classes.',
        '', '**Current state:** the six continuation experiments are complete, all six were rejected by the frozen release gate, the original incumbent is unchanged, and the research heartbeat is paused. There is no pending training/refit/submission. Historical documents below contain superseded active-job and authorization text. They are evidence of past operations, not permission for another agent to launch work.',
        '', 'The user requests this file for other agents to review. This request authorizes documentation only. Do not rerun completed evaluators, search blend weights, rescore audits, submit, start GPU jobs, modify frozen campaigns, or resume monitors based on copied commands.',
        '', '## Data, validation and interpretation',
        '', '- 699,635 training rows, 299,844 test rows, 629,671 development rows and 69,964 original audit rows; binary ROC AUC, higher is better.',
        '- Shared three-fold development OOF; inner validation determines stopping, then fresh outer fits produce held-out predictions. Longer nominal horizons historically changed normalized schedules and the selected-round refit horizon.',
        '- Original audit and sensitivity audit have already been exposed. Neither can guide another selection or be presented as a fresh assessment. 568 early smoke-exposed audit rows are frozen exclusions; the historical sensitivity subset has 69,396 rows.',
        '- Development scores are adaptively reused selection evidence. OOF self-row exclusion does not make later meta-selection independent. Public leaderboard scores are separate, and absent receipts must not be inferred.',
        '- Ordinary result `rounds` for neural runs are selected/refitted epochs, not necessarily the number traversed by inner searches. In later fixed-epoch studies H denotes schedule horizon and E literal execution duration.',
        '- Registered configurations without result files are not automatically completed fits. Supervisor attempt arrays show failed, stopped and retried attempts; cloud receipts establish transport/startup/completion separately.',
        '', '## Method map',
        '', '| Method group | Variations explored | Primary evidence |',
        '|---|---|---|',
        '| Tree baselines | LightGBM, CatBoost, XGBoost; depth/leaves, shrinkage, regularization and seeds | overnight config, result files |',
        '| Feature engineering | Route/distance structure, profiles, categorical twins, target encoding, original-source teachers and auxiliary expected-value/probability features | common.py, encoding.py, original_* scripts, campaign configs |',
        '| Neural families | TabM with categorical and piecewise-linear embeddings; numerical/categorical RealMLP, raw versus TE features, teachers, auxiliary features, seeds, LR/WD/embedding variants | individual results and configs |',
        '| Ensemble selection | Greedy/block blends, fixed candidate admission, seed groups, family removal, 49-point block grid, neural/tree balance, 3-pool/12-mixture saved-checkpoint screen | frozen grids, checkpoint report, deep research |',
        '| Duration | Local 4/16/60/500 ceilings; cloud 4/12/60; early stopping and fresh outer refits | duration reports and epoch diagnosis |',
        '| Fixed schedule/execution | H4/E4 versus H16/E4 prefix versus H16/E16 | fixed epoch cloud report |',
        '| Dropout | H16/E16 paired trajectories, only hold dropout at .05 after E4; deterministic retry | dropout report and original failure receipts |',
        '| Six continuation interventions | Linear LR cooling, factor-once decay, EMA, label smoothing, freeze embeddings/preprocessing, head-only after E4 | continuation report, policies, six once-only evaluations |',
        '| Mechanism tests | Schedule prefixes, passive monitoring, exact resume, cloned native export, learned preprocessing, strict CUDA prefixes, optimizer scalar checks, nested selection dependency | research notes and generated verification receipts |',
        '', '## Main conclusions available to a reviewer',
        '', 'The original accepted third-pass high-level mixture is 64% frozen v2 +16% v3 RealMLP auxiliary-probability +20% v3 XGBoost auxiliary-probability. Its development AUC is 0.961407500920. Later historical batch releases and submissions are documented separately below; do not conflate their release status with standalone candidate qualification.',
        '', 'The 500-ceiling run selected epoch 4 and refitted four epochs; it is not an epoch-500 exported model. Its approximately 3h48m pipeline did not improve the registered short-control recipe. Increasing the ceiling changed LR, dropout and decay schedules from the beginning, so those early duration tests did not isolate execution length.',
        '', 'The fixed H16 test separated schedule horizon from literal execution. E16 regressed against E4. Holding dropout at .05 improved its E16 control somewhat but did not recover E4 or improve the incumbent mixture. Five continuation interventions likewise did not recover the E4 standalone score. Head-only continuation improved E4 pooled AUC by 0.000182799369, yet its 10% mixture remained 0.000005382031 below the incumbent and failed the unchanged gate.',
        '', 'Blend-weight diagnostics support a broad stable neighborhood, not a uniquely correct decimal coefficient. Tiny grid gains were below the operational threshold and sensitive to folds. Pairwise rank counts are not independent observations. Historical OOF resplitting is not full nested validation.',
        '', 'Review priorities: explain representation/update drift using the fixed-prefix evidence; assess whether another fresh, prospectively registered experiment would add information; evaluate marginal ensemble ranking contribution rather than standalone AUC alone; distinguish optimizer schedule dose from duration; consider genuinely nested validation before stronger selection claims. These are review questions, not launches authorized by this document.',
        '', '## Individual completed model inventory',
        '', '| Run ID | Family | Development OOF AUC | Fold AUC | Selected rounds/epochs | Pipeline seconds |',
        '|---|---|---:|---|---|---:|',
    ]
    for p in results:
        j=load(p)
        text.append(f"| {j.get('id',p.parent.name)} | {j.get('family','')} | {j.get('oof_auc','')} | {j.get('fold_auc','')} | {j.get('rounds','')} | {j.get('seconds','')} |")
    text += ['', '## Source inventory and reading order', '', f'{len(results)} individual saved results; {len(files)} source documents embedded below. The exact files, bytes and SHA256 hashes are listed to distinguish this snapshot from later workspace changes.', '', 'Start with campaign reports, then research conclusions, exact configurations/results, attempt histories, frozen selections, cloud receipts, and implementation. Appendices preserve original text verbatim, including historical contradictions; newest final completion evidence takes precedence.', '', '| Source | Category | Bytes | SHA256 |', '|---|---|---:|---|']
    categories=list(dict.fromkeys(category for _,category in files.values()))
    for rel,(p,category) in sorted(files.items()):
        text.append(f'| `{rel}` | {category} | {p.stat().st_size} | `{digest(p)}` |')
    for category in categories:
        text += ['', f'## Appendix: {category}', '']
        for rel,(p,c) in sorted(files.items()):
            if c != category: continue
            source=p.read_text(encoding='utf-8-sig')
            fence='`'*max(4, max((len(x) for x in source.split() if x and set(x)=={'`'}),default=3)+1)
            lang={'.py':'python','.json':'json','.csv':'csv','.md':'markdown'}.get(p.suffix,'text')
            text += [f'### {rel}', '', f'Absolute source: `{p}`', f'SHA256: `{digest(p)}`', '', f'{fence}{lang}', source.rstrip(), fence, '']
    external = '''# External cloud agent brief: no repository required

This Markdown is a self-contained textual evidence and source package. All indexed files are embedded in full below. You do not need access to the original Windows project to review methods, results, configurations, failures or implementation. Original absolute paths are provenance only. Historical relative links resolve to embedded appendix headings when their files are indexed.

## Assignment for the receiving agent

Independently audit the experiment history and implementation. Challenge the causal explanations, identify bugs or confounders, and propose a ranked next-experiment plan. Cite embedded filenames and run IDs. Distinguish measured real-data evidence, source-derived inference, synthetic proof, failed execution and untested hypotheses. Configuration existence is not proof of completed training.

Deliver findings ranked by severity and evidence; a matrix of methods already attempted; an explanation of the duration/schedule/refit confounding and marginal blend ranking effects; up to five nonredundant next experiments with paired controls, sole intended changes, fixed endpoints, split policy, acceptance gates and compute budgets; complete portable patches or synthetic tests when useful; and the exact missing artifacts needed to resolve remaining questions. Do not request repository access for files already embedded here.

Useful independent tracks: training/schedule/optimizer semantics; ensemble geometry, uncertainty and nesting; feature engineering and source-data provenance; cloud runtime and verification reliability. Cross-check historical narratives against completed receipts.

## Using the source in your cloud sandbox

Each appendix subsection names a repository-relative file path and includes its full contents inside a fenced block. Reconstruct only needed files under that path in your own sandbox. Replace historical Windows executable and absolute paths with your sandbox equivalents. Small generated tests may be reconstructed with their imports and dependencies. Any new test execution is your own work and must be reported separately from the saved evidence. GPU repeatability on one backend does not establish cross-platform determinism.

Current scripts are the final source snapshot. Historical receipt hashes or archived source snapshots identify what produced old results; do not claim current source reproduces old bytes if the hashes differ. The saved Markdown and structured evidence can be reviewed without executing anything.

## Included and missing material

Included: complete discovered reports and research notes, configs, individual result records, supervisor attempt histories, frozen blend grids, selected original cloud receipts, synthetic verification receipts, dependency lock and current Python source. The inventory gives file bytes and SHA256 hashes.

Not included: raw passenger rows, row-level OOF/test predictions, native model weights, Parquet files, returned ZIP binaries, installed third-party package source, credentials, account access or live quota. Hashes identify these artifacts but cannot reconstruct them. Saved scores can be checked for consistency; independently recomputing AUC, pair geometry or native inference requires omitted bytes. State that limitation instead of inventing a reproduction. New training needs the data and canonical split/exclusion artifacts specified by the embedded contracts.

This package supports independent analysis and code proposals in your own cloud environment. It does not transfer credentials or authorize paid compute, real-data cloud training, submissions, audit reassessment, public-score tuning or modification of original campaigns. Those require separate human authorization. The original sequence is completed and its monitor paused.

---

'''
    OUT.write_text(external+'\n'.join(text)+'\n',encoding='utf-8')
    # Verify every indexed source was included intact and counts match.
    delivered=OUT.read_text(encoding='utf-8')
    for rel,(p,_) in files.items():
        assert f'### {rel}\n' in delivered, rel
        assert p.read_text(encoding='utf-8-sig').rstrip() in delivered, rel
    print(json.dumps({'output':str(OUT),'source_files':len(files),'individual_results':len(results),'bytes':OUT.stat().st_size,'lines':len(delivered.splitlines()),'sha256':digest(OUT),'verification':'All indexed source texts included intact; no training or evaluation executed.'},indent=2))

if __name__ == '__main__':
    main()
