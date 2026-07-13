# SOTA Task Scheduling Algorithms for MetaSpiderMark

This document lists the task scheduling / task sampling algorithms that should
be considered for the MetaSpiderMark scheduler benchmark. The scope here is not
general meta-learning algorithms such as MAML, ANIL, Reptile, ProtoNet, or R2D2.
Those belong in the meta-learning algorithm benchmark. This file is only about
which meta-training tasks are selected, weighted, or ordered.

## Benchmark Position

For the paper, the scheduler benchmark should answer:

> Under the same SpiderMark verifier, same meta-learning update, same support
> and query construction, same attack pool, same training budget, and same
> evaluation suite, which task scheduler gives the strongest downstream
> watermark verifier?

The scheduler table should therefore fix the meta-learning algorithm and vary
only the scheduler. A fair first version can use one strong meta-learning update
as the backbone, then run the schedulers below under the same seed and compute
budget. After the seed-0 ranking is clear, only the strongest rows should get
extra seeds.

## Recommended Main Scheduler Set

These are the methods I would prioritize for the main table.

| Priority | Method | Paper | Venue / Year | Scheduler family | Why include it | Practical implementation target |
|---|---|---|---|---|---|---|
| Required anchor | Uniform Sampling | Common episodic meta-learning baseline; used as baseline in MAML and many later papers | canonical baseline | random sampling | Necessary sanity anchor. It tells us whether scheduling matters at all. | Already available as `uniform`; one seed is enough initially. |
| Required | ATS | Huaxiu Yao et al., "Meta-learning with an Adaptive Task Scheduler" | NeurIPS 2021 | learned adaptive scheduler | Core learned scheduler baseline before BASS. It predicts task sampling probabilities from meta-model-related task difficulty factors. | Implement local ATS-style scheduler; cite as inspired unless reproducing official details exactly. |
| Required | BASS | Yunzhe Qi et al., "Meta-Learning with Neural Bandit Scheduler" | NeurIPS 2023 | contextual neural bandit scheduler | Strongest direct predecessor for scheduler benchmarking. BASS formulates meta-learning task scheduling as contextual bandits and explicitly balances exploration/exploitation. | Implement BASS-style neural contextual bandit row; if local implementation differs, label it clearly as BASS-inspired. |
| Required | Bandit-UCB | Based on contextual bandit / UCB scheduling family; BASS uses neural bandit ideas | classical bandit baseline | non-neural bandit scheduler | Useful ablation between random scheduling and neural/adaptive scheduling. It tests whether simple exploration already explains the gain. | Already available as `bandit_ucb`; treat as local bandit baseline, not a named paper reproduction. |
| Required if feasible | DERTS | Donglin Zhan and James Anderson, "Data-Efficient and Robust Task Selection for Meta-Learning" | CVPR Workshop 2024 | gradient-approximation task subset selection | Most relevant post-BASS recent method. It selects weighted task subsets by minimizing approximation error to the full task-pool gradient and is designed for noisy/limited data settings. | Implement as official DERTS if possible; otherwise keep `derts_proxy` exploratory and do not put it in the main SOTA table as official DERTS. |
| Required | ASr / Adaptive Sampler | Jingyao Wang et al., "Towards Task Sampler Learning for Meta-Learning" | IJCV 2024; arXiv 2023/2024 | learned task sampler | Recent task sampler that weights tasks using task diversity, entropy, and difficulty. It is plug-and-play and directly targets meta-training task sampling. | Implemented as local `asr` ASr-style scheduler using online diversity/entropy/difficulty proxies. |
| Proposed method | MetaSpiderMark scheduler / residual scheduler | This paper | target paper | proposed adaptive scheduler | This is the method we want to compare against the external scheduler baselines. | Use the current local residual / LLM-derived scheduler row, with exact naming decided in the paper. |

## BASS Table Baselines

BASS compares against seven baselines plus itself. These are important because
they define the expected scheduler-baseline ecosystem for this literature.

| Method | Paper | Venue / Year | What it is | Include in our benchmark? |
|---|---|---|---|---|
| Uniform Sampling | Canonical meta-training task sampling baseline | standard baseline | Randomly sample tasks uniformly. | Yes, but only as a sanity anchor. |
| SPL | M. Kumar, Benjamin Packer, Daphne Koller, "Self-Paced Learning for Latent Variable Models" | NeurIPS 2010 | Easy-to-hard self-paced weighting. BASS adapts it as a non-adaptive task scheduling baseline. | Optional. Useful for completeness, but less important than ATS/BASS/DERTS/ASr. |
| FOCAL | Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection" | ICCV 2017 | Reweights hard examples; BASS adapts the idea to task scheduling. | Optional. It is not originally a meta-learning scheduler, but BASS uses it as a simple hard-task weighting baseline. |
| DAML | Xiaomeng Li et al., "Difficulty-Aware Meta-Learning for Rare Disease Diagnosis" | MICCAI 2020 | Difficulty-aware meta-learning scheduler. | Good optional baseline if easy to implement. Domain is medical classification, so adaptation details matter. |
| GCP | Chenghao Liu et al., "Adaptive Task Sampling for Meta-Learning" | ECCV 2020 | Greedy class-pair based adaptive task sampling. | Optional appendix row as local `gcp_proxy`. It is classification-oriented and does not map cleanly to all watermark attacks. |
| PAML | Jean Kaddour and Steindor Saemundsson et al., "Probabilistic Active Meta-Learning" | NeurIPS 2020 | Active task selection based on uncertainty / informativeness. | Good optional baseline; relevant because it is a meta-learning task-selection method. |
| ATS | Huaxiu Yao et al., "Meta-learning with an Adaptive Task Scheduler" | NeurIPS 2021 | Learned task scheduler using task difficulty factors. | Yes, required. |
| BASS | Yunzhe Qi et al., "Meta-Learning with Neural Bandit Scheduler" | NeurIPS 2023 | Neural contextual bandit task scheduler. | Yes, required. |

## Recent Candidates After BASS

These are the methods worth tracking beyond the original BASS baseline table.

| Method | Paper | Venue / Year | Why it matters | Recommendation |
|---|---|---|---|---|
| DERTS | Donglin Zhan and James Anderson, "Data-Efficient and Robust Task Selection for Meta-Learning" | CVPR Workshop 2024 | Directly targets noisy and limited-data task selection, which matches our downstream watermark robustness setting. | High priority. Add as official row if we can implement the gradient subset objective correctly. |
| ASr / Adaptive Sampler | Jingyao Wang et al., "Towards Task Sampler Learning for Meta-Learning" | IJCV 2024 | Recent plug-and-play task sampler using diversity, entropy, and difficulty. | High priority after BASS/ATS; local `asr` row is implemented. |
| ATSVR | Zhuoqun Liu et al., "Adaptive Task Sampling and Variance Reduction for Gradient-Based Meta-Learning" | BMVC 2022 | Adjusts task sampling distribution and uses variance reduction for gradient-based meta-learning. | Medium priority. Good if our backbone is gradient-based and we want a stronger non-bandit adaptive sampling baseline. |
| MI / information-theoretic task selection | Ricardo Luna Gutierrez and Matteo Leonetti, "Information-Theoretic Task Selection for Meta-Reinforcement Learning" | NeurIPS 2020 | Selects tasks by information content. Related but meta-RL oriented. | Low to medium priority; include only if adaptation is straightforward. |
| ATU | "Adversarial Task Up-sampling for Meta-learning" | NeurIPS 2022 | Upweights adversarial/difficult tasks during meta-training. | Medium priority if we want a hard-task curriculum comparator. |

## Suggested Main Table

The first publishable scheduler table should be compact:

| Row | Scheduler | Status |
|---|---|---|
| 1 | Uniform | sanity anchor |
| 2 | Bandit-UCB | simple bandit baseline |
| 3 | ATS-style | learned adaptive scheduler |
| 4 | BASS-style | neural bandit scheduler |
| 5 | DERTS or DERTS-proxy | recent 2024 task selection; official only if implemented faithfully |
| 6 | ASr-style | recent IJCV 2024 adaptive sampler |
| 7 | MetaSpiderMark proposed scheduler | proposed method |

Then the appendix can add SPL, FOCAL, DAML, GCP, PAML if we implement them
cleanly. They are useful for matching the BASS experimental ecosystem, but they
should not delay the main comparison against ATS, BASS, DERTS, ASr, and our
proposed scheduler.

## Recommended Paper Result Tables

My recommendation is to use one main scheduler table, one efficiency table, and
one compact per-attack robustness table. The main table should not include every
historical baseline from BASS, because that will consume compute and make the
paper less focused. The main comparison should be against the strongest and most
relevant scheduler families.

### Main Table: Scheduler Robustness

This should be the main result table in the paper.

Fixed across all rows:

- same SpiderMark verifier architecture
- same meta-learning algorithm / update rule
- same support/query construction
- same attack-task pool
- same number of meta-training steps
- same checkpoint selection rule
- same downstream evaluation attacks
- same seed set, after seed-0 ranking is clear

Recommended rows:

| Method | Source | Type | Mean Acc. | Worst Acc. | Mean AUROC | Worst AUROC | Mean F1 | Robustness Gain vs Uniform | Rank |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | canonical | random task sampling |  |  |  |  |  |  |  |
| Bandit-UCB | local bandit baseline | non-neural exploration |  |  |  |  |  |  |  |
| ATS-style | Yao et al., NeurIPS 2021 | learned adaptive scheduler |  |  |  |  |  |  |  |
| BASS-style | Qi et al., NeurIPS 2023 | neural contextual bandit |  |  |  |  |  |  |  |
| DERTS / DERTS-proxy | Zhan and Anderson, CVPRW 2024 | gradient subset selection |  |  |  |  |  |  |  |
| ASr-style | Wang et al., IJCV 2024 | diversity/entropy/difficulty sampler |  |  |  |  |  |  |  |
| MetaSpiderMark scheduler | this paper | proposed scheduler |  |  |  |  |  |  |  |

Primary metric definitions:

- `Mean Acc.`: average fixed-threshold watermark detection accuracy across all
  downstream attacks.
- `Worst Acc.`: minimum fixed-threshold accuracy over attacks. This is important
  because watermark robustness should not collapse under one attack.
- `Mean AUROC`: average threshold-free detection quality across attacks.
- `Worst AUROC`: minimum AUROC over attacks.
- `Mean F1`: average F1 at the selected operating threshold.
- `Robustness Gain vs Uniform`: `Mean AUROC(method) - Mean AUROC(uniform)` or
  `Mean Acc(method) - Mean Acc(uniform)`. Use one consistently.
- `Rank`: average rank across `Mean Acc.`, `Worst Acc.`, `Mean AUROC`, and
  `Worst AUROC`.

If the table becomes too wide, keep only:

| Method | Mean Acc. | Worst Acc. | Mean AUROC | Worst AUROC | Gain vs Uniform | Rank |
|---|---:|---:|---:|---:|---:|---:|

### Secondary Table: Per-Attack Robustness

This table should show where each scheduler wins or fails. It can go in the
main paper if space allows; otherwise it goes in the appendix.

| Method | Clean | JPEG | Resize | Blur | Crop | Occlusion | Geom. Warp | Aug. Mix | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform |  |  |  |  |  |  |  |  |  |  |
| ATS-style |  |  |  |  |  |  |  |  |  |  |
| BASS-style |  |  |  |  |  |  |  |  |  |  |
| DERTS / ASr |  |  |  |  |  |  |  |  |  |  |
| MetaSpiderMark scheduler |  |  |  |  |  |  |  |  |  |  |

Use AUROC here if the threshold is unstable; use fixed-threshold accuracy if
the operating threshold is part of the verifier claim.

### Efficiency Table

This table protects the paper from the objection that a scheduler only wins
because it uses much more compute.

| Method | Train steps | Wall-clock time | Scheduler overhead | GPU memory | Checkpoints evaluated | Mean AUROC | AUROC / hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform |  |  |  |  |  |  |  |
| Bandit-UCB |  |  |  |  |  |  |  |
| ATS-style |  |  |  |  |  |  |  |
| BASS-style |  |  |  |  |  |  |  |
| DERTS / ASr |  |  |  |  |  |  |  |
| MetaSpiderMark scheduler |  |  |  |  |  |  |  |

## Final Baseline Recommendation

The main paper should compare against these schedulers:

1. `uniform`: mandatory sanity anchor.
2. `bandit_ucb`: simple exploration baseline; cheap and interpretable.
3. `ats`: required learned adaptive scheduler baseline.
4. `bass`: required SOTA neural bandit baseline.
5. `derts` or `derts_proxy`: recent 2024 robust task-selection baseline.
6. `asr`: recent IJCV 2024 adaptive sampler.
7. our proposed MetaSpiderMark scheduler.

If compute or implementation time is limited, use this minimum publishable set:

1. `uniform`
2. `bandit_ucb`
3. `ats`
4. `bass`
5. our proposed MetaSpiderMark scheduler

Then add DERTS and ASr as the strongest recent baselines. SPL, FOCAL, DAML,
GCP, and PAML are useful for matching the BASS table, but I would treat them as
appendix baselines unless implementation becomes easy.

## Code Availability Notes

Based on the initial source check:

- BASS has an apparent author GitHub implementation:
  https://github.com/yunzhe0306/Bandit_Task_Scheduler
- ASr / Adaptive Sampler has an apparent official GitHub implementation:
  https://github.com/WangJingyao07/Adaptive-Sampler
- GCP / Adaptive Task Sampling has a public implementation:
  https://github.com/ptkin/gcp-sampling
- DERTS has a paper and CVPRW PDF, but I did not find an obvious official
  GitHub repository in the initial search.
- ATS has OpenReview / arXiv sources, but I did not find an obvious official
  GitHub repository in the initial search.

Local implementation status:

- `asr` is implemented as an ASr-style online attack-task scheduler. It adapts
  the external Adaptive-Sampler repo's diversity/entropy/difficulty idea to our
  task-level feedback statistics.
- `gcp_proxy` is implemented as an optional appendix row. It adapts GCP's
  exponential class-weight update to attack-task weights, so it should be
  described as GCP-style / proxy rather than official GCP.
- `bass` remains BASS-style. The official BASS repo uses neural exploit and
  exploration networks over meta-parameter contexts; the current local row keeps
  the same contextual-bandit motivation but uses the lightweight online
  statistics available in our harness.

Practical policy:

- If official code exists, use it as a reference for scheduler logic, but adapt
  it into our training/evaluation harness rather than replacing the whole
  codebase.
- If official code does not exist, implement the scheduler from the paper and
  label it as `method-style` unless the objective and features match exactly.
- Do not copy a full external training framework into the repo unless the
  license and dependencies are clear. Prefer small, auditable scheduler modules.

## Implementation Notes for Our Codebase

- `uniform` is an anchor, not a competitive SOTA method.
- `bandit_ucb` is a useful local bandit baseline, but should not be cited as
  BASS.
- `ats` should be described as ATS-style unless the exact ATS training objective
  and features are reproduced.
- `bass` should be described as BASS-style unless the neural contextual bandit
  training, reward definition, and exploration module match the paper.
- `derts_proxy` is exploratory. It should not be labeled as DERTS in the paper
  unless we implement the weighted task subset selection objective from DERTS.
- `residual` or the local MetaSpiderMark scheduler should be the proposed row
  once its final name and mechanism are fixed.

## Source Links

- BASS: "Meta-Learning with Neural Bandit Scheduler", NeurIPS 2023:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/c9e6ac15e689e06139d7b39e1667b165-Paper-Conference.pdf
- ATS: "Meta-learning with an Adaptive Task Scheduler", NeurIPS 2021:
  https://openreview.net/forum?id=MTs2adH_Qq
- DERTS: "Data-Efficient and Robust Task Selection for Meta-Learning", CVPR
  Workshop 2024:
  https://arxiv.org/abs/2405.07083
- ASr: "Towards Task Sampler Learning for Meta-Learning", IJCV 2024:
  https://arxiv.org/abs/2307.08924
- GCP / Adaptive Task Sampling: "Adaptive Task Sampling for Meta-Learning",
  ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630732.pdf
- PAML: "Probabilistic Active Meta-Learning", NeurIPS 2020:
  https://arxiv.org/abs/2007.08949
- ATSVR: "Adaptive Task Sampling and Variance Reduction for Gradient-Based
  Meta-Learning", BMVC 2022:
  https://bmvc2022.mpi-inf.mpg.de/0876.pdf
