<div align="center">

# FASTER: Value-Guided Sampling for Fast RL

[Perry Dong](https://scholar.google.com/citations?user=3Eu7CagAAAAJ)<sup>\*</sup> &nbsp;·&nbsp;
[Alexander Swerdlow](https://aswerdlow.com/)<sup>\*</sup> &nbsp;·&nbsp;
[Dorsa Sadigh](https://dorsa.fyi/) &nbsp;·&nbsp;
[Chelsea Finn](https://ai.stanford.edu/~cbfinn/)

Stanford University

<sub><sup>*</sup>Equal contribution</sub>

<a href="https://arxiv.org/abs/2604.19730"><img src="https://img.shields.io/badge/arXiv-2604.19730-b31b1b?logo=arxiv&logoColor=white"></a>
<a href="https://pd-perry.github.io/faster/"><img src="https://img.shields.io/badge/Website-FASTER-2f80ed"></a>


<img src="assets/method.png" alt="FASTER overview" width="95%"/>

</div>

---

## Overview

Many of the strongest RL algorithms today rely on **best-of-N action sampling** with a value critic — they pay to fully denoise *N* candidates and keep only one. **FASTER** recovers the gains of best-of-N without the same sampling cost.

FASTER frames best-of-N denoising as a Markov Decision Process over the diffusion trajectory and learns a **denoise critic** that scores candidates *before* denoising completes. At inference time we sample *N* noise seeds, rank them with the critic, and fully denoise only the top-ranked seed — collapsing inference cost to a single rollout regardless of *N*.

## Setup

```bash
uv sync
source .env && python scripts/download_robomimic_datasets.py
```

The code expects the Robomimic low-dim datasets `low_dim_v141.hdf5` in `$ROBOMIMIC_DATASETS_PATH`, which defaults to `./datasets/robomimic`.

<details>
<summary><b>Short sanity run of FASTER-EXPO online to check setup</b></summary>

```bash
source .env && WANDB_MODE=offline python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=BetterDiffusionSACLearner \
  --log_dir=exp \
  --env_name=can \
  --eval_interval=1 \
  --eval_episodes=2 \
  --start_training=1 \
  --max_steps=2
```
</details>

## Training Commands

Please see the following scripts for the different training settings.

| Setting | Task | Script |
| --- | --- | --- |
| FASTER-EXPO online | can | [`scripts/faster_expo_online_can.sh`](scripts/faster_expo_online_can.sh) |
| FASTER-EXPO online | lift | [`scripts/faster_expo_online_lift.sh`](scripts/faster_expo_online_lift.sh) |
| FASTER-EXPO online | square | [`scripts/faster_expo_online_square.sh`](scripts/faster_expo_online_square.sh) |
| FASTER-EXPO online | tool_hang | [`scripts/faster_expo_online_tool_hang.sh`](scripts/faster_expo_online_tool_hang.sh) |
| FASTER-IDQL online | can | [`scripts/faster_idql_online_can.sh`](scripts/faster_idql_online_can.sh) |
| FASTER-IDQL online | lift | [`scripts/faster_idql_online_lift.sh`](scripts/faster_idql_online_lift.sh) |
| FASTER-IDQL online | square | [`scripts/faster_idql_online_square.sh`](scripts/faster_idql_online_square.sh) |
| FASTER-EXPO batch-online | can | [`scripts/faster_expo_batch_online_can.sh`](scripts/faster_expo_batch_online_can.sh) |
| FASTER-EXPO batch-online | lift | [`scripts/faster_expo_batch_online_lift.sh`](scripts/faster_expo_batch_online_lift.sh) |
| FASTER-EXPO batch-online | square | [`scripts/faster_expo_batch_online_square.sh`](scripts/faster_expo_batch_online_square.sh) |

Each script accepts extra CLI overrides via `$@`. For example:

```bash
bash scripts/faster_expo_online_can.sh --log_dir=exp --seed=0
```

## Outputs

Each run creates a dir under `--log_dir` which defaults to `./exp`, e.g.:

```text
exp/2026_04_17__12_34_56__s0/
├── flags.json
├── train.csv
├── eval.csv
├── checkpoints/   # if --checkpoint_model=True
└── buffers/       # if --checkpoint_buffer=True
```

## Acknowledgements

The training infrastructure builds on [RLPD](https://github.com/ikostrikov/rlpd), [IDQL](https://github.com/philippe-eecs/IDQL), and [Robomimic](https://robomimic.github.io/). We thank the authors for their open-source releases.

## Citation

```bibtex
@article{dong2026faster,
  title   = {FASTER: Value-Guided Sampling for Fast RL},
  author  = {Dong, Perry and Swerdlow, Alexander and Sadigh, Dorsa and Finn, Chelsea},
  journal = {arXiv preprint arXiv:2604.19730},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.19730}
}
```
