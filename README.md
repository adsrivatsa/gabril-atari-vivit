# gabril-atari-vivit

Gaze-regularized behavior cloning on Atari, with a ViViT encoder instead of the CNN in the original [GABRIL](https://liralab.usc.edu/gabril/) paper. The question is whether the gaze loss still helps when the encoder is a transformer, as a step toward gaze-regularized imitation for web agents.

This page is the quickstart. The full state of the project is in [docs/handoff.md](docs/handoff.md).

## Quickstart

```bash
uv venv --python 3.11
uv pip install -r pyproject.toml
wandb login
```

Download the Atari gaze dataset from the [GABRIL project page](https://liralab.usc.edu/gabril/) into `atari-dataset/<Game>/num_episodes_<N>_fs4_human.pt`, then:

```bash
python train.py --algorithm AuxGazeFactorizedViViT --loading-method gabril \
  --gaze-loss-mode kl_then_mean --game ChopperCommand --seed 100
```

Add `--no-gaze` for the behavioral cloning baseline. To sweep on a cluster:

```bash
sbatch slurm/train-snoopy.slurm ChopperCommand
sbatch slurm/train-carc.slurm Alien
```

## Where everything is

| | |
| --- | --- |
| Project state, model details, known bugs, next steps | [docs/handoff.md](docs/handoff.md) |
| Why the method is set up this way | [docs/decisions.md](docs/decisions.md) |
| Every run, generated from wandb | [docs/experiments.md](docs/experiments.md) and [experiments.csv](docs/experiments.csv) |
| Weights & Biases | https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari |
| Web-agent gaze project | https://github.com/adsrivatsa/gaze-web-nav |
| GABRIL paper | https://arxiv.org/abs/2507.19647 |

## Repo map

`train.py` is the entry point. `config.py` holds the CLI and the `Config` dataclass. `vivit.py` is the custom transformer and the model variants. `dataset.py` loads the GABRIL `.pt` files and builds the gaze targets. `env_manager.py` wraps Atari for evaluation rollouts. `augmentation.py`, `checkpoint.py`, and `device.py` are support. `slurm/` has the cluster scripts and `docs/` has the handoff documents.

There is no `transformers` dependency. The transformer is written from scratch with `einops` and `torch`.

## People

- PI: Erdem Bıyık, USC LIRA Lab.
- PhD mentor: Yutai Zhou.
- Atari ViViT experiments: Abhinav Srivatsa.
- Web-agent gaze work: Danie Craig Kulandai.

Ask Yutai first for anything methodology related.
