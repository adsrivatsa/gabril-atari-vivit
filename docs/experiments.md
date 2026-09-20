# Experiment registry

This file is generated from the Weights & Biases project, so it reflects what was actually logged rather than what any one of us remembers running. The full per-run dump, including run ids and URLs, is in `experiments.csv` next to this file. Regenerate both with the wandb API if runs are added.

- Project: https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari
- Runs in the dump: 372
- Metric reported below: `test/best/mean_return`, the mean undiscounted return over 100 evaluation episodes for the best-return checkpoint.

Two caveats. First, the numbers in the slide deck do not line up exactly with what is logged here, especially for the factorized gaze runs, which suggests the deck was built from earlier runs that are no longer in this project. Treat wandb as the record of truth and regenerate the deck before you cite it. Second, a few runs finished before the final test block and have no `test/best` value; they are counted as zero-metric and skipped.

## 8-seed sweep (seeds 100-107)

The default config: `gaze_loss_mode=kl_then_mean`, all layers available, `gaze_loss_layers` and `gaze_loss_heads` unset in config (they default to `last` and `all` in code).

| Game | Variant | n | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Alien | FactorizedViViT, gaze | 8 | 1,431.7 | 104.0 | 1,234.0 | 1,620.6 |
| Alien | AuxGazeFactorizedViViT, gaze | 8 | 1,472.8 | 237.0 | 1,119.1 | 1,888.1 |
| Alien | FactorizedViViT, no gaze | 8 | 950.6 | 389.8 | 336.1 | 1,701.6 |
| Alien | AuxGazeFactorizedViViT, no gaze | 8 | 1,085.7 | 165.9 | 798.2 | 1,327.4 |
| Asterix | FactorizedViViT, gaze | 8 | 957.8 | 256.0 | 514.0 | 1,335.5 |
| Asterix | AuxGazeFactorizedViViT, gaze | 8 | 948.8 | 174.6 | 767.5 | 1,350.0 |
| Asterix | FactorizedViViT, no gaze | 8 | 200.0 | 0.0 | 200.0 | 200.0 |
| Asterix | AuxGazeFactorizedViViT, no gaze | 8 | 200.0 | 0.0 | 200.0 | 200.0 |
| Breakout | FactorizedViViT, gaze | 8 | 24.5 | 2.5 | 21.0 | 28.1 |
| Breakout | AuxGazeFactorizedViViT, gaze | 8 | 25.9 | 2.7 | 22.6 | 31.8 |
| Breakout | FactorizedViViT, no gaze | 8 | 0.2 | 0.6 | 0.0 | 1.8 |
| Breakout | AuxGazeFactorizedViViT, no gaze | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| ChopperCommand | FactorizedViViT, gaze | 8 | 2,839.4 | 521.9 | 1,914.0 | 3,479.0 |
| ChopperCommand | AuxGazeFactorizedViViT, gaze | 8 | 2,986.2 | 777.4 | 1,890.0 | 4,749.0 |
| ChopperCommand | FactorizedViViT, no gaze | 8 | 1,938.4 | 654.5 | 775.0 | 2,864.0 |
| ChopperCommand | AuxGazeFactorizedViViT, no gaze | 8 | 1,989.0 | 700.1 | 806.0 | 3,037.0 |
| MsPacman | FactorizedViViT, gaze | 8 | 1,683.0 | 224.1 | 1,447.9 | 2,098.7 |
| MsPacman | AuxGazeFactorizedViViT, gaze | 8 | 2,017.9 | 235.7 | 1,698.5 | 2,409.4 |
| MsPacman | FactorizedViViT, no gaze | 8 | 1,489.5 | 105.0 | 1,322.9 | 1,646.6 |
| MsPacman | AuxGazeFactorizedViViT, no gaze | 8 | 1,564.1 | 212.4 | 1,158.4 | 1,853.7 |
| Qbert | FactorizedViViT, gaze | 8 | 7,135.6 | 2,011.9 | 3,325.5 | 10,557.8 |
| Qbert | AuxGazeFactorizedViViT, gaze | 8 | 6,716.3 | 996.4 | 4,804.5 | 8,097.2 |
| Qbert | FactorizedViViT, no gaze | 8 | 4,190.8 | 1,125.8 | 3,071.5 | 6,110.0 |
| Qbert | AuxGazeFactorizedViViT, no gaze | 8 | 3,022.1 | 1,301.7 | 1,475.8 | 5,275.5 |
| Seaquest | FactorizedViViT, gaze | 8 | 902.9 | 142.3 | 632.0 | 1,069.2 |
| Seaquest | AuxGazeFactorizedViViT, gaze | 8 | 940.5 | 190.5 | 513.2 | 1,098.8 |
| Seaquest | FactorizedViViT, no gaze | 8 | 575.9 | 42.0 | 506.8 | 630.6 |
| Seaquest | AuxGazeFactorizedViViT, no gaze | 8 | 614.1 | 123.2 | 430.8 | 853.4 |
| UpNDown | FactorizedViViT, gaze | 8 | 7,548.0 | 396.4 | 6,946.0 | 8,189.4 |
| UpNDown | AuxGazeFactorizedViViT, gaze | 8 | 7,519.6 | 978.5 | 5,827.5 | 9,122.6 |
| UpNDown | FactorizedViViT, no gaze | 8 | 2,579.5 | 512.7 | 1,844.5 | 3,198.7 |
| UpNDown | AuxGazeFactorizedViViT, no gaze | 8 | 2,866.5 | 509.5 | 2,041.8 | 3,754.0 |

## Breakout layer and head ablation (seeds 100-107)

Same runs as the sweep but with `gaze_loss_layers` and `gaze_loss_heads` set explicitly. The pattern is stark: restricting the gaze loss to earlier layers kills Breakout, while last layer runs score in the double digits.

| Game | Variant | n | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- |
| Breakout | AuxGazeFactorizedViViT, layers=all, heads=all | 8 | 6.7 | 2.5 | 4.3 | 10.2 |
| Breakout | FactorizedViViT, layers=all, heads=all | 8 | 6.8 | 2.1 | 3.1 | 10.8 |
| Breakout | AuxGazeFactorizedViViT, layers=all, heads=half | 8 | 9.4 | 3.3 | 4.7 | 14.7 |
| Breakout | FactorizedViViT, layers=all, heads=half | 8 | 6.5 | 3.1 | 3.4 | 13.1 |
| Breakout | AuxGazeFactorizedViViT, layers=first, heads=all | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Breakout | FactorizedViViT, layers=first, heads=all | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Breakout | AuxGazeFactorizedViViT, layers=first, heads=half | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Breakout | FactorizedViViT, layers=first, heads=half | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Breakout | AuxGazeFactorizedViViT, layers=last, heads=half | 8 | 12.0 | 3.7 | 7.5 | 18.6 |
| Breakout | FactorizedViViT, layers=last, heads=half | 8 | 14.1 | 4.7 | 7.4 | 22.8 |

## Single-seed runs (seed 42)

The first experiments, one seed each, mixed modes. Useful for a smoke test, not for a claim.

| Game | Algorithm | Gaze | Mode | `test/best/mean_return` | Run |
| --- | --- | --- | --- | --- | --- |
| Alien | AuxGazeFactorizedViViT | yes | kl_then_mean | 1,690.9 | [5974873a1926](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/5974873a1926) |
| Alien | AuxGazeFactorizedViViT | yes | mean_then_kl | not logged | [e693797bbb63_new](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/e693797bbb63_new) |
| Alien | FactorizedViViT | yes | mean_then_kl | 1,363.6 | [9cdba493799e](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/9cdba493799e) |
| Alien | FactorizedViViT | yes | kl_then_mean | 1,450.1 | [65011f59a030](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/65011f59a030) |
| Asterix | AuxGazeFactorizedViViT | yes | mean_then_kl | 1,502.0 | [1f69d38cc28f](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/1f69d38cc28f) |
| Asterix | AuxGazeFactorizedViViT | yes | kl_then_mean | 958.5 | [fd42ce2954b5](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/fd42ce2954b5) |
| Asterix | FactorizedViViT | yes | mean_then_kl | 795.5 | [bbce18932c78](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/bbce18932c78) |
| Asterix | FactorizedViViT | yes | kl_then_mean | 1,605.5 | [d9f61e699e6a](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/d9f61e699e6a) |
| Breakout | AuxGazeFactorizedViViT | yes | mean_then_kl | 26.9 | [6a51d11374f1](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/6a51d11374f1) |
| Breakout | AuxGazeFactorizedViViT | yes | kl_then_mean | 25.9 | [a81a7c762507](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/a81a7c762507) |
| Breakout | FactorizedViViT | yes | mean_then_kl | 26.3 | [77bd30fd8973](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/77bd30fd8973) |
| Breakout | FactorizedViViT | yes | kl_then_mean | 24.4 | [2ba31efc7f30](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/2ba31efc7f30) |
| ChopperCommand | AuxGazeFactorizedViViT | yes | mean_then_kl | 2,566.0 | [795a674dab5c](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/795a674dab5c) |
| ChopperCommand | AuxGazeFactorizedViViT | yes | kl_then_mean | 2,478.0 | [8922e09e817d](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/8922e09e817d) |
| ChopperCommand | AuxGazeFactorizedViViT | yes | mean_then_kl | not logged | [eb447f84efed](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/eb447f84efed) |
| ChopperCommand | AuxGazeFactorizedViViT | yes | mean_then_kl | not logged | [b5ba267e5d74](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/b5ba267e5d74) |
| ChopperCommand | FactorizedViViT | yes | mean_then_kl | 3,229.0 | [0afd396ca80a](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/0afd396ca80a) |
| ChopperCommand | FactorizedViViT | yes | kl_then_mean | 2,867.0 | [fb19f79422b8](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/fb19f79422b8) |
| ChopperCommand | FactorizedViViT | yes | mean_then_kl | not logged | [d59f71fe01e0](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/d59f71fe01e0) |
| ChopperCommand | FactorizedViViT | yes | mean_then_kl | not logged | [ac3df14293ec](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/ac3df14293ec) |
| MsPacman | AuxGazeFactorizedViViT | yes | kl_then_mean | 1,882.1 | [870eed2790a5](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/870eed2790a5) |
| MsPacman | AuxGazeFactorizedViViT | yes | mean_then_kl | not logged | [bcdfba3a5ea8_new](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/bcdfba3a5ea8_new) |
| MsPacman | FactorizedViViT | yes | mean_then_kl | 1,508.5 | [cac5b5cd3474](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/cac5b5cd3474) |
| MsPacman | FactorizedViViT | yes | kl_then_mean | 1,815.6 | [46007f46ac17](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/46007f46ac17) |
| Qbert | AuxGazeFactorizedViViT | yes | mean_then_kl | 4,488.0 | [58d017137720](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/58d017137720) |
| Qbert | AuxGazeFactorizedViViT | yes | kl_then_mean | 7,487.8 | [d2f33ea18961](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/d2f33ea18961) |
| Qbert | FactorizedViViT | yes | mean_then_kl | 6,023.0 | [7b3426371f7f](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/7b3426371f7f) |
| Qbert | FactorizedViViT | yes | kl_then_mean | 7,800.0 | [e71288901fcd](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/e71288901fcd) |
| Seaquest | AuxGazeFactorizedViViT | yes | kl_then_mean | 760.6 | [006160da9e30](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/006160da9e30) |
| Seaquest | AuxGazeFactorizedViViT | yes | mean_then_kl | not logged | [424443860489_new](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/424443860489_new) |
| Seaquest | FactorizedViViT | yes | mean_then_kl | 845.8 | [69a5930b9bd0](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/69a5930b9bd0) |
| Seaquest | FactorizedViViT | yes | kl_then_mean | not logged | [78aedb21842d_new](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/78aedb21842d_new) |
| UpNDown | AuxGazeFactorizedViViT | yes | mean_then_kl | 8,368.6 | [8d541db3392a](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/8d541db3392a) |
| UpNDown | AuxGazeFactorizedViViT | yes | kl_then_mean | 7,663.7 | [7bedae159997](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/7bedae159997) |
| UpNDown | FactorizedViViT | yes | mean_then_kl | 7,737.3 | [5d86bc876bd1](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/5d86bc876bd1) |
| UpNDown | FactorizedViViT | yes | kl_then_mean | 9,223.2 | [ecb211adde18](https://wandb.ai/papaya147-ml/ViViT-GABRIL-Atari/runs/ecb211adde18) |

If you add runs, keep the CSV in sync and update the tables above. The grouping key is the seed plus whether `gaze_loss_layers` or `gaze_loss_heads` was set.
