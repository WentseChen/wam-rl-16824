# RL fine-tuning of Efficient-WAM-RT (CMU 16-824)

Which part of a world-action model should RL update: the **action** expert, the **video** model, or **both**?
We fine-tune [Efficient-WAM-RT](https://github.com/jiajun613/Efficient-WAM) with Flow-GRPO (sparse success reward) on two RoboTwin 2.0 tasks and evaluate on four.

> **Status:** BC, Action-only and Video-only are final. Joint training is finished (12/12 iterations); its evaluation is **pending**.

## Results

Successes out of 48 (paired evaluation seeds, `demo_randomized`, unseen instructions, no exploration noise):

| Task | BC | Action-only | Video-only | Joint | Paper (rand.) |
|---|---|---|---|---|---|
| handover_mic (train) | 23 (47.9%) | 22 (45.8%) | 21 (43.8%) | pending | 69% |
| stack_blocks_three (train) | 25 (52.1%) | 30 (62.5%) | 27 (56.2%) | pending | 60% |
| blocks_ranking_size (not in RL) | 19 (39.6%) | 22 (45.8%) | 19 (39.6%) | pending | 45% |
| handover_block (not in RL) | 28 (58.3%) | 29 (60.4%) | 22 (45.8%) | pending | 58% |
| **Total / 192** | 95 (49.5%) | 103 (53.6%) | 89 (46.4%) | pending | |

Paired exact McNemar vs. BC, pooled: Action +26/-18, p=0.29; Video +14/-20, p=0.39. **Neither differs significantly from BC.** The smallest per-task p is Video on handover_block (+0/-6, p=0.031), which does not survive correction for 8 comparisons.

| Training | Action | Video | Joint |
|---|---|---|---|
| Optimizer steps | 287 | 236 | 155 |
| Iterations stopped by KL > 0.05 (of 12) | 1 | 7 | 4 |
| Iterations skipped (too few informative chunks) | 0 | 0 | 1 |
| Resumes during training (Adam moments reset) | 2 | 1 | 2 |

## Method

- **Policy:** 10 denoising steps as a Gaussian SDE (eta 0.3, sigma_min 0.02); the last step is deterministic because its tiny noise amplifies bf16 error in the log-probability.
- **GRPO:** 12 iterations x 4 groups x 8 rollouts with the same seed; advantage normalized per group; PPO clip 0.2 per denoising step; AdamW, lr 1e-6, mini-batch 16, at most 512 chunks per iteration, KL stop 0.05, fp32 master weights.
- **Gradient to the video model:** the stock code detaches the video KV cache; `kv_grad.patch` removes that.

## Files

- `src/train.py` trainer and evaluation, `src/env_worker.py` RoboTwin worker, `src/cfg.yml` model config, `src/seeds_train.json` seeds (subset of RLinf)
- `train.sbatch` Slurm job (downloads weights and assets to node-local disk)
- `results/` per-seed evaluations and training curves; `python summarize.py` reprints the table and tests (run it from the repo root)

No weights or third-party source are included.

## Reproduce

```bash
export WAM_ROOT=/path/to/wam && mkdir -p $WAM_ROOT/run $WAM_ROOT/logs
git clone https://github.com/jiajun613/Efficient-WAM && git -C Efficient-WAM checkout 5851bae
patch -p1 -d Efficient-WAM/inference/robotwin/EfficientWAM < kv_grad.patch
cp -r Efficient-WAM/inference/robotwin/EfficientWAM src/* $WAM_ROOT/run/ && cp req.txt $WAM_ROOT/
for m in prep action video both; do (cd $WAM_ROOT && sbatch --export=ALL,MODE=$m --gres=gpu:a6000:1 -c 8 --mem-per-gpu=33000 $OLDPWD/train.sbatch); done
```

Run `prep` first (text cache and BC baseline); the others wait for its cache. Cluster paths were replaced by variables for this release; the cleaned-up script has not been re-run end to end. Needs a 48 GB GPU with Vulkan ray tracing.

## Limitations

- **One random seed, 48 episodes per task** (about +/-14 points per cell) and 12 iterations (reduced from 50 for time). Small differences are noise.
- lr was raised from 3e-7 to 1e-6 after seeing the iteration-0 KL (not evaluation results); it is probably too high for Video and Joint (KL stops).
- BC on handover_mic is below the paper (47.9% vs 69%); cause not found, execution mode ruled out.
- "Not in RL" is not zero-shot: BC pretraining saw all four tasks.
- Seeds of stack_blocks_three and blocks_ranking_size were not expert-checked for solvability.
- Joint was rerun from scratch after a disk-quota failure corrupted the first run's checkpoint (the first run was never evaluated). Its last 3 iterations ran on an H200, everything else on A6000.
- A first Joint evaluation was preempted at 40/48 episodes of handover_mic on an H200 (16 successes); it is discarded and rerun on an A6000.
- Micro-batch size differs per condition (16/8/4) for memory only; gradients agree (cosine >= 0.998).

## Credits

Efficient-WAM (Apache-2.0; the source of `cfg.yml` and of the patched line), RoboTwin 2.0 (`RLinf_support` branch, `dca9ec6`), RLinf (Apache-2.0; `rlinf/envs/sim/robotwin/seeds/train_seeds.json`), Wan 2.2.
