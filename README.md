# RL fine-tuning of Efficient-WAM-RT (CMU 16-824)

Which part of a world-action model should RL update: the **action** expert, the **video** model, or **both**?
We fine-tune [Efficient-WAM-RT](https://github.com/jiajun613/Efficient-WAM) with Flow-GRPO (sparse success reward) on two RoboTwin 2.0 tasks and evaluate on four.

> **Status (Nov 26, 00:40 ET):** the 12-iteration comparison below is final (Joint lacks 8 of 48 handover_block episodes, see Limitations). Longer runs to convergence are in progress, see [Ongoing](#ongoing-longer-runs).

## Results

Successes out of 48 (paired evaluation seeds, `demo_randomized`, unseen instructions, no exploration noise):

| Task | BC | Action-only | Video-only | Joint | Paper (rand.) |
|---|---|---|---|---|---|
| handover_mic (train) | 23 (47.9%) | 22 (45.8%) | 21 (43.8%) | 21 (43.8%) | 69% |
| stack_blocks_three (train) | 25 (52.1%) | 30 (62.5%) | 27 (56.2%) | 32 (66.7%) | 60% |
| blocks_ranking_size (not in RL) | 19 (39.6%) | 22 (45.8%) | 19 (39.6%) | 23 (47.9%) | 45% |
| handover_block, first 8 seeds excluded (n=40) | 23 (57.5%) | 25 (62.5%) | 18 (45.0%) | 24 (60.0%)† | |
| handover_block, all 48 (not in RL) | 28 (58.3%) | 29 (60.4%) | 22 (45.8%) | — | 58% |
| **Total, seeds Joint has (184)** | 90 (48.9%) | 99 (53.8%) | 85 (46.2%) | 100 (54.3%) | |
| **Total, all 192** | 95 (49.5%) | 103 (53.6%) | 89 (46.4%) | — | |

† Joint was evaluated on an H200, the others on A6000 (see Limitations).

Paired exact McNemar vs. BC, pooled: Action +26/-18, p=0.29; Video +14/-20, p=0.39. Joint (184 seeds): +23/-13, p=0.13. **None differs significantly from BC.** The smallest per-task p-values (Video on handover_block +0/-6, p=0.031; Joint on stack_blocks_three +8/-1, p=0.039) do not survive correction for 12 comparisons.

Joint vs. Action: +15/-14, p=1.00. Joint vs. Video: +24/-9, p=0.014, not corrected and not planned in advance. Run `python summarize.py` for every test.

| Training | Action | Video | Joint |
|---|---|---|---|
| Optimizer steps | 287 | 236 | 155 |
| Iterations stopped by KL > 0.05 (of 12) | 1 | 7 | 4 |
| Iterations skipped (too few informative chunks) | 0 | 0 | 1 |
| Resumes during training (Adam moments reset) | 2 | 1 | 2 |

The Joint column counts resumes of its second run; its first run was lost to the quota failure.

## Ongoing: longer runs

Goal: train each method to convergence, then test once on the four tasks (A6000 only).

- **Convergence rule (fixed in advance):** every 4 iterations, run a noise-free validation on 24 fixed training seeds per training task (48 episodes). Stop once the last 3 validations set no new best, after at least 24 iterations (cap: 40).
- **Video-only, rerun:** lr lowered from 1e-6 to 3e-7, from BC. This change was chosen after seeing the 12-iteration test results.
- **Action-only and Joint:** continue from iteration 12 with unchanged settings. Not started yet; waiting for GPUs.

Video-only (lr 3e-7) validation, successes out of 24 per task:

| Iteration | 0 (BC) | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| handover_mic | 12 | 13 | 14 | 14 | 14 | 14 |
| stack_blocks_three | 11 | 15 | 16 | 13 | 14 | 12 |
| Mean | 47.9% | 58.3% | 62.5% | 56.2% | 58.3% | 54.2% |

24 of 40 iterations done. The best validation is still iteration 8; iterations 12, 16 and 20 did not beat it, so the run stops at 24 unless that validation sets a new best. KL stops: 0 (7 of 12 at lr 1e-6); per-iteration max KL 0.0–0.05 (0.008–0.19 at 1e-6). Iterations 5, 13 were skipped because every group was all-success or all-failure. Curves: `results/val_video_lr3e-7.json`, `results/train_video_lr3e-7.json`.

## Model

Efficient-WAM-RT (`Efficient-WAM-RT_stage3.pt`): a compact Wan 2.2 video model (849M parameters, 12 layers) and an action expert (139M) that reads the video model's KV cache, 10 flow-matching steps, 16-action chunks. UMT5-XXL text embeddings are precomputed and cached. "Video-only" trains the 849M video model, "Action-only" the 139M expert, "Joint" both.

## Method

- **Policy:** 10 denoising steps as a Gaussian SDE (eta 0.3, sigma_min 0.02); the last step is deterministic because its tiny noise amplifies bf16 error in the log-probability.
- **GRPO:** 12 iterations x 4 groups x 8 rollouts with the same seed; advantage normalized per group; PPO clip 0.2 per denoising step; AdamW, lr 1e-6, mini-batch 16, at most 512 chunks per iteration, KL stop 0.05, fp32 master weights.
- **Gradient to the video model:** the stock code detaches the video KV cache; `kv_grad.patch` removes that.

## Compute and time

About 25 GPU-hours so far (22 on RTX A6000, 3 on H200), over about 24 hours of wall-clock time (Nov 24 22:00 to Nov 25 22:20 ET). Each run uses 1 GPU and 8 parallel simulators. One training iteration takes 7–15 min on an A6000; a 4-task test takes about 75 min on an A6000 (about 2 h on an H200). Most wall-clock time was spent waiting for shared-cluster GPUs, and on reruns after a disk-quota failure and preemptions.

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
- **Joint was evaluated on an H200 (other methods: A6000)**, because no A6000 was available; rendering may differ, so cross-method comparisons with Joint carry this extra difference. A first Joint evaluation attempt (H200) was preempted at 40/48 handover_mic episodes and its results discarded; the rerun gave the same 16/40 at the same point.
- On the H200, rendering of handover_block evaluation seed index 3 (seed 145050) hangs. The 8-episode batch containing it was skipped, so Joint has 40/48 episodes there; the other methods are unaffected.
- Micro-batch size differs per condition (16/8/4) for memory only; gradients agree (cosine >= 0.998).

## Credits

Code and runs were made with Claude Code (Claude Opus 5.5).

Efficient-WAM (Apache-2.0; the source of `cfg.yml` and of the patched line), RoboTwin 2.0 (`RLinf_support` branch, `dca9ec6`), RLinf (Apache-2.0; `rlinf/envs/sim/robotwin/seeds/train_seeds.json`), Wan 2.2.
