# RL fine-tuning of Efficient-WAM-RT (CMU 16-824)

Which part of a world-action model should RL update: the **action** expert, the **video** model, or **both**?
We fine-tune [Efficient-WAM-RT](https://github.com/jiajun613/Efficient-WAM) with Flow-GRPO (sparse success reward) on two RoboTwin 2.0 tasks and evaluate on four.

> **Status (Nov 27, 15:30 ET): all experiments are finished**, including a second setup with held-out validation ([v2](#v2-held-out-validation-larger-batches)).

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

## Longer runs (to convergence)

**Summary** (test successes out of 192, all on A6000 except † on H200):

| | BC | Action | Video | Joint |
|---|---|---|---|---|
| 12 iterations (lr 1e-6) | 95 | **103** | 89 | 100/184† |
| Converged, 24 iterations | — | 86 | 87 (lr 3e-7) | 96 |

Training to convergence did not help any method: Action got worse (+13/-30 vs. its 12-iteration result, p=0.014), Video stayed flat, and Joint ended at the BC level (96 vs. 95; +20/-19, p=1.0). Training-seed validation stopped improving after iteration 8–12, and test success never rose above the 12-iteration results; Video-only even gained 10–15 points on training seeds without any test gain. This points to overfitting the 60 training seeds per task; a held-out validation set would be a better stopping signal. **Best result overall: Action-only after 12 iterations, 103/192 (53.6%), not significantly better than BC (p=0.29).**


Goal: train each method to convergence, then test once on the four tasks (A6000 only).

- **Convergence rule (fixed in advance):** every 4 iterations, run a noise-free validation on 24 fixed training seeds per training task (48 episodes). Stop once the last 3 validations set no new best, after at least 24 iterations (cap: 40).
- **Video-only, rerun:** lr lowered from 1e-6 to 3e-7, from BC. This change was chosen after seeing the 12-iteration test results.
- **Action-only and Joint:** continue from iteration 12 with unchanged settings.

**Video-only (lr 3e-7): converged at iteration 24.** Validation (training seeds), successes out of 24 per task:

| Iteration | 0 (BC) | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| handover_mic | 12 | 13 | 14 | 14 | 14 | 14 | 13 |
| stack_blocks_three | 11 | 15 | 16 | 13 | 14 | 12 | 15 |
| Mean | 47.9% | 58.3% | 62.5% | 56.2% | 58.3% | 54.2% | 58.3% |

No KL stops (7 of 12 at lr 1e-6); iterations 5 and 13 were skipped (every group all-success or all-failure).

Test (A6000, the same held-out test seeds as in Results, disjoint from the validation seeds), successes out of 48:

| Task | BC | Video, lr 1e-6, 12 it. | Video, lr 3e-7, 24 it. |
|---|---|---|---|
| handover_mic (train) | 23 | 21 | 24 |
| stack_blocks_three (train) | 25 | 27 | 23 |
| blocks_ranking_size (not in RL) | 19 | 19 | 13 |
| handover_block (not in RL) | 28 | 22 | 27 |
| **Total / 192** | 95 (49.5%) | 89 (46.4%) | 87 (45.3%) |

vs. BC: +14/-22, p=0.24; vs. Video lr 1e-6: +20/-22, p=0.88. **The lower lr made training stable and raised training-seed validation (48% to 58–63%), but test success did not improve**, even on the two training tasks: the gain did not carry over from the 60 training seeds to new seeds. The drop on blocks_ranking_size (+5/-11, p=0.21) is not significant.

**Action-only, continued from iteration 12: converged at iteration 24.** Validation mean (training seeds, 48 episodes): 54.2% / 54.2% / 50.0% / 50.0% at iterations 12 / 16 / 20 / 24, i.e. no gain after iteration 12.

Test, successes out of 48:

| Task | BC | Action, 12 it. | Action, 24 it. |
|---|---|---|---|
| handover_mic (train) | 23 | 22 | 21 |
| stack_blocks_three (train) | 25 | 30 | 23 |
| blocks_ranking_size (not in RL) | 19 | 22 | 15 |
| handover_block (not in RL) | 28 | 29 | 27 |
| **Total / 192** | 95 (49.5%) | 103 (53.6%) | 86 (44.8%) |

24 vs. 12 iterations: +13/-30, p=0.014 (not corrected); vs. BC: +16/-25, p=0.21. **Training longer made Action-only worse on test seeds**, mostly on stack_blocks_three and blocks_ranking_size, while training-seed validation stayed flat. Together with Video-only this suggests the policy overfits the 60 training seeds; a validation set of held-out seeds would be a better stopping signal.

**Joint, continued from iteration 12: converged at iteration 24.** Validation mean: 56.2% / 54.2% / 52.1% / 54.2% at iterations 12 / 16 / 20 / 24.

| Task | BC | Joint, 12 it. (H200) | Joint, 24 it. (A6000) |
|---|---|---|---|
| handover_mic (train) | 23 | 21 | 22 |
| stack_blocks_three (train) | 25 | 32 | 29 |
| blocks_ranking_size (not in RL) | 19 | 23 | 19 |
| handover_block (not in RL) | 28 | 24/40 | 26 |
| **Total** | 95/192 | 100/184 | 96/192 |

24 vs. 12 iterations (184 shared seeds): +13/-20, p=0.30; vs. BC: +20/-19, p=1.0. On the A6000 the seed that hangs on the H200 rendered normally.

## v2: held-out validation, larger batches

A second setup to fix the problems found above, all three methods trained again from BC:

- **Held-out validation:** 24 seeds per training task that are never trained on (disjoint from the test seeds), every 4 iterations; the checkpoint with the best validation is tested.
- **Larger batches:** 64 episodes per iteration (was 32), up to 1024 chunks per update, 96 training seeds per task.
- **Joint:** separate learning rates (action expert 1e-6, video model 3e-7) and separate gradient clipping per part.
- Video: lr 3e-7. Stop when 3 validations in a row set no new best (at least 12 iterations, at most 24). 4 CPU cores per job.

Held-out validation, successes out of 48 (24 per task):

| Iteration | 0 (BC) | 4 | 8 | 12 |
|---|---|---|---|---|
| Action | 70.8% | 60.4% | 58.3% | 60.4% |
| Video | 70.8% | 68.8% | 64.6% | 66.7% |
| Joint | 70.8% | 56.2% | 54.2% | 50.0% |

**All three stopped at iteration 12 with BC (iteration 0) as the best checkpoint**, so each method's test result is BC's: 95/192. Action and Joint fell 10–20 points on held-out seeds; Video, with the smaller learning rate, fell least (4–6 points) but never beat BC. Each run took 12 iterations (about 590–600 optimizer steps, 1–3 KL stops).

**Conclusion:** with this reward (sparse task success), batch size and 1 seed, GRPO fine-tuning of Efficient-WAM-RT did not improve held-out success for any choice of trained parameters. The 12-iteration gains in Results are within noise and do not survive held-out model selection.

Files: `results/val_{action,video,joint}_v2.json`, `results/train_{action,video,joint}_v2.json`. Flags: `--heldout_val 1 --test_best 1 --groups_per_task 4 --max_chunks 1024 --n_train_seeds 96` (and `--lr_video`, `--sep_clip` for Joint).

## Model

Efficient-WAM-RT (`Efficient-WAM-RT_stage3.pt`): a compact Wan 2.2 video model (849M parameters, 12 layers) and an action expert (139M) that reads the video model's KV cache, 10 flow-matching steps, 16-action chunks. UMT5-XXL text embeddings are precomputed and cached. "Video-only" trains the 849M video model, "Action-only" the 139M expert, "Joint" both.

## Method

- **Policy:** 10 denoising steps as a Gaussian SDE (eta 0.3, sigma_min 0.02); the last step is deterministic because its tiny noise amplifies bf16 error in the log-probability.
- **GRPO:** 12 iterations x 4 groups x 8 rollouts with the same seed; advantage normalized per group; PPO clip 0.2 per denoising step; AdamW, lr 1e-6, mini-batch 16, at most 512 chunks per iteration, KL stop 0.05, fp32 master weights.
- **Gradient to the video model:** the stock code detaches the video KV cache; `kv_grad.patch` removes that.

## Compute and time

About 61 GPU-hours in total (57 on RTX A6000, 3 on H200; v2 alone 24), over about 65 hours of wall-clock time (Nov 24 22:00 to Nov 27 15:22 ET). Each run uses 1 GPU and 8 parallel simulators. One training iteration takes 7–15 min on an A6000; a 4-task test takes about 75 min on an A6000 (about 2 h on an H200). Most wall-clock time was spent waiting for shared-cluster GPUs, and on reruns after a disk-quota failure and preemptions.

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
