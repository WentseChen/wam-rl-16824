#!/bin/bash
#SBATCH -J wam_train
#SBATCH -o logs/train_%j.out
RTW=dca9ec682688821c42944eacd7f5dd9bfb15396f
ulimit -c 0
P=${WAM_ROOT:?set to the directory that holds run/ and req.txt}
T=/var/tmp/wam_$SLURM_JOB_ID; mkdir -p $T/{tmp,home}; trap 'rm -rf $T' EXIT TERM INT
export UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=8
export TMPDIR=$T/tmp HOME=$T/home UV_CACHE_DIR=$T/uvc UV_PYTHON_INSTALL_DIR=$T/uvpy UV_TOOL_DIR=$T/uvt HF_HOME=$T/hf PYTHONUTF8=1 HF_HUB_DISABLE_TELEMETRY=1
UV=${UV:-$(command -v uv)}
say() { echo "=== $1 $(date +%T)"; }
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /var/tmp | tail -1
say stage
MODE=${MODE:?}; TASK=${TASK:-handover_mic}; LIM=${LIM:-600}; T5W=; [ $MODE = prep ] && [ -z "$NOT5" ] && T5W=models_t5_umt5-xxl-enc-bf16.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $P/{ckpt,ctl,logs}
dl() {  # dl <repo> <files...> [--repo-type dataset] --local-dir <dir>: resumable, a stalled connection is dropped after 30 s
  local repo=$1 kind="" dir="" files=() f k; shift
  while [ $# -gt 0 ]; do case $1 in --repo-type) kind=datasets/; shift 2;; --local-dir) dir=$2; shift 2;; *) files+=("$1"); shift;; esac; done
  for f in "${files[@]}"; do
    mkdir -p "$dir/$(dirname "$f")"
    for k in $(seq 1 60); do curl -sfL -C - --connect-timeout 20 --speed-limit 200000 --speed-time 30 -o "$dir/$f.part" "https://huggingface.co/$kind$repo/resolve/main/$f" && { mv "$dir/$f.part" "$dir/$f"; break; }; sleep 3; done
    [ -s "$dir/$f" ] || echo "DL_FAIL $repo $f"
  done
}
(  $UV venv -q --python 3.10 $T/venv && for k in 1 2 3 4; do $UV pip install -q --python $T/venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match -r $P/req.txt&& break; done; $T/venv/bin/python -c "import numpy, torch, sapien" 2>/dev/null && sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $T/venv/lib/python3.10/site-packages/mplib/planner.py && say venv_ok ) &
( dl jiajun0613/Efficient-WAM_RoboTwin Efficient-WAM-RT/Efficient-WAM-RT_stage3.pt Efficient-WAM-RT/Efficient-WAM-RT_action_stats.json --local-dir $T/ckpt
  dl Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth $T5W config.json google/umt5-xxl/special_tokens_map.json google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json --local-dir $T/wan; say weights_ok ) &
( git clone -q -b RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git $T/RoboTwin && git -C $T/RoboTwin checkout -q $RTW
  dl TianxingChen/RoboTwin2.0 embodiments.zip objects.zip background_texture.zip --repo-type dataset --local-dir $T/z
  cd $T/RoboTwin/assets && for f in embodiments objects background_texture; do python3 -m zipfile -e $T/z/$f.zip . && rm $T/z/$f.zip || echo "UNZIP_FAIL $f"; done
  cd .. && python3 script/update_embodiment_config_path.py < /dev/null | tail -1; say assets_ok ) &
wait; du -sh $T/venv $T/ckpt $T/wan $T/RoboTwin
export W=$T WAM_RUN=$T/run WAM_TMP=$T/tmp ROBOTWIN_ROOT=$T/RoboTwin SEEDS=$T/run/seeds_train.json OMP_NUM_THREADS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
# training needs the text of every task: the prep job rewrites the cache (it only held handover_mic before)
if [ $MODE != prep ]; then for i in $(seq 1 100); do [ $(stat -c %s $P/ckpt/text_all.pt) -gt 119000000 ] && break; sleep 15; done; say "text cache $(stat -c %s $P/ckpt/text_all.pt)"; fi
mkdir -p /var/tmp/wam_keep
runall() {
  rm -rf $T/run; cp -r $P/run $T/run
  sed -i "s#@ROBOTWIN@#$T/RoboTwin#; s#@WAM_DIR@#$T#g" $T/run/cfg.yml
  export PYTHONPATH=$T/run; cd $T/run
  say $MODE; $T/venv/bin/python train.py --mode $MODE --n_env 8 --recycle 2 --text_cache $P/ckpt/text_all.pt --ckpt_dir $P/ckpt/$([ $MODE = prep ] && echo bc || echo ${MODE}_s0) --out $P/logs/${MODE}_$SLURM_JOB_ID.json $EXTRA 2>&1 | tee -ai -p /var/tmp/wam_keep/train_$SLURM_JOB_ID.log; [ ${PIPESTATUS[0]} = 0 ] || return 12
  # after Action: the BC baseline of the tasks the prep job did not reach (cached text, no T5)
  if [ $MODE = action ]; then say bc_rest; $T/venv/bin/python train.py --mode prep --n_env 8 --recycle 2 --text_cache $P/ckpt/text_all.pt --ckpt_dir $P/ckpt/bc --out $P/logs/bcrest_$SLURM_JOB_ID.json || echo BC_REST_FAILED; fi
}
rm -f $P/ctl/retry_$SLURM_JOB_ID $P/ctl/quit_$SLURM_JOB_ID
for a in 1 2 3 4 5 6 7 8; do
  runall; rc=$?; say "attempt $a rc=$rc"; [ $rc = 0 ] && break
  for i in $(seq 1 72); do [ -e $P/ctl/quit_$SLURM_JOB_ID ] && break 2; [ -e $P/ctl/retry_$SLURM_JOB_ID ] && { rm -f $P/ctl/retry_$SLURM_JOB_ID; continue 2; }; sleep 10; done
  say "no retry in 12 min, releasing the GPU"; break
done
say END
