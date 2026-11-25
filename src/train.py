"""GRPO on Efficient-WAM-RT. --mode action|video|both chooses the weights that are trained, --mode prep caches text and evaluates BC."""
import argparse, functools, json, math, os, sys, threading, time, zipfile
import multiprocessing as mp
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np


def log(*a):
    try: print(time.strftime("[%H:%M:%S]"), *a, flush=True)
    except OSError: pass  # a full disk (quota) must not stop the run


def write_json(x, path):
    """Atomic, with retries; the last resort is the node's own disk."""
    for k in range(10):
        try:
            with open(path + ".tmp", "w") as f: json.dump(x, f, indent=1); f.flush(); os.fsync(f.fileno())
            os.replace(path + ".tmp", path); return True
        except OSError as e:
            log("not written:", path, e); time.sleep(30)
    keep = "/var/tmp/wam_keep"; os.makedirs(keep, exist_ok=True)
    json.dump(x, open(os.path.join(keep, path.strip("/").replace("/", "_")), "w")); return False


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", action="store_true")
    p.add_argument("--n_env", type=int, default=8)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--smin", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--mb", type=int, default=16)
    p.add_argument("--micro", type=int, default=0, help="0 = fastest that fits next to 8 simulators on 48 GB: action 16, video 8, both 4")
    p.add_argument("--ckpt", type=int, default=2, help="0 none, 1 all steps, 2 joint video-action steps, 3 cached action steps")
    p.add_argument("--skip_last", type=int, default=1)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--camera_near", type=float, default=0.01)
    p.add_argument("--mode", choices=["prep", "action", "video", "both"], required=True)
    p.add_argument("--train_tasks", default="handover_mic,stack_blocks_three")
    p.add_argument("--eval_tasks", default="handover_mic,stack_blocks_three,blocks_ranking_size,handover_block")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--groups_per_task", type=int, default=2, help="per iteration and task: groups of n_env rollouts with the same seed")
    p.add_argument("--n_train_seeds", type=int, default=60, help="per task")
    p.add_argument("--n_eval", type=int, default=48, help="per task")
    p.add_argument("--max_chunks", type=int, default=512, help="chunks used per update, random subset")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--max_kl", type=float, default=0.05, help="stop the iteration when a mini-batch exceeds it")
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--park", type=int, default=1, help="0: the optimizer state stays on the GPU during rollouts (less host memory)")
    p.add_argument("--exec", choices=["official", "chunk"], default="official", help="how a 16-action chunk is executed")
    p.add_argument("--recycle", type=int, default=2, help="restart a simulator after this many resets")
    p.add_argument("--recompute", type=int, default=0, help="recompute old log-probs with the training micro-batches")
    p.add_argument("--text_cache", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out", default="bench.json")
    a = p.parse_args()
    os.environ["WAM_EXEC"] = a.exec
    a.micro = a.micro or {"action": 16, "video": 8, "both": 4, "prep": 8}[a.mode]
    assert a.mb % a.micro == 0
    return a


class Envs:
    def __init__(self, a):
        self.a, self.ctx = a, mp.get_context("spawn")
        self.c, self.p, self.nr, self.rss = [None] * a.n_env, [None] * a.n_env, [0] * a.n_env, []
        self.spawn(range(a.n_env))

    def spawn(self, ids):
        from env_worker import worker
        a = self.a
        for i in ids:
            if self.p[i] is not None:
                try: self.c[i].send(("close", None))
                except Exception: pass
                self.p[i].join(20)
                if self.p[i].is_alive(): self.p[i].kill(); self.p[i].join(20)
                self.c[i].close()
            x, y = self.ctx.Pipe()
            pr = self.ctx.Process(target=worker, args=(y, i, None, None, not a.clean, "seen", a.camera_near), daemon=True)
            pr.start(); y.close(); self.c[i], self.p[i], self.nr[i] = x, pr, 0
        for i in ids: self.recv(i, 600)

    def recv(self, i, timeout=600):
        if not self.c[i].poll(timeout):
            raise RuntimeError(f"env {i} timeout")
        s, r = self.c[i].recv()
        if s == "error":
            raise RuntimeError(f"env {i}:\n{r}")
        return r

    def reset(self, seeds):
        old = [i for i in range(len(seeds)) if self.p[i] is None or self.nr[i] >= self.a.recycle]
        if old: self.spawn(old)
        for c, s in zip(self.c, seeds): c.send(("reset", s))
        out = [self.recv(i) for i in range(len(seeds))]
        for i in range(len(seeds)): self.nr[i] += 1
        def rss(pid):
            try: return int(next(l for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")).split()[1]) / 1e6
            except Exception: return 0.0
        self.rss = [round(rss(q.pid), 2) for q in self.p if q] + [round(rss(os.getpid()), 2)]  # GB, trainer last
        return out

    def stop(self):  # frees the simulators' GPU memory; reset() starts them again
        for i in range(len(self.p)):
            if self.p[i] is None: continue
            try: self.c[i].send(("close", None))
            except Exception: pass
        for i in range(len(self.p)):
            if self.p[i] is None: continue
            self.p[i].join(20)
            if self.p[i].is_alive(): self.p[i].kill(); self.p[i].join(20)
            self.c[i].close(); self.c[i] = self.p[i] = None

    def step(self, ids, acts):
        for i, x in zip(ids, acts): self.c[i].send(("step", x))
        return [self.recv(i) for i in ids]

    def close(self):
        for c in self.c:
            try: c.send(("close", None))
            except Exception: pass


def main():
    a = get_args()
    t_start = time.time()
    res = {"args": vars(a)}
    envs = None
    if True:
        t = time.time(); envs = Envs(a); res["t_env_start"] = time.time() - t
        log(f"envs up {res['t_env_start']:.0f}s")
    import torch
    torch.load = functools.partial(torch.load, weights_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    sys.path.insert(0, os.environ["WAM_RUN"])
    from EfficientWAM.models import compact_wan as cw
    from EfficientWAM.models.wan_model import WanVideoModel
    from EfficientWAM import model_loader as ml
    from EfficientWAM.third_party.wan.utils.fm import FlowMatchScheduler

    def from_cfg(cls, config, device="cuda"):  # skip the 20 GB teacher; the RL checkpoint overwrites it
        arch = json.load(open(os.path.join(config.config_path, "config.json")))
        arch = {k: v for k, v in arch.items() if not k.startswith("_")}
        arch.update(config.to_wan_model_config())
        return cls(config=config, video_model=WanVideoModel(arch, config.vae_path, device, config.precision))
    cw.CompactWANModel.from_teacher_checkpoint = classmethod(from_cfg)
    ml.CompactWANModel = cw.CompactWANModel

    build_t5 = ml._build_t5_encoder
    ml._build_t5_encoder = lambda config, device: None  # training reads the text cache, prep builds it while the simulators are down
    t = time.time()
    rt = ml.build_runtime_from_config(ml.load_deploy_config(os.path.join(os.environ["WAM_RUN"], "cfg.yml")), device="cuda")
    model = rt.model
    model._start_latency_event = lambda *x, **k: None
    model._finish_latency_event = lambda *x, **k: None
    res["t_model_load"] = time.time() - t
    act_p = [(n, p) for n, p in model.named_parameters() if n.startswith("action_expert.")]
    vid_p = [(n, p) for n, p in model.named_parameters() if ".wan_model." in n]
    other = [n for n, p in model.named_parameters() if not n.startswith("action_expert.") and ".wan_model." not in n]
    nparam = lambda ps: sum(p.numel() for _, p in ps)
    res["n_action"], res["n_video"], res["n_other"] = nparam(act_p), nparam(vid_p), len(other)
    res["dtype_action"], res["dtype_video"] = str(act_p[0][1].dtype), str(vid_p[0][1].dtype)
    log(f"model {res['t_model_load']:.0f}s action={res['n_action']/1e6:.0f}M video={res['n_video']/1e6:.0f}M other={other[:3]}")
    log("gpu mem after load GB", torch.cuda.memory_allocated() / 1e9)

    K, KV = rt.num_inference_steps, len(rt.video_refresh_steps)
    assert rt.video_refresh_steps == (0, 1), rt.video_refresh_steps

    def sched(n):
        s = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True); s.set_timesteps(n)
        return s.timesteps.cuda(), torch.cat([s.sigmas, torch.zeros(1)]).tolist()
    at, asig = sched(K); vt, vsig = sched(KV)
    fh, fw = [x // 16 for x in model.compact_wan.config.future_video_size]
    zshape = (48, rt.num_video_frames // 4, fh, fw)

    def mean_std(x, v, k):
        s, sn = asig[k], asig[k + 1]
        x0, eps = x - s * v, x + (1 - s) * v
        std = max(a.eta * sn, a.smin)
        m = (1 - sn) * x0 + math.sqrt(max(sn * sn - std * std, 0.0)) * eps
        return m, std

    from torch.utils.checkpoint import checkpoint
    def run_step(fn, batch):
        joint = fn is model  # 1 = all steps, 2 = joint video-action steps only, 3 = cached action steps only
        if torch.is_grad_enabled() and (a.ckpt == 1 or (a.ckpt == 2 and joint) or (a.ckpt == 3 and not joint)):
            return checkpoint(fn, batch, use_reentrant=False, preserve_rng_state=False)
        return fn(batch)

    def velocities(cond, z, state, text, get_x, keep):
        """Run the 10-step chain. get_x(k, v) returns the next action latent."""
        B = cond.shape[0]
        x, vlat, cache, out = get_x(-1, None), z, None, []
        for k in range(K - (a.skip_last if keep else 0)):
            tk = at[k].expand(B).to(torch.float32)
            if k < KV:
                o = run_step(model, {"video_t": vt[k].to(torch.bfloat16).expand(B), "initial_state": state, "noisy_actions": x,
                                     "action_t": tk, "text_embeddings": text, "condition_latent": cond, "future_latent": vlat,
                                     "return_video_cache": True})
                cache = o["video_cache"]
                vlat = vlat + o["video_pred"] * (vsig[k + 1] - vsig[k])
            else:
                o = run_step(model.forward_action_with_video_cache, {"video_cache": cache, "initial_state": state,
                                                                     "noisy_actions": x, "action_t": tk})
            v = o["action_pred"].float()
            if keep: out.append(v)
            x = get_x(k, v)
        return out

    def logp_of(v, x, xn, k):
        m, std = mean_std(x, v, k)
        return (-((xn - m) ** 2) / (2 * std * std) - math.log(std) - 0.5 * math.log(2 * math.pi)).sum((1, 2))

    text_cache = torch.load(a.text_cache) if os.path.exists(a.text_cache) else {}
    text_cache = {k: v.cuda() for k, v in text_cache.items()}
    def text_emb(s):
        for _ in range(120):  # the prep job may still be writing the cache
            if s in text_cache: break
            if a.mode != "prep" and os.path.exists(a.text_cache):
                for k, v in torch.load(a.text_cache).items(): text_cache.setdefault(k, v.cuda())
            if s not in text_cache: time.sleep(15)
        return text_cache[s]
    peak_gpu = [0.0]
    def watch():  # device-wide use, simulators included
        while True:
            f, t = torch.cuda.mem_get_info(); peak_gpu[0] = max(peak_gpu[0], (t - f) / 1e9); time.sleep(0.3)
    threading.Thread(target=watch, daemon=True).start()
    def gpu_peak():
        v = round(peak_gpu[0], 1); peak_gpu[0] = 0.0; return v

    @torch.no_grad()
    def act(obs, sde=True):
        B = len(obs)
        img = torch.from_numpy(np.stack([o["image"] for o in obs])).cuda().to(torch.bfloat16)
        cond = model.compact_wan.encode_video((img * 2 - 1).unsqueeze(2))
        cond = cond if torch.is_tensor(cond) else torch.stack(list(cond))
        state = rt.action_normalizer.normalize(torch.from_numpy(np.stack([o["state"] for o in obs])).cuda().float())
        text = [text_emb(o["instruction"]) for o in obs]
        z = torch.randn((B,) + zshape, device="cuda", dtype=torch.bfloat16)
        chain, lps = [], []
        def get_x(k, v):
            if k < 0:
                chain.append(torch.randn(B, rt.chunk_size, 14, device="cuda"))
            else:
                m, std = mean_std(chain[-1], v, k)
                last = not sde or (a.skip_last and k == K - 1)
                if last: m = chain[-1] + (asig[k + 1] - asig[k]) * v
                chain.append(m if last else m + std * torch.randn_like(m))
                if not last: lps.append(logp_of(v, chain[-2], chain[-1], k))
            return chain[-1]
        velocities(cond, z, state, text, get_x, False)
        a_env = rt.action_normalizer.denormalize(chain[-1]).float().cpu().numpy()
        return a_env, dict(cond=cond.cpu(), z=z.cpu(), state=state.cpu(), chain=torch.stack(chain, 1).cpu(),
                           lp=torch.stack(lps, 1).cpu() if lps else None, text=[o["instruction"] for o in obs])

    def rollout(specs, sde=True, collect=True):
        """specs: one (task, step limit, seed, instruction type) per env. Returns stats, chunk data, trajectory index of every chunk, success."""
        r = dict(t_reset=0., t_policy=0., t_env=0.)
        t = time.time(); rs = envs.reset(specs); r["t_reset"] = time.time() - t
        obs = {i: x["obs"] for i, x in enumerate(rs)}
        succ, data, traj = np.zeros(len(specs)), [], []
        lim, n = math.ceil(max(x[1] for x in specs) / rt.chunk_size) + 1, 0
        while obs and n < lim:
            ids = sorted(obs)
            torch.cuda.synchronize(); t = time.time()
            acts, d = act([obs[i] for i in ids], sde); torch.cuda.synchronize(); r["t_policy"] += time.time() - t
            if collect: data.append(d); traj += ids
            t = time.time(); out = envs.step(ids, list(acts)); r["t_env"] += time.time() - t
            obs = {}
            for i, o in zip(ids, out):
                if o["done"]: succ[i] = o["success"]
                else: obs[i] = o["obs"]
            n += 1
        r["rounds"] = n; r["instruction"] = rs[0]["obs"]["instruction"]; r["real_seed"] = [x["seed"] for x in rs]
        b = None
        if collect:
            b = {k: torch.cat([d[k] for d in data]) for k in ("cond", "z", "state", "chain", "lp")}
            b["text"] = sum([d["text"] for d in data], [])
        return r, b, np.array(traj), succ

    def chunk_logp(b, idx):
        chain = b["chain"][idx].cuda()
        vs = velocities(b["cond"][idx].cuda(), b["z"][idx].cuda(), b["state"][idx].cuda(),
                        [text_emb(b["text"][i]) for i in idx], lambda k, v: chain[:, k + 1], True)
        return torch.stack([logp_of(v, chain[:, k], chain[:, k + 1], k) for k, v in enumerate(vs)], 1)

    LIM = dict(handover_mic=600, stack_blocks_three=1200, blocks_ranking_size=1200, handover_block=800)  # RoboTwin step limits
    rl = json.load(open(os.environ["SEEDS"]))
    def seeds_of(task):
        """RLinf's expert-checked seeds where they exist, plain spaced seeds otherwise (spacing leaves room for unstable-scene retries)."""
        if task in rl:
            x = [int(v) for v in rl[task]["success_seeds"]]
            return x[:a.n_train_seeds], x[-100:][:a.n_eval]
        return [100000 + 20 * i for i in range(a.n_train_seeds)], [200000 + 20 * i for i in range(a.n_eval)]
    train_tasks, eval_tasks = a.train_tasks.split(","), a.eval_tasks.split(",")
    SEEDS = {t: seeds_of(t) for t in set(train_tasks + eval_tasks)}
    for t, (x, y) in SEEDS.items(): assert not set(x) & set(y), t

    def evaluate(tasks, tag=""):
        out = {}
        for tk in tasks:
            ev, succ, tr = SEEDS[tk][1], [], dict(t_reset=0., t_policy=0., t_env=0., rounds=0)
            sf = os.path.join(a.ckpt_dir, f"skip_eval_{tk}.json")  # batch starts that hang on this GPU: run the other batches into their own file, no final result
            if os.path.exists(sf):
                skip, rf = json.load(open(sf)), os.path.join(a.ckpt_dir, f"rest_eval_{tk}_{tag}.json")
                rest = json.load(open(rf)) if os.path.exists(rf) else {}
                for s in range(0, len(ev), a.n_env):
                    if s in skip or str(s) in rest: continue
                    torch.manual_seed(10_000 + s)
                    rest[str(s)] = rollout([(tk, LIM[tk], x, "unseen") for x in ev[s:s + a.n_env]], sde=False, collect=False)[3].tolist()
                    write_json(rest, rf); log(f"  rest eval {tk} batch {s} {sum(rest[str(s)])}/{a.n_env}")
                continue
            pf = os.path.join(a.ckpt_dir, f"partial_eval_{tk}_{tag}.json")  # finished batches survive a preemption
            if tag and os.path.exists(pf): succ = json.load(open(pf)); log(f"  eval {tk} continues at {len(succ)}")
            for s in range(len(succ), len(ev), a.n_env):
                torch.manual_seed(10_000 + s)  # same initial noise for every checkpoint
                r, _, _, sc = rollout([(tk, LIM[tk], x, "unseen") for x in ev[s:s + a.n_env]], sde=False, collect=False); succ += sc.tolist()
                for k in tr: tr[k] += r[k]
                if tag: write_json(succ, pf)
                log(f"  eval {tk} {len(succ)}/{len(ev)} success {np.mean(succ):.3f} rss GB {envs.rss}")
            out[tk] = dict(n=len(ev), success=float(np.mean(succ)), per_seed=succ, exec=a.exec, **tr)
        return out

    def save():
        res["elapsed"] = time.time() - t_start
        try:
            tmp = a.out + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, a.out)
        except OSError as e: log("result file not written:", e)

    if a.mode == "prep":
        seedf = a.text_cache + ".instr.json"  # "task seed type" -> instruction
        instr = json.load(open(seedf)) if os.path.exists(seedf) else {}
        want = [(t, LIM[t], v, "seen") for t in train_tasks for v in SEEDS[t][0]] + [(t, LIM[t], v, "unseen") for t in eval_tasks for v in SEEDS[t][1]]
        todo = [w for w in want if f"{w[0]} {w[2]} {w[3]}" not in instr]
        for s in range(0, len(todo), a.n_env):
            batch = todo[s:s + a.n_env]; t = time.time()
            for w, x in zip(batch, envs.reset(batch)): instr[f"{w[0]} {w[2]} {w[3]}"] = x["obs"]["instruction"]
            json.dump(instr, open(seedf + ".tmp", "w")); os.replace(seedf + ".tmp", seedf)
            log(f"prep seeds {len(want) - len(todo) + s + len(batch)}/{len(want)} {time.time() - t:.0f}s rss GB {envs.rss} gpu peak {gpu_peak()}")
        missing = sorted(set(instr.values()) - set(text_cache))
        if missing:
            envs.close(); [q.join(30) for q in envs.p]
            t5 = build_t5(ml.load_deploy_config(os.path.join(os.environ["WAM_RUN"], "cfg.yml")), "cuda")
            with torch.no_grad():
                for x in missing: text_cache[x] = t5([rt.scene_prefix + x], "cuda")[0].to(torch.bfloat16)
            del t5; import gc; gc.collect(); torch.cuda.empty_cache()
            tmp = a.text_cache + ".tmp"; torch.save({k: v.cpu() for k, v in text_cache.items()}, tmp); os.replace(tmp, a.text_cache)
            log(f"encoded {len(missing)} instructions, cache {len(text_cache)}")
            envs = Envs(a)
        bcf = os.path.join(a.ckpt_dir, f"bc_eval_{a.exec}.json")
        bc_res = json.load(open(bcf)) if os.path.exists(bcf) else {}
        os.makedirs(a.ckpt_dir, exist_ok=True)
        for tk in eval_tasks:
            if tk in bc_res and bc_res[tk]["n"] >= a.n_eval: continue
            bc_res.update(evaluate([tk])); write_json(bc_res, bcf)
            log(f"BC eval {tk} {bc_res[tk]['success']:.3f} env {bc_res[tk]['t_env']:.0f}s policy {bc_res[tk]['t_policy']:.0f}s reset {bc_res[tk]['t_reset']:.0f}s rounds {bc_res[tk]['rounds']}")
        res["bc_eval"] = bc_res; save(); envs.close(); log("ALL_DONE"); return

    for _, p in act_p: p.requires_grad_(a.mode in ("action", "both"))
    for _, p in vid_p: p.requires_grad_(a.mode in ("video", "both"))
    train = [(n, p) for n, p in act_p + vid_p if p.requires_grad]
    res["n_train"] = nparam(train)
    masters = [p.detach().float().clone().requires_grad_(True) for _, p in train]  # fp32 copies: bf16 weights would lose small updates
    opt = torch.optim.AdamW(masters, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    bc_masters = None
    os.makedirs(a.ckpt_dir, exist_ok=True)
    latest = os.path.join(a.ckpt_dir, "latest.pt")
    start, res["iters"], res["evals"] = 0, [], {}

    def push():  # masters -> the weights the model computes with
        with torch.no_grad():
            for m, (_, p) in zip(masters, train): p.copy_(m)
    def save_ckpt(it, path):
        """Verified before it replaces the previous checkpoint: a full disk (quota) must not destroy the old one."""
        tmp = path + ".tmp"
        want = sum(m.numel() for m in masters) * 4
        for k in range(4):
            try:
                torch.save(dict(it=it, names=[n for n, _ in train], masters=[m.detach().cpu() for m in masters], iters=res["iters"], evals=res["evals"]), tmp)
                with open(tmp, "rb") as f: os.fsync(f.fileno())
                ok = os.path.getsize(tmp) > want and zipfile.is_zipfile(tmp)
                if ok:
                    with zipfile.ZipFile(tmp) as z: ok = len(z.namelist()) > len(masters)
            except Exception as e:
                ok = False; log("checkpoint failed:", repr(e)[:200])
            if ok:
                os.replace(tmp, path); return True
            log(f"checkpoint {it} incomplete, keeping the previous one (try {k})")
            try: os.remove(tmp)
            except OSError: pass
            time.sleep(60)
        return False
    if os.path.exists(latest) and not zipfile.is_zipfile(latest): log("ignoring a corrupt checkpoint, starting over")
    elif os.path.exists(latest):  # a preempted run continues; Adam moments start again
        c = torch.load(latest, map_location="cpu")
        assert c["names"] == [n for n, _ in train]
        with torch.no_grad():
            for m, v in zip(masters, c["masters"]): m.copy_(v.to(m.device))
        push(); start, res["iters"], res["evals"] = c["it"], c["iters"], c["evals"]
        log("resumed at iteration", start); del c

    def move(dev):
        for m in masters: m.data = m.data.to(dev)
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0: st[k] = v.to(dev)
        torch.cuda.empty_cache()
    big = a.mode in ("video", "both"); park = big and a.park

    def update(b, adv, idx_all, rng):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        perm = rng.permutation(idx_all)
        micros = [perm[k:k + a.micro].tolist() for k in range(0, len(perm), a.micro)]
        o = {}
        if a.recompute:
            with torch.no_grad(): olds = [chunk_logp(b, m) for m in micros]
            dr = torch.cat([ol - b["lp"][m].cuda() for ol, m in zip(olds, micros)])
            o["rollout_vs_recompute"] = float((dr.exp() - 1).abs().mean())
        else: olds = [b["lp"][m].cuda() for m in micros]  # bf16 rollout and training forward agree to about 1% in the ratio
        torch.cuda.synchronize(); o["t_recompute"] = time.time() - t
        acc, t = max(1, a.mb // a.micro), time.time()
        gn, kls, clipf = [], [], []
        for s in range(0, len(micros), acc):
            group = micros[s:s + acc]; n_tot = sum(len(m) for m in group); kl = cf = 0.0
            for q, idx in enumerate(group):
                ad = torch.tensor(adv[idx], device="cuda", dtype=torch.float32)[:, None]
                lr = chunk_logp(b, idx) - olds[s + q]; ratio = lr.exp()
                loss = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean() * len(idx) / n_tot
                loss.backward()
                kl += float((ratio - 1 - lr).mean().detach()) * len(idx) / n_tot
                cf += float(((ratio - 1).abs() > a.clip).float().mean()) * len(idx) / n_tot
            kls.append(kl); clipf.append(cf)
            if kl > a.max_kl:
                model.zero_grad(set_to_none=True); o["kl_stop"] = s // acc; break
            for m, (_, p) in zip(masters, train):
                m.grad = None if p.grad is None else p.grad.float()
                p.grad = None
            g = float(torch.nn.utils.clip_grad_norm_(masters, 1.0)); gn.append(g)
            if not math.isfinite(g): raise RuntimeError("non-finite gradient")
            opt.step(); push()
            for m in masters: m.grad = None
        torch.cuda.synchronize()
        o.update(t_update=time.time() - t, peak_gb=torch.cuda.max_memory_allocated() / 1e9, n_chunk=len(perm), n_step=len(gn),
                 grad_norm=float(np.mean(gn)) if gn else 0.0, kl=kls[:1] + kls[-1:] + [float(np.max(kls))], clip_frac=float(np.mean(clipf)))
        return o

    def run_eval():
        if park: move("cpu")
        torch.cuda.empty_cache()
        f = os.path.join(a.ckpt_dir, "final_eval.json"); done = json.load(open(f)) if os.path.exists(f) else {}
        for tk in eval_tasks:
            if tk in done: continue
            done.update(evaluate([tk], f"it{a.iters}")); write_json(done, f)
            if tk not in done: continue
            log(f"FINAL EVAL {tk} {done[tk]['success']:.3f} (n={done[tk]['n']})")
        res["evals"]["final"] = done; save()

    for it in range(start, a.iters):
        t0 = time.time(); rng = np.random.RandomState(1000 * (a.rng + 1) + it); torch.manual_seed(1000 * (a.rng + 1) + it)
        seeds = [(t, SEEDS[t][0][(it * a.groups_per_task + j) % len(SEEDS[t][0])]) for j in range(a.groups_per_task) for t in train_tasks]
        roll, parts, advs, succs = [], [], [], []
        if park: move("cpu")
        torch.cuda.empty_cache()
        for tk, sd in seeds:
            r, b, traj, succ = rollout([(tk, LIM[tk], sd, "seen")] * a.n_env)
            roll.append(r); succs.append(succ)
            av = (succ - succ.mean()) / (succ.std() + 1e-6) if succ.std() > 0 else np.zeros(a.n_env)
            parts.append(b); advs.append(av[traj])
        b = {k: torch.cat([p[k] for p in parts]) for k in ("cond", "z", "state", "chain", "lp")}
        b["text"] = sum([p["text"] for p in parts], [])
        adv = np.concatenate(advs); use = np.nonzero(adv)[0]  # groups that all succeed or all fail carry no signal
        n_used_all = len(use)
        if len(use) > a.max_chunks: use = rng.choice(use, a.max_chunks, replace=False)
        o = dict(it=it, n_signal_chunk=n_used_all, seeds=seeds, success=[float(x.mean()) for x in succs], t_rollout=time.time() - t0, n_total_chunk=len(adv),
                 rss=envs.rss, t_reset=sum(x["t_reset"] for x in roll), t_policy=sum(x["t_policy"] for x in roll), t_env=sum(x["t_env"] for x in roll))
        o["gpu_peak_rollout"] = gpu_peak()
        if big: envs.stop()
        if park: move("cuda")
        if len(use) >= a.mb: o.update(update(b, adv, use, rng))
        else: o["skipped"] = True
        o["gpu_peak_update"] = gpu_peak()
        del b, parts
        res["iters"].append(o); save()
        log("ITER", it, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != "seeds"})
        if (it + 1) % a.save_every == 0 or it + 1 == a.iters: save_ckpt(it + 1, latest)
    run_eval()
    envs.close(); save(); log("ALL_DONE", f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
