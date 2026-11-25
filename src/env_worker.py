"""One RoboTwin env per process. Talks to the trainer over a Pipe."""
import importlib.util, os, sys, time, traceback
import numpy as np

ROBOTWIN = os.environ["ROBOTWIN_ROOT"]


def _build_args(task, step_lim, randomized):
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    name = "demo_randomized" if randomized else "demo_clean"
    with open(os.path.join(ROBOTWIN, "task_config", name + ".yml")) as f:
        a = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
        emb = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
        cam = yaml.safe_load(f)
    robot = os.path.join(ROBOTWIN, emb[a["embodiment"][0]]["file_path"])
    with open(os.path.join(robot, "config.yml")) as f:
        ecfg = yaml.safe_load(f)
    a.update(task_name=task, step_lim=step_lim, planner_backend="mplib", eval_mode=True,
             eval_video_log=False, render_freq=0, n_envs=1, action_dim=14, rdt_step=10,
             head_camera_h=cam["D435"]["h"], head_camera_w=cam["D435"]["w"],
             left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True,
             left_embodiment_config=ecfg, right_embodiment_config=ecfg,
             embodiment_name=a["embodiment"][0], instruction=None)
    a["save_path"] = os.path.join(os.environ["WAM_TMP"], task)
    return a


def _info(env, task):
    """RLinf's get_info() of these tasks plays the expert demonstration, which solves the scene; the fields only need the blocks' sides."""
    side = lambda x: "left" if x.get_pose().p[0] < 0 else "right"
    if task in ("stack_blocks_three", "blocks_ranking_size"):
        names = ("red block", "green block", "blue block") if task == "stack_blocks_three" else ("large block", "medium block", "small block")
        return {"{A}": names[0], "{B}": names[1], "{C}": names[2],
                "{a}": side(env.block1), "{b}": side(env.block2), "{c}": side(env.block3)}
    return env.get_info()


def worker(conn, idx, task, step_lim, randomized, instruction_type, camera_near):
    try:
        os.chdir(ROBOTWIN)
        sys.path.insert(0, ROBOTWIN)
        os.environ["ASSETS_PATH"] = ROBOTWIN
        import importlib
        from envs.camera import Camera
        from description.utils.generate_episode_instructions import generate_episode_descriptions
        spec = importlib.util.spec_from_file_location(
            "imu", os.path.join(os.environ["WAM_RUN"], "EfficientWAM/utils/image_utils.py"))
        imu = importlib.util.module_from_spec(spec); spec.loader.exec_module(imu)
        import cv2
        if camera_near > 0:
            orig = Camera.load_camera
            class Near:
                def __init__(s, sc): s._s = sc
                def add_camera(s, *a, **k): k["near"] = camera_near; return s._s.add_camera(*a, **k)
                def __getattr__(s, n): return getattr(s._s, n)
            Camera.load_camera = lambda self, scene, *a, **k: orig(self, Near(scene), *a, **k)
        cfgs = {}  # task -> (args, module)
        env, n_reset, instr = None, 0, ""
        official = os.environ.get("WAM_EXEC", "official") == "official"
        sys.stdout = open(os.devnull, "w")  # take_action prints every step

        def pack(raw):
            o = raw["observation"]
            row = np.concatenate([cv2.resize(o["left_camera"]["rgb"], (160, 120)),
                                  cv2.resize(o["right_camera"]["rgb"], (160, 120))], axis=1)
            img = np.concatenate([o["head_camera"]["rgb"], row], axis=0)
            if img.shape[:2] != (384, 320):
                img = imu.resize_with_padding(img, (384, 320))
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            return dict(image=img.transpose(2, 0, 1), instruction=instr,
                        state=np.asarray(raw["joint_action"]["vector"], np.float32))

        conn.send(("ready", None))
        while True:
            cmd, arg = conn.recv()
            t0 = time.time()
            if cmd == "reset":
                task, step_lim, seed, itype = arg
                seed = int(seed)
                if task not in cfgs: cfgs[task] = (_build_args(task, step_lim, randomized), importlib.import_module(f"envs.{task}"))
                args, mod = cfgs[task]
                if env is not None:
                    n_reset += 1
                    env.close_env(clear_cache=n_reset % 5 == 0)
                for k in range(20):
                    env = getattr(mod, task)()
                    try:
                        env.setup_demo(now_ep_num=seed + k, seed=seed + k, **args)
                        break
                    except Exception as e:
                        if type(e).__name__ != "UnStableError":
                            raise
                        env.close_env(clear_cache=True)
                else:
                    raise RuntimeError("no stable seed")
                env.step_lim, env.run_steps, env.reward_step = step_lim, 0, 0
                np.random.seed(seed)
                d = generate_episode_descriptions(task, [_info(env, task)], 1, seed)
                instr = str(np.random.choice(d[0][itype]))
                env.instruction = instr
                conn.send(("ok", dict(obs=pack(env.get_obs()), seed=seed + k, t=time.time() - t0)))
            elif cmd == "step":
                if official:  # one action at a time, as in RoboTwin's script/eval_policy.py
                    for x in np.asarray(arg, np.float64):
                        if env.eval_success or env.take_action_cnt >= env.step_lim: break
                        env.take_action(x, action_type="qpos")
                    info = dict(success=bool(env.eval_success))
                    done = info["success"] or env.take_action_cnt >= env.step_lim
                else:  # RLinf: the whole chunk as one compressed, downsampled path
                    r, term, trunc, info = env.gen_sparse_reward_data(np.asarray(arg, np.float64))
                    done = bool(term[0]) or bool(trunc[0])
                t1 = time.time()
                conn.send(("ok", dict(obs=None if done else pack(env.get_obs()), done=done,
                                      success=bool(info["success"]), t=time.time() - t0,
                                      t_render=time.time() - t1)))
            else:
                break
    except Exception:
        conn.send(("error", traceback.format_exc()))
