"""
Multiprocessing vectorized environment wrapper.

Wraps any VecEnv class across worker processes. Generic — no game-specific
assumptions. Obs is a NamedTuple pytree; field shapes are derived from a
prototype reset() call.

Agent model + opponent state live on GPU, shared via CUDA IPC (spawn).
Workers run the underlying VecEnv directly — no reimplementation.

Interface:
    obs, masks = vec.reset(seed=...)
    obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
"""

import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple


# ── Commands ────────────────────────────────────────────────────────────────

CMD_STEP = 0
CMD_RESET = 1
CMD_SHUTDOWN = 2


# ── OffsetBatchedPlayer ────────────────────────────────────────────────────

class OffsetBatchedPlayer:
    """Maps local env indices [0,K) → global [start, start+K) in shared GPU state."""

    def __init__(self, parent, start_idx: int, num_envs: int):
        self.parent = parent
        self.start_idx = start_idx
        self.num_envs = num_envs

    def batch_action(self, obs_batch, mask_batch, idxs: list) -> np.ndarray:
        return self.parent.batch_action(obs_batch, mask_batch,
                                         [i + self.start_idx for i in idxs])

    def reset(self, env_indices=None):
        if env_indices is None:
            self.parent.reset(list(range(self.start_idx, self.start_idx + self.num_envs)))
        else:
            self.parent.reset([i + self.start_idx for i in env_indices])

    def slice(self, env_idx: int):
        return self.parent.slice(env_idx + self.start_idx)


# ── SharedMemory helpers ───────────────────────────────────────────────────

def _shm_alloc(shape, dtype):
    """Allocate SharedMemory-backed numpy array. Returns (shm, view)."""
    nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
    shm = SharedMemory(create=True, size=nbytes)
    view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    view[:] = 0
    return shm, view


def _shm_attach(name, shape, dtype):
    """Attach to existing SharedMemory. Returns (shm, view)."""
    shm = SharedMemory(name=name, create=False)
    return shm, np.ndarray(shape, dtype=dtype, buffer=shm.buf)


# ── Worker ──────────────────────────────────────────────────────────────────

def _worker_fn(
    worker_id, start_idx, end_idx, batched_player,
    cmd_pipe, result_pipe, shm_names,
    obs_field_names,       # list of str: field names of the obs NamedTuple
    vec_env_cls_path,      # "module:ClassName" for the VecEnv class
    vec_env_kwargs,        # dict of kwargs for VecEnv constructor
):
    import traceback
    try:
        torch.set_num_threads(1)
        # Import VecEnv class by path
        mod_name, cls_name = vec_env_cls_path.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_name)
        VecEnvCls = getattr(mod, cls_name)

        local_n = end_idx - start_idx
        opp = OffsetBatchedPlayer(batched_player, start_idx, local_n)
        env = VecEnvCls(num_envs=local_n, opponent=opp, **vec_env_kwargs)

        # Attach to shared memory
        shm_handles = {}
        views = {}
        for key, (name, shape, dtype_str) in shm_names.items():
            s, v = _shm_attach(name, tuple(shape), np.dtype(dtype_str))
            shm_handles[key] = s
            views[key] = v

        def _write_obs(obs):
            for fname in obs_field_names:
                views[f"obs_{fname}"][:] = getattr(obs, fname)

        while True:
            cmd = cmd_pipe.recv()

            if cmd == CMD_SHUTDOWN:
                for s in shm_handles.values():
                    s.close()
                env.close()
                result_pipe.send("done")
                break

            elif cmd == CMD_RESET:
                seed_val = int(views["seed"][0])
                obs, masks = env.reset(seed=seed_val if seed_val >= 0 else None)
                opp.reset()
                _write_obs(obs)
                views["masks"][:] = masks
                result_pipe.send("done")

            elif cmd == CMD_STEP:
                actions = views["actions"].copy()
                obs, masks, rewards, terminated, _, infos = env.step(actions)
                done_idxs = list(np.where(terminated)[0])
                if done_idxs:
                    opp.reset(done_idxs)
                _write_obs(obs)
                views["masks"][:] = masks
                views["rewards"][:] = rewards
                views["terminated"][:] = terminated
                data = pickle.dumps(infos)
                views["infos_len"][0] = len(data)
                views["infos"][:len(data)] = np.frombuffer(data, dtype=np.uint8)
                result_pipe.send("done")

    except Exception:
        traceback.print_exc()
        result_pipe.send("error")


# ── Main ────────────────────────────────────────────────────────────────────

class MultiProcessVecEnv:
    """
    Multiprocessing vectorized environment wrapper.

    Generic: derives obs layout from a prototype reset() call. No game-specific
    assumptions. Agent + opponent state on GPU, shared across workers via CUDA IPC.

    Parameters
    ----------
    num_envs     : total number of environments
    num_workers  : number of worker processes
    opponent     : BatchedPlayer (or compatible) on GPU
    vec_env_cls  : the VecEnv class to instantiate in each worker
    vec_env_kwargs : kwargs passed to vec_env_cls (excluding num_envs and opponent)
    """

    def __init__(
        self,
        num_envs: int,
        num_workers: int,
        opponent,
        vec_env_cls,
        vec_env_kwargs: dict = None,
    ):
        assert opponent is not None
        assert num_envs % num_workers == 0
        self.num_envs = num_envs
        self.num_workers = num_workers
        epw = num_envs // num_workers
        self._epw = epw
        vec_env_kwargs = vec_env_kwargs or {}

        ctx = mp.get_context('spawn')

        # Create a prototype env to derive obs shapes and num_actions
        proto_env = vec_env_cls(num_envs=1, opponent=opponent, **vec_env_kwargs)
        proto_obs, proto_masks = proto_env.reset()
        proto_env.close()

        # Obs field names and shapes (game-agnostic: derived from prototype)
        self._obs_field_names = list(proto_obs._fields)
        obs_field_specs = []
        for fname in self._obs_field_names:
            field = getattr(proto_obs, fname)
            # field shape is (1, *per_env_shape) from prototype; replace batch dim with epw
            per_env_shape = field.shape[1:]
            obs_field_specs.append((fname, (epw, *per_env_shape), field.dtype))
        num_actions = proto_masks.shape[1]

        # Resolve VecEnv class path for pickling across spawn
        vec_env_cls_path = f"{vec_env_cls.__module__}:{vec_env_cls.__name__}"

        # Allocate shared memory per worker
        self._all_shm = []
        self._worker_views = []
        self._cmd_pipes = []
        self._result_pipes = []
        self._workers = []

        for w in range(num_workers):
            shm_list = []
            views = {}
            shm_names = {}

            # Obs fields
            for fname, shape, dtype in obs_field_specs:
                key = f"obs_{fname}"
                shm, view = _shm_alloc(shape, dtype)
                shm_list.append((key, shm))
                views[key] = view
                shm_names[key] = (shm.name, list(shape), np.dtype(dtype).str)

            # Control/result buffers
            for key, shape, dtype in [
                ("masks",      (epw, num_actions), np.bool_),
                ("actions",    (epw,),             np.int32),
                ("rewards",    (epw,),             np.float32),
                ("terminated", (epw,),             np.bool_),
                ("seed",       (1,),               np.int64),
                ("infos",      (65536,),           np.uint8),
                ("infos_len",  (1,),               np.int32),
            ]:
                shm, view = _shm_alloc(shape, dtype)
                shm_list.append((key, shm))
                views[key] = view
                shm_names[key] = (shm.name, list(shape), np.dtype(dtype).str)

            self._all_shm.append(shm_list)
            self._worker_views.append(views)

            cmd_recv, cmd_send = ctx.Pipe(duplex=False)
            res_recv, res_send = ctx.Pipe(duplex=False)
            self._cmd_pipes.append(cmd_send)
            self._result_pipes.append(res_recv)

            start = w * epw
            end = start + epw
            p = ctx.Process(
                target=_worker_fn,
                args=(w, start, end, opponent,
                      cmd_recv, res_send, shm_names,
                      self._obs_field_names, vec_env_cls_path, vec_env_kwargs),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Obs NamedTuple class (for reconstructing results)
        self._obs_cls = type(proto_obs)

    def _send_and_wait(self, cmd: int):
        for pipe in self._cmd_pipes:
            pipe.send(cmd)
        for pipe in self._result_pipes:
            pipe.recv()

    def _read_all_obs(self):
        all_obs = {f: [] for f in self._obs_field_names}
        all_masks = []
        for views in self._worker_views:
            for fname in self._obs_field_names:
                all_obs[fname].append(views[f"obs_{fname}"].copy())
            all_masks.append(views["masks"].copy())
        obs = self._obs_cls(**{f: np.concatenate(arrs) for f, arrs in all_obs.items()})
        return obs, np.concatenate(all_masks)

    def reset(self, seed: Optional[int] = None):
        for w, views in enumerate(self._worker_views):
            views["seed"][0] = (seed + w * self._epw) if seed is not None else -1
        self._send_and_wait(CMD_RESET)
        return self._read_all_obs()

    def step(self, actions: np.ndarray):
        for w, views in enumerate(self._worker_views):
            s = w * self._epw
            views["actions"][:] = actions[s:s + self._epw]
        self._send_and_wait(CMD_STEP)

        obs, masks = self._read_all_obs()
        all_r, all_t, all_infos = [], [], []
        for views in self._worker_views:
            all_r.append(views["rewards"].copy())
            all_t.append(views["terminated"].copy())
            n = int(views["infos_len"][0])
            all_infos.extend(pickle.loads(bytes(views["infos"][:n])))

        return (obs, masks, np.concatenate(all_r), np.concatenate(all_t),
                np.zeros(self.num_envs, dtype=bool), all_infos)

    def close(self):
        self._send_and_wait(CMD_SHUTDOWN)
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for shm_list in self._all_shm:
            for _, shm in shm_list:
                shm.close()
                shm.unlink()
