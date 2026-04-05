"""
Multiprocessing vectorized environment wrapper.

Generic — no game-specific assumptions. Wraps a VecSinglePlayerEnv across
worker processes. Agent model + opponent state live on GPU, shared via CUDA IPC.

Obs layout is derived from a prototype reset() call.

Interface:
    obs, masks = vec.reset()
    obs, masks, rewards, terminated, truncated, infos = vec.step(actions)
"""

import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

from .player import BasePlayer, OffsetPlayer
from .vec_env import VecSinglePlayerEnv, EnvFactory
from .key import Key


# ── Commands ────────────────────────────────────────────────────────────────

CMD_STEP = 0
CMD_RESET = 1
CMD_SHUTDOWN = 2


# ── SharedMemory helpers ───────────────────────────────────────────────────

def _shm_alloc(shape, dtype):
    nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
    shm = SharedMemory(create=True, size=nbytes)
    view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    view[:] = 0
    return shm, view


def _shm_attach(name, shape, dtype):
    shm = SharedMemory(name=name, create=False)
    return shm, np.ndarray(shape, dtype=dtype, buffer=shm.buf)


# ── Worker ──────────────────────────────────────────────────────────────────

def _worker_fn(
    worker_id, start_idx, end_idx, batched_player,
    cmd_pipe, result_pipe, shm_names,
    obs_field_names,
    env_factory,     # EnvFactory (picklable OOP object, config only)
    worker_key,      # Key for this worker's envs (converted to RNGs inside)
):
    import traceback
    try:
        torch.set_num_threads(1)

        local_n = end_idx - start_idx
        opp = OffsetPlayer(batched_player, start_idx, local_n)
        env = VecSinglePlayerEnv(num_envs=local_n, opponent=opp,
                                  env_factory=env_factory, key=worker_key)

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
                obs, masks = env.reset()
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
    Multiprocessing vectorized environment.

    Generic: derives obs layout from a prototype reset() call.
    Agent + opponent state on GPU, shared via CUDA IPC.

    Parameters
    ----------
    num_envs       : total number of environments
    num_workers    : number of worker processes
    opponent       : BasePlayer on GPU (shared across workers)
    env_factory    : EnvFactory — creates SinglePlayerEnv per slot (config only)
    key            : Key — master key; children derived for each worker
    """

    def __init__(
        self,
        num_envs: int,
        num_workers: int,
        opponent: BasePlayer,
        env_factory: EnvFactory,
        key: Key,
    ):
        assert num_envs % num_workers == 0
        self.num_envs = num_envs
        self.num_workers = num_workers
        epw = num_envs // num_workers
        self._epw = epw

        ctx = mp.get_context('spawn')

        # Derive keys: one for prototype, one per worker
        all_keys = key.spawn(num_workers + 1)
        proto_key = all_keys[0]
        worker_keys = all_keys[1:]

        # Create prototype to derive obs shapes
        proto_env = VecSinglePlayerEnv(num_envs=1, opponent=opponent,
                                        env_factory=env_factory, key=proto_key)
        proto_obs, proto_masks = proto_env.reset()
        proto_env.close()

        self._obs_field_names = list(proto_obs._fields)
        self._obs_cls = type(proto_obs)
        obs_field_specs = []
        for fname in self._obs_field_names:
            field = getattr(proto_obs, fname)
            per_env_shape = field.shape[1:]
            obs_field_specs.append((fname, (epw, *per_env_shape), field.dtype))
        num_actions = proto_masks.shape[1]

        # Allocate shared memory and spawn workers
        self._all_shm = []
        self._worker_views = []
        self._cmd_pipes = []
        self._result_pipes = []
        self._workers = []

        for w in range(num_workers):
            shm_list = []
            views = {}
            shm_names = {}

            for fname, shape, dtype in obs_field_specs:
                key = f"obs_{fname}"
                shm, view = _shm_alloc(shape, dtype)
                shm_list.append((key, shm))
                views[key] = view
                shm_names[key] = (shm.name, list(shape), np.dtype(dtype).str)

            for key, shape, dtype in [
                ("masks",      (epw, num_actions), np.bool_),
                ("actions",    (epw,),             np.int32),
                ("rewards",    (epw,),             np.float32),
                ("terminated", (epw,),             np.bool_),
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
                      self._obs_field_names, env_factory, worker_keys[w]),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

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

    def reset(self):
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
