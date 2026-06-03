"""
Multiprocessing rollout workers for PPO+LSTM training.

Each worker process independently:
  1. Steps its own envs (JIT-compiled)
  2. Runs agent + opponent NN inference (its own GPU context)
  3. Writes rollout data into shared memory buffers

The main process:
  1. Waits for all workers to finish a rollout
  2. Reads trajectories from shared memory
  3. Runs PPO update
  4. Broadcasts new weights to workers

Uses torch.multiprocessing with spawn to avoid CUDA fork issues.
"""

import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils._pytree import tree_map

from .ppo_lstm import obs_to_tensor, make_lstm_state, LSTMState


def _worker_fn(rank, num_workers, envs_per_worker, num_steps, num_actions,
               agent_cls, agent_kwargs, factory_cls, factory_kwargs,
               seed, device_str,
               # Shared memory buffers (allocated by main process)
               shm_obs_fields,  # dict: field_name -> (shape, dtype, tensor)
               shm_masks,       # (num_workers, num_steps, envs_per_worker, num_actions)
               shm_actions,     # (num_workers, num_steps, envs_per_worker)
               shm_logprobs,    # (num_workers, num_steps, envs_per_worker)
               shm_rewards,     # (num_workers, num_steps, envs_per_worker)
               shm_dones,       # (num_workers, num_steps, envs_per_worker)
               shm_values,      # (num_workers, num_steps, envs_per_worker)
               # For LSTM: initial and final states
               shm_lstm_h,      # (num_workers, num_layers, envs_per_worker, lstm_hidden)
               shm_lstm_c,
               shm_final_lstm_h,
               shm_final_lstm_c,
               shm_final_obs_fields,  # next_obs after rollout
               shm_final_masks,
               shm_final_dones,
               # Synchronization
               weight_version,   # mp.Value: current weight version
               worker_ready,     # mp.Barrier: workers signal rollout done
               start_barrier,    # mp.Barrier: wait for main to signal start
               weight_lock,      # mp.Lock: protect weight reads
               weight_path,      # str: path to shared weight file (initial load)
               shared_weights,   # dict: param_name -> shared memory tensor
               stop_flag,        # mp.Value: signal workers to exit
               ):
    """Worker process: collect rollouts independently."""
    try:
        # Seed Numba RNG uniquely per worker
        from game.lc.engine import seed_numba_rng
        seed_numba_rng(seed + rank * 1000)

        device = torch.device(device_str)
        N = envs_per_worker

        # Create agent (own copy for inference)
        agent = agent_cls(**agent_kwargs).to(device)
        agent.eval()

        # Create envs
        from abstract.vec_env import VecSinglePlayerEnv
        from abstract import LSTMBatchedPlayer
        from abstract.key import key_from_seed

        opponent = LSTMBatchedPlayer(agent, device, num_envs=N)
        factory = factory_cls(**factory_kwargs)
        envs = VecSinglePlayerEnv(
            num_envs=N, opponent=opponent,
            env_factory=factory, key=key_from_seed(seed + rank),
        )

        obs_np, masks_np = envs.reset()
        next_obs = obs_to_tensor(obs_np, device)
        next_masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
        next_done = torch.zeros(N, device=device)
        lstm_state = make_lstm_state(agent.lstm_layers, N, agent.lstm_hidden, device)

        local_weight_ver = -1  # force initial load

        while not stop_flag.value:
            # Wait for main to signal start
            start_barrier.wait()
            if stop_flag.value:
                break

            # Load weights from shared memory if updated
            with weight_lock:
                ver = weight_version.value
            if ver != local_weight_ver:
                if local_weight_ver == -1:
                    # First load from file (shared_weights may not be populated yet)
                    sd = torch.load(weight_path, map_location=device, weights_only=True)
                    agent.load_state_dict(sd)
                else:
                    # Fast load from shared memory (zero-copy read)
                    sd = {k: v.to(device) for k, v in shared_weights.items()}
                    agent.load_state_dict(sd)
                local_weight_ver = ver

            # Read initial LSTM state from shared memory
            lstm_state = LSTMState(
                h=shm_lstm_h[rank].clone().to(device),
                c=shm_lstm_c[rank].clone().to(device),
            )

            # ── Rollout ──
            for step in range(num_steps):
                # Write obs to shared memory
                obs_np_step = tree_map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, next_obs)
                for field_name, shm_t in shm_obs_fields.items():
                    src = getattr(obs_np_step, field_name) if hasattr(obs_np_step, field_name) else getattr(next_obs, field_name)
                    if isinstance(src, torch.Tensor):
                        shm_t[rank, step].copy_(src)
                    else:
                        shm_t[rank, step].copy_(torch.as_tensor(src))

                shm_dones[rank, step].copy_(next_done.cpu())
                shm_masks[rank, step].copy_(next_masks.cpu())

                with torch.no_grad():
                    action, logprob, _, value, lstm_state = agent.get_action_and_value(
                        next_obs, next_masks, lstm_state, next_done)

                shm_values[rank, step].copy_(value.flatten().cpu())
                shm_actions[rank, step].copy_(action.cpu())
                shm_logprobs[rank, step].copy_(logprob.cpu())

                # Step envs
                obs_np, masks_np, rew, term, trunc, infos = envs.step(action.cpu().numpy())
                done_np = np.logical_or(term, trunc)
                done_indices = list(np.where(done_np)[0])
                if done_indices:
                    opponent.reset(done_indices)

                shm_rewards[rank, step].copy_(torch.tensor(rew, dtype=torch.float32))

                next_obs = obs_to_tensor(obs_np, device)
                next_masks = torch.as_tensor(masks_np, dtype=torch.bool, device=device)
                next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Write final state to shared memory
            shm_final_lstm_h[rank].copy_(lstm_state.h.cpu())
            shm_final_lstm_c[rank].copy_(lstm_state.c.cpu())
            final_obs_np = tree_map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, next_obs)
            for field_name, shm_t in shm_final_obs_fields.items():
                src = getattr(final_obs_np, field_name) if hasattr(final_obs_np, field_name) else getattr(next_obs, field_name)
                if isinstance(src, torch.Tensor):
                    shm_t[rank].copy_(src)
                else:
                    shm_t[rank].copy_(torch.as_tensor(src))
            shm_final_masks[rank].copy_(next_masks.cpu())
            shm_final_dones[rank].copy_(next_done.cpu())

            # Signal done
            worker_ready.wait()

    except Exception as e:
        import traceback
        print(f"[Worker {rank}] ERROR: {e}")
        traceback.print_exc()
        worker_ready.wait()  # don't deadlock other workers


class MPRolloutManager:
    """Manages multiprocessing rollout workers for PPO training."""

    def __init__(self, num_workers, envs_per_worker, num_steps, num_actions,
                 agent_cls, agent_kwargs, agent,
                 factory_cls, factory_kwargs,
                 seed, device, lstm_layers, lstm_hidden):
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.num_envs = num_workers * envs_per_worker
        self.num_steps = num_steps
        self.num_actions = num_actions
        self.device = device
        self.agent = agent
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

        N = envs_per_worker
        T = num_steps
        W = num_workers

        # Shared weight tensors (zero-copy via shared memory)
        self._weight_path = f"/tmp/ppo_weights_{os.getpid()}.pt"
        sd = agent.state_dict()
        self._shared_weights = {}
        for k, v in sd.items():
            self._shared_weights[k] = v.cpu().share_memory_()
        # Also save initial file for first load
        torch.save(sd, self._weight_path)

        # Synchronization primitives
        self._weight_version = mp.Value('i', 0)
        self._worker_ready = mp.Barrier(W + 1)  # W workers + main
        self._start_barrier = mp.Barrier(W + 1)
        self._weight_lock = mp.Lock()
        self._stop_flag = mp.Value('i', 0)

        # Allocate shared memory buffers
        # We need a prototype obs to know field shapes
        from abstract.key import key_from_seed
        factory = factory_cls(**factory_kwargs)
        tmp_env = factory.create(np.random.default_rng(0))
        tmp_env.reset()
        proto_obs = tmp_env.encode(0)

        self._obs_fields = {}
        self._final_obs_fields = {}
        obs_cls = type(proto_obs)
        self._obs_cls = obs_cls
        for field in obs_cls._fields:
            v = getattr(proto_obs, field)
            shape = (W, T, N) + v.shape
            dtype = torch.long if np.issubdtype(v.dtype, np.integer) else torch.float32
            t = torch.zeros(shape, dtype=dtype).share_memory_()
            self._obs_fields[field] = t
            # Final obs (no T dim)
            ft = torch.zeros((W, N) + v.shape, dtype=dtype).share_memory_()
            self._final_obs_fields[field] = ft

        self._masks = torch.zeros(W, T, N, num_actions, dtype=torch.bool).share_memory_()
        self._actions = torch.zeros(W, T, N, dtype=torch.long).share_memory_()
        self._logprobs = torch.zeros(W, T, N).share_memory_()
        self._rewards = torch.zeros(W, T, N).share_memory_()
        self._dones = torch.zeros(W, T, N).share_memory_()
        self._values = torch.zeros(W, T, N).share_memory_()

        self._lstm_h = torch.zeros(W, lstm_layers, N, lstm_hidden).share_memory_()
        self._lstm_c = torch.zeros(W, lstm_layers, N, lstm_hidden).share_memory_()
        self._final_lstm_h = torch.zeros(W, lstm_layers, N, lstm_hidden).share_memory_()
        self._final_lstm_c = torch.zeros(W, lstm_layers, N, lstm_hidden).share_memory_()
        self._final_masks = torch.zeros(W, N, num_actions, dtype=torch.bool).share_memory_()
        self._final_dones = torch.zeros(W, N).share_memory_()

        # Spawn workers
        device_str = str(device)
        self._workers = []
        for rank in range(W):
            p = mp.Process(target=_worker_fn, args=(
                rank, W, N, T, num_actions,
                agent_cls, agent_kwargs, factory_cls, factory_kwargs,
                seed, device_str,
                self._obs_fields, self._masks, self._actions,
                self._logprobs, self._rewards, self._dones, self._values,
                self._lstm_h, self._lstm_c,
                self._final_lstm_h, self._final_lstm_c,
                self._final_obs_fields, self._final_masks, self._final_dones,
                self._weight_version, self._worker_ready,
                self._start_barrier, self._weight_lock,
                self._weight_path, self._shared_weights, self._stop_flag,
            ))
            p.start()
            self._workers.append(p)

    def broadcast_weights(self):
        """Copy current agent weights into shared memory tensors."""
        with self._weight_lock:
            sd = self.agent.state_dict()
            for k, v in sd.items():
                self._shared_weights[k].copy_(v.cpu())
            self._weight_version.value += 1

    def set_initial_lstm_state(self, lstm_state: LSTMState):
        """Set LSTM state for all workers (reshaped from (layers, total_envs, H))."""
        N = self.envs_per_worker
        for w in range(self.num_workers):
            self._lstm_h[w].copy_(lstm_state.h[:, w*N:(w+1)*N].cpu())
            self._lstm_c[w].copy_(lstm_state.c[:, w*N:(w+1)*N].cpu())

    def collect_rollout(self):
        """Signal workers to collect a rollout, wait for completion, return data on device."""
        # Signal workers to start
        self._start_barrier.wait()

        # Wait for all workers to finish
        self._worker_ready.wait()

        # Reshape from (W, T, N, ...) → (T, W*N, ...)
        W, T, N = self.num_workers, self.num_steps, self.envs_per_worker
        device = self.device

        obs = self._obs_cls(**{
            field: t.reshape(W, T, N, *t.shape[3:]).permute(1, 0, 2, *range(3, t.dim())).reshape(T, W*N, *t.shape[3:]).to(device)
            for field, t in self._obs_fields.items()
        })
        masks = self._masks.reshape(W, T, N, -1).permute(1, 0, 2, 3).reshape(T, W*N, -1).to(device)
        actions = self._actions.reshape(W, T, N).permute(1, 0, 2).reshape(T, W*N).to(device)
        logprobs = self._logprobs.reshape(W, T, N).permute(1, 0, 2).reshape(T, W*N).to(device)
        rewards = self._rewards.reshape(W, T, N).permute(1, 0, 2).reshape(T, W*N).to(device)
        dones = self._dones.reshape(W, T, N).permute(1, 0, 2).reshape(T, W*N).to(device)
        values = self._values.reshape(W, T, N).permute(1, 0, 2).reshape(T, W*N).to(device)

        # Final states (after rollout)
        final_obs = self._obs_cls(**{
            field: t.reshape(W*N, *t.shape[2:]).to(device)
            for field, t in self._final_obs_fields.items()
        })
        final_masks = self._final_masks.reshape(W*N, -1).to(device)
        final_dones = self._final_dones.reshape(W*N).to(device)

        # LSTM states: reshape (W, layers, N, H) → (layers, W*N, H)
        initial_lstm = LSTMState(
            h=self._lstm_h.permute(1, 0, 2, 3).reshape(self.lstm_layers, W*N, self.lstm_hidden).to(device),
            c=self._lstm_c.permute(1, 0, 2, 3).reshape(self.lstm_layers, W*N, self.lstm_hidden).to(device),
        )
        final_lstm = LSTMState(
            h=self._final_lstm_h.permute(1, 0, 2, 3).reshape(self.lstm_layers, W*N, self.lstm_hidden).to(device),
            c=self._final_lstm_c.permute(1, 0, 2, 3).reshape(self.lstm_layers, W*N, self.lstm_hidden).to(device),
        )

        return obs, masks, actions, logprobs, rewards, dones, values, \
               initial_lstm, final_lstm, final_obs, final_masks, final_dones

    def shutdown(self):
        """Stop all workers."""
        self._stop_flag.value = 1
        try:
            self._start_barrier.wait(timeout=5)
        except Exception:
            pass
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        if os.path.exists(self._weight_path):
            os.remove(self._weight_path)
