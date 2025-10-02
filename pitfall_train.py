# pitfall_train.py
# Rainbow DQN + RND for Pitfall with a simple Go-Explore-style archive
# Python 3.10+, gymnasium[atari]>=0.29, torch>=2.0
#
# Notes:
# - Uses ALE savestates if available (env.unwrapped.ale.cloneState/restoreState).
#   If not available on your platform/build, archive restore will be skipped gracefully.
# - Intrinsic motivation via RND (conv encoder). reward_used = extrinsic + beta * rnd_bonus.
# - Rainbow bits: Dueling + Noisy layers, PER, n-step targets, Double-DQN, target net.
# - Action pruning: reduce from 18 to a compact set that’s useful in Pitfall.
# - Deterministic exploration (v4, sticky=0) is recommended for archive building.
#   Then train/validate on v5 (sticky=0.25) for robustness.

import argparse
import collections
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium.wrappers import AtariPreprocessing, FrameStack

# ===========================
# Tunables (safe starting points)
# ===========================
ENV_ID_EXPLORE = "PitfallNoFrameskip-v4"   # sticky=0 (deterministic), good for building archive
ENV_ID_TRAIN   = "ALE/Pitfall-v5"          # sticky=0.25 (default), robustify here
TOTAL_FRAMES   = 2_000_000                 # training budget
LEARNING_START = 20_000                    # warm-up frames (collect experience first)
TARGET_UPDATE  = 10_000                    # target net sync
TRAIN_FREQ     = 4                         # optimize every K frames
GAMMA          = 0.99
N_STEPS        = 3                         # n-step returns
BATCH_SIZE     = 64
LR             = 2.5e-4
PER_ALPHA      = 0.5
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_BETA_FRAMES = TOTAL_FRAMES
BUFFER_CAP     = 1_000_000
EPS_START      = 1.0
EPS_END        = 0.01
EPS_DECAY_FRAMES = 1_000_000
FRAME_SKIP     = 4
RND_BETA       = 0.2                       # intrinsic reward scaling
RND_LR         = 1e-4

ARCHIVE_JUMP_EVERY = 20_000                # how often to jump to an archived cell (frames)
ARCHIVE_EXPLORE_STEPS = 2_000              # steps after a jump with higher epsilon
CELL_X_BUCKETS = 10
CELL_Y_BUCKETS = 6

EVAL_EVERY     = 100_000
EVAL_EPISODES  = 3

SEED           = 1337
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================
# Helpers
# ===========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_anneal(start: float, end: float, curr: int, total: int) -> float:
    t = min(1.0, curr / max(1, total))
    return start + t * (end - start)

def to_chw(obs) -> np.ndarray:
    """
    Return channel-first (4,84,84) uint8 from Gym/Gymnasium observations.
    Works with LazyFrames and ndarray.
    """
    # Convert LazyFrames -> ndarray without copy if possible
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs)  # triggers LazyFrames.__array__()

    if obs.ndim != 3:
        raise ValueError(f"Unexpected obs ndim: {obs.ndim}, shape={obs.shape}")

    # HWC -> CHW
    if obs.shape == (84, 84, 4):
        obs = np.transpose(obs, (2, 0, 1))
    elif obs.shape != (4, 84, 84):
        raise ValueError(f"Unexpected obs shape: {obs.shape}")

    # Ensure dtype/contiguity for torch.from_numpy
    if obs.dtype != np.uint8:
        obs = obs.astype(np.uint8, copy=False)
    if not obs.flags.c_contiguous:
        obs = np.ascontiguousarray(obs)
    return obs

# ===========================
# Reduced action wrapper
# ===========================
class ReducedActionSet(gym.ActionWrapper):
    """
    Map a compact action set index -> full 18-action space index.
    Pitfall useful controls: idle, left, right, up, down, jump, left+jump, right+jump, up+jump, down+jump.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Full meaning list for ALE (18 actions)
        # 0: NOOP, 1:FIRE, 2:UP, 3:RIGHT, 4:LEFT, 5:DOWN, 6:UPRIGHT, 7:UPLEFT,
        # 8:DOWNRIGHT, 9:DOWNLEFT, 10:UPFIRE, 11:RIGHTFIRE, 12:LEFTFIRE,
        # 13:DOWNFIRE, 14:UPRIGHTFIRE, 15:UPLEFTFIRE, 16:DOWNRIGHTFIRE, 17:DOWNLEFTFIRE
        self.map = np.array([
            0,   # 0 NOOP
            3,   # 1 RIGHT
            4,   # 2 LEFT
            2,   # 3 UP (ladder/vine)
            5,   # 4 DOWN (ladder)
            1,   # 5 JUMP (FIRE)
            11,  # 6 RIGHT+JUMP
            12,  # 7 LEFT+JUMP
            10,  # 8 UP+JUMP
            13,  # 9 DOWN+JUMP
        ], dtype=np.int64)
        self.action_space = gym.spaces.Discrete(len(self.map))

    def action(self, a):
        return int(self.map[a])

# ===========================
# Env builder
# ===========================
def make_env(env_id: str, seed: int, render: bool = False) -> gym.Env:
    env = gym.make(env_id, render_mode=("human" if render else None), frameskip=1)
    # AtariPreprocessing: scale to 84x84 grayscale, frame_skip inside wrapper (we keep as provided by env_id)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, 4)
    env = ReducedActionSet(env)
    env.reset(seed=seed)
    return env

# ===========================
# Rainbow Network (Dueling + Noisy)
# ===========================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class QNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Input: (B, 4, 84,84) grayscale stack
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(inplace=True),
        )
        self.conv_out = 3136  # 64*7*7
        # Dueling heads with Noisy layers
        self.val = nn.Sequential(
            NoisyLinear(self.conv_out, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, 1),
        )
        self.adv = nn.Sequential(
            NoisyLinear(self.conv_out, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        h = self.conv(x).view(x.size(0), -1)
        v = self.val(h)
        a = self.adv(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ===========================
# RND (Random Network Distillation)
# ===========================
class RNDTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )
        for p in self.parameters():
            p.requires_grad_(False)  # fixed random target


class RNDPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x_float01: torch.Tensor):
        # x_float01 is float32 scaled to [0,1], shape (B,4,84,84)
        return self.net(x_float01)

class RNDModule:
    def __init__(self, beta=RND_BETA, lr=RND_LR):
        self.target = RNDTarget().to(DEVICE)
        self.pred = RNDPredictor().to(DEVICE)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=lr)
        self.beta = beta

    @torch.no_grad()
    def bonus(self, obs_4x84x84: np.ndarray) -> float:
        x = torch.from_numpy(obs_4x84x84).unsqueeze(0).to(DEVICE).float().div_(255.0)
        t = self.target.enc(x)
        p = self.pred(x)
        return F.mse_loss(p, t, reduction="mean").item()

    def update_batch(self, obs_batch: torch.Tensor) -> float:
        # obs_batch: (B,4,84,84) uint8
        x = obs_batch.to(DEVICE).float().div(255.0)
        with torch.no_grad():
            tgt = self.target.enc(x)
        pred = self.pred(x)
        loss = F.mse_loss(pred, tgt)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

# ===========================
# Prioritized Replay with n-step
# ===========================
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool
    intr: float

class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf: Deque[Transition] = collections.deque()

    def push(self, tr: Transition):
        self.buf.append(tr)

    def pop_ready(self) -> Optional[Transition]:
        if len(self.buf) < self.n:
            return None
        R, Intr = 0.0, 0.0
        for i, tr in enumerate(self.buf):
            R += (self.gamma ** i) * tr.r
            Intr += (self.gamma ** i) * tr.intr
            if tr.d:
                # truncate on early done
                break
        s0, a0 = self.buf[0].s, self.buf[0].a
        sN, dN = self.buf[-1].s2, any(t.d for t in self.buf)
        # shift one
        self.buf.popleft()
        return Transition(s0, a0, R, sN, dN, Intr)

    def flush(self):
        while len(self.buf) > 0:
            yield self.pop_ready()

class SumTree:
    def __init__(self, capacity):
        self.n = 1
        while self.n < capacity:
            self.n <<= 1
        self.size = capacity
        self.tree = np.zeros(2 * self.n)
        self.data = [None] * capacity
        self.ptr = 0
        self.count = 0

    def add(self, p, data):
        i = self.ptr
        self.data[i] = data
        self.update(i, p)
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def update(self, idx, p):
        pos = idx + self.n
        self.tree[pos] = p
        pos >>= 1
        while pos:
            self.tree[pos] = self.tree[pos << 1] + self.tree[(pos << 1) | 1]
            pos >>= 1

    def total(self):
        return self.tree[1]

    def get(self, v):
        pos = 1
        while pos < self.n:
            left = pos << 1
            if self.tree[left] >= v:
                pos = left
            else:
                v -= self.tree[left]
                pos = left | 1
        idx = pos - self.n
        idx = min(idx, self.count - 1)
        return idx, self.tree[pos], self.data[idx]

class PERBuffer:
    def __init__(self, capacity, alpha=0.5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_p = 1.0

    def push(self, tr: Transition):
        self.tree.add(self.max_p, tr)

    def sample(self, batch_size, beta):
        total = self.tree.total()
        seg = total / batch_size
        samples = []
        idxs = []
        probs = []
        for i in range(batch_size):
            v = random.random() * seg + i * seg
            idx, p, data = self.tree.get(v)
            idxs.append(idx)
            samples.append(data)
            probs.append(p / total)
        weights = (np.array(probs) * self.tree.count) ** (-beta)
        weights /= weights.max() + 1e-8
        return idxs, samples, torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            p = float(p)
            self.tree.update(i, (p + 1e-6) ** self.alpha)
            self.max_p = max(self.max_p, p + 1e-6)

    def __len__(self):
        return self.tree.count

# ===========================
# Agent
# ===========================
class RainbowAgent:
    def __init__(self, action_dim: int):
        self.q = QNet(action_dim).to(DEVICE)
        self.tgt = QNet(action_dim).to(DEVICE)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.action_dim = action_dim
        self.steps = 0

    @torch.no_grad()
    def act(self, s: np.ndarray, epsilon: float) -> int:
        # s: (4,84,84) uint8
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        x = torch.from_numpy(s).unsqueeze(0).to(DEVICE)
        q = self.q(x)
        return int(q.argmax(dim=1).item())

    def update(self, batch: List[Transition], weights: torch.Tensor) -> np.ndarray:
        s = torch.from_numpy(np.stack([b.s for b in batch])).to(DEVICE)  # (B,4,84,84), u8
        a = torch.tensor([b.a for b in batch], device=DEVICE, dtype=torch.long)  # (B,)
        r = torch.tensor([b.r + RND_BETA * b.intr for b in batch], device=DEVICE, dtype=torch.float32)
        s2 = torch.from_numpy(np.stack([b.s2 for b in batch])).to(DEVICE)
        d = torch.tensor([b.d for b in batch], device=DEVICE, dtype=torch.float32)

        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN target
            a2 = self.q(s2).argmax(dim=1)
            q2 = self.tgt(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            target = r + (1.0 - d) * (GAMMA ** N_STEPS) * q2

        td = target - q
        loss = (weights * (td ** 2)).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        # Reset Noisy layers' noise after each update
        self.q.reset_noise()
        self.tgt.reset_noise()

        prios = td.abs().detach().cpu().numpy()
        return prios

    def sync_target(self):
        self.tgt.load_state_dict(self.q.state_dict())

# ===========================
# Simple "cell" key + archive (Go-Explore-ish)
# ===========================
def cell_key(obs: np.ndarray) -> Tuple[int, int, int]:
    """
    Make a coarse "room-ish" key from the observation:
    - compute Pitfall Harry approx position from foreground centroid (cheap proxy)
    - bucket x/y into coarse bins; include a crude "background hash" to separate rooms
    This is intentionally simple; RAM-based keys are also possible if you read RAM.
    """
    # obs: (4,84,84) uint8; use the latest frame (index -1 if stacked channel-first)
    # Our stack is channel-first; use the last channel as "current frame".
    frame = obs[-1]  # (84,84)
    # background hash: downsample and hash
    small = frame[::12, ::12]
    bg = int(np.sum((small > 50).astype(np.int32)))  # cheap count
    bg_mod = bg % 16  # keep small
    # position proxy: bright pixels centroid
    ys, xs = np.where(frame > 160)
    if len(xs) == 0:
        cx, cy = 42, 42
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
    x_bin = min(CELL_X_BUCKETS - 1, cx * CELL_X_BUCKETS // 84)
    y_bin = min(CELL_Y_BUCKETS - 1, cy * CELL_Y_BUCKETS // 84)
    return (bg_mod, x_bin, y_bin)


class Archive:
    def __init__(self):
        self.cells: Dict[Tuple[int, int, int], Tuple[Optional[bytes], np.ndarray, int]] = {}
        # key -> (ale_state_bytes or None, obs, visits)

    def add(self, key, ale_state: Optional[bytes], obs: np.ndarray):
        if key not in self.cells:
            self.cells[key] = (ale_state, obs.copy(), 0)

    def sample(self) -> Optional[Tuple[Tuple[int, int, int], Optional[bytes]]]:
        if not self.cells:
            return None
        # bias towards lower-visit cells
        items = list(self.cells.items())
        visits = np.array([v[2] for _, v in items], dtype=np.float32)
        probs = (visits + 1.0) ** -1.0
        probs /= probs.sum()
        idx = np.random.choice(len(items), p=probs)
        k, (st, _, v) = items[idx]
        # increment visit count
        self.cells[k] = (st, self.cells[k][1], v + 1)
        return k, st

# ===========================
# Training / Evaluation
# ===========================
def evaluate(agent: RainbowAgent, make_env_fn, episodes=3, seed=SEED) -> float:
    env = make_env_fn(render=False)
    scores = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        s = to_chw(s)  # (84,84,4)->(4,84,84)
        done = False
        score = 0.0
        while not done:
            a = agent.act(s, epsilon=0.001)
            s2, r, terminated, truncated, _ = env.step(a)
            s2 = to_chw(s2)
            done = terminated or truncated
            score += r
            s = s2
        scores.append(score)
    env.close()
    return float(np.mean(scores))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_explore", type=str, default=ENV_ID_EXPLORE)
    parser.add_argument("--env_train", type=str, default=ENV_ID_TRAIN)
    parser.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # Phase 1/2 simplified in one loop: we build archive (deterministic env recommended)
    # and also train on sticky action env for robustness. You can run two separate phases if you like.
    env = make_env(args.env_explore, args.seed, render=True)
    env_train = make_env(args.env_train, args.seed + 123, render=False)

    action_dim = env.action_space.n
    agent = RainbowAgent(action_dim)
    rnd = RNDModule(beta=RND_BETA, lr=RND_LR)

    per = PERBuffer(BUFFER_CAP, alpha=PER_ALPHA)
    nbuf = NStepBuffer(N_STEPS, GAMMA)
    archive = Archive()

    eps = EPS_START
    explore_boost_left = 0
    last_eval = 0

    s, _ = env.reset(seed=args.seed)
    s = to_chw(s)  # (84,84,4)->(4,84,84)

    # if ALE save/restore available?
    def clone_state_if_possible(e):
        try:
            return e.unwrapped.ale.cloneState()
        except Exception:
            return None

    def restore_state_if_possible(e, st):
        if st is None:
            return False
        try:
            e.unwrapped.ale.restoreState(st)
            return True
        except Exception:
            return False

    frame_count = 0
    episode = 0
    ep_reward = 0.0
    t0 = time.time()

    while frame_count < args.frames:
        # Epsilon schedule; if exploring from archive, temporarily increase ε
        base_eps = linear_anneal(EPS_START, EPS_END, frame_count, EPS_DECAY_FRAMES)
        eps = max(base_eps, 0.05) if explore_boost_left > 0 else base_eps

        # Select action
        a = agent.act(s, epsilon=eps)

        # Step env
        s2, r, terminated, truncated, _ = env.step(a)
        s2 = to_chw(s2)
        done = terminated or truncated

        # Intrinsic reward from RND (on next obs is common; either is fine)
        intr = rnd.bonus(s2)

        # Cache transition via n-step buffer
        nbuf.push(Transition(s, a, float(r), s2, done, intr))
        tr_n = nbuf.pop_ready()
        if tr_n is not None:
            per.push(tr_n)

        # Add to archive
        key = cell_key(s2)
        ale_state = clone_state_if_possible(env)
        archive.add(key, ale_state, s2)

        ep_reward += r
        s = s2
        frame_count += 1

        # Training step
        if frame_count > LEARNING_START and frame_count % TRAIN_FREQ == 0 and len(per) >= BATCH_SIZE:
            beta = linear_anneal(PER_BETA_START, PER_BETA_END, frame_count, PER_BETA_FRAMES)
            idxs, batch, weights = per.sample(BATCH_SIZE, beta)
            prios = agent.update(batch, weights)
            per.update_priorities(idxs, prios)
            # RND update on the same batch next-states for stability
            obs_batch = torch.from_numpy(np.stack([b.s2 for b in batch])).to(DEVICE)
            rnd.update_batch(obs_batch)

        # Target sync
        if frame_count % TARGET_UPDATE == 0:
            agent.sync_target()

        # Periodic archive jump to explore new areas deterministically
        if ARCHIVE_JUMP_EVERY > 0 and frame_count % ARCHIVE_JUMP_EVERY == 0:
            pick = archive.sample()
            if pick is not None:
                _, st = pick
                ok = restore_state_if_possible(env, st)
                if ok:
                    # Boost exploration for a short window
                    explore_boost_left = ARCHIVE_EXPLORE_STEPS

        if explore_boost_left > 0:
            explore_boost_left -= 1

        # Episode end
        if done:
            episode += 1
            fps = int(frame_count / max(1e-3, (time.time() - t0)))
            print(f"[{frame_count}/{args.frames}] Episode={episode} Reward={ep_reward:.1f} eps={eps:.3f} "
                  f"Buffer={len(per)} Archive={len(archive.cells)} fps~{fps}")
            s, _ = env.reset()
            s = to_chw(s)
            ep_reward = 0.0

        # Periodic evaluation on sticky action env
        if frame_count - last_eval >= EVAL_EVERY:
            last_eval = frame_count
            avg = evaluate(agent, lambda **kw: make_env(args.env_train, args.seed + 999, **kw),
                           episodes=EVAL_EPISODES, seed=args.seed + 999)
            print(f"   >>> Eval@{frame_count}: avg score over {EVAL_EPISODES} eps = {avg:.1f}")

    # Flush remaining n-step transitions into buffer (optional)
    for tr in nbuf.flush():
        if tr is not None:
            per.push(tr)

    print("Training complete.")
    env.close()
    env_train.close()
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"q": agent.q.state_dict()}, "checkpoints/pitfall_rainbow_rnd.pt")
    print("Saved to checkpoints/pitfall_rainbow_rnd.pt")


if __name__ == "__main__":
    train()
