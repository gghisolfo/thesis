"""
Train DQN on ArkanoidEnv using God Acts:
- ArkanoidEnv (pygame)
- GodActRule, GodActRewardShaper, GodActPrioritizedReplayBuffer, GodActCurriculumGenerator
- GodActEnvWrapper
- Simple PyTorch DQN with epsilon-greedy
"""

import json
import math
import random
from collections import deque
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import pygame
import time
import os

RULES_PATH"god_acts_rules.json"

# ---------------------------
# ArkanoidEnv (identico al tuo)
# ---------------------------
class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, width=600, height=400, n_bricks_rows=3, n_bricks_cols=8):
        super().__init__()
        self.width = width
        self.height = height
        self.n_brick_rows = n_bricks_rows
        self.n_brick_cols = n_bricks_cols
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.paddle_w = 0.15
        self.paddle_x = 0.5
        self.paddle_speed = 0.04
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = 0.02
        self.ball_vy = -0.03
        self.ball_speed_cap = 0.06
        self.bricks = None
        self.max_steps = 2000
        self.step_count = 0

        # pygame setup
        self.screen = None
        self.clock = None

    def reset(self):
        self.paddle_x = 0.5
        self.ball_x = 0.5 + (random.random() - 0.5) * 0.1
        self.ball_y = 0.5
        angle = random.uniform(-math.pi / 4, math.pi / 4) + math.pi / 2
        speed = 0.035
        self.ball_vx = speed * math.cos(angle)
        self.ball_vy = -abs(speed * math.sin(angle))
        self.bricks = np.ones((self.n_brick_rows, self.n_brick_cols), dtype=np.int8)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(
            [
                self.ball_x * 2 - 1,
                self.ball_y * 2 - 1,
                self.ball_vx * 2 / self.ball_speed_cap,
                self.ball_vy * 2 / self.ball_speed_cap,
                self.paddle_x * 2 - 1,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        done = False

        if action == 0:
            self.paddle_x -= self.paddle_speed
        elif action == 2:
            self.paddle_x += self.paddle_speed
        self.paddle_x = float(np.clip(self.paddle_x, 0.0, 1.0))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x <= 0.0 or self.ball_x >= 1.0:
            self.ball_vx *= -1
        if self.ball_y <= 0.0:
            self.ball_vy *= -1

        paddle_y = 0.95
        if self.ball_y >= paddle_y - 0.01:
            half_pw = self.paddle_w / 2
            if (
                (self.ball_x >= (self.paddle_x - half_pw))
                and (self.ball_x <= (self.paddle_x + half_pw))
                and self.ball_vy > 0
            ):
                self.ball_vy *= -1
                offset = (self.ball_x - self.paddle_x) / half_pw
                self.ball_vx += offset * 0.015
                v = math.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
                if v > self.ball_speed_cap:
                    self.ball_vx *= self.ball_speed_cap / v
                    self.ball_vy *= self.ball_speed_cap / v
                reward += 1.0
            elif self.ball_y > 1.0:
                reward -= 10.0
                done = True

        brick_top, brick_bottom = 0.05, 0.25
        if brick_top <= self.ball_y <= brick_bottom:
            col = int(self.ball_x * self.n_brick_cols)
            row = int(
                (self.ball_y - brick_top)
                / (brick_bottom - brick_top)
                * self.n_brick_rows
            )
            if 0 <= row < self.n_brick_rows and 0 <= col < self.n_brick_cols:
                if self.bricks[row, col] == 1:
                    self.bricks[row, col] = 0
                    reward += 2.0
                    self.ball_vy *= -1

        reward -= abs(self.paddle_x - self.ball_x) * 0.05
        reward += 0.1

        if np.sum(self.bricks) == 0:
            reward += 50.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Arkanoid DQN")
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))
        scale_x, scale_y = self.width, self.height

        # Draw ball
        bx, by = int(self.ball_x * scale_x), int(self.ball_y * scale_y)
        pygame.draw.circle(self.screen, (255, 255, 255), (bx, by), 5)

        # Draw paddle
        paddle_y = int(0.95 * scale_y)
        pw = int(self.paddle_w * scale_x)
        px = int(self.paddle_x * scale_x)
        pygame.draw.rect(
            self.screen,
            (100, 200, 255),
            (px - pw // 2, paddle_y, pw, 10),
        )

        # Draw bricks
        bw = scale_x / self.n_brick_cols
        bh = (scale_y * 0.2) / self.n_brick_rows
        for r in range(self.n_brick_rows):
            for c in range(self.n_brick_cols):
                if self.bricks[r, c] == 1:
                    rect = pygame.Rect(c * bw, r * bh + 30, bw - 2, bh - 2)
                    pygame.draw.rect(self.screen, (255, 80, 80), rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()

    # helpers for wrapper
    def reset_with_params(self, params: Dict):
        # set ball/paddle from curriculum and return obs
        self.paddle_x = params.get('paddle_x', 0.5)
        self.ball_x = params.get('ball_x', 0.5)
        self.ball_y = params.get('ball_y', 0.5)
        self.ball_vx = params.get('ball_vx', 0.02)
        self.ball_vy = params.get('ball_vy', -0.03)
        self.bricks = np.ones((self.n_brick_rows, self.n_brick_cols), dtype=np.int8)
        self.step_count = 0
        return self._get_obs()

    def get_observation(self):
        return self._get_obs()

# ---------------------------
# God Acts classes (essenziali)
# ---------------------------
class GodActRule:
    def __init__(self, rule_data: dict):
        self.rule_id = rule_data['rule_id']
        self.trigger = rule_data['trigger']
        self.effect = rule_data.get('effect', '')
        self.confidence = rule_data.get('confidence', 1.0)
        self.priority = rule_data.get('priority', 1.0)
        self.reward_modifier = rule_data.get('reward_modifier', 1.0)
        self.conditions = rule_data.get('conditions', {})

    def matches_state_transition(self, state, next_state, reward) -> bool:
        if state is None or next_state is None:
            return False
        if self.trigger == 'ball_paddle_collision':
            return self._check_ball_paddle_collision(state, next_state)
        elif self.trigger == 'ball_brick_collision':
            return self._check_ball_brick_collision(state, next_state, reward)
        elif self.trigger == 'ball_wall_collision':
            return self._check_ball_wall_collision(state, next_state)
        elif self.trigger == 'ball_lost':
            return reward < -5
        return False

    def _check_ball_paddle_collision(self, state, next_state) -> bool:
        # state = [ball_x, ball_y, ball_vx, ball_vy, paddle_x]
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        next_ball_vy = next_state[3]
        return ball_y > 0.85 and ball_vy > 0 and next_ball_vy < 0

    def _check_ball_brick_collision(self, state, next_state, reward) -> bool:
        ball_y = (state[1] + 1) / 2
        return 0.05 < ball_y < 0.3 and reward > 1.0

    def _check_ball_wall_collision(self, state, next_state) -> bool:
        ball_vx = state[2]
        next_ball_vx = next_state[2]
        return abs(ball_vx + next_ball_vx) < 0.01

class GodActPrioritizedReplayBuffer:
    def __init__(self, capacity: int, rules: List[GodActRule]):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.rules = rules
        self.rule_hit_counts = {rule.rule_id: 0 for rule in rules}
        self.total_transitions = 0

    def push(self, state, action, reward, next_state, done):
        priority = 1.0
        matched_rules = []
        for rule in self.rules:
            try:
                if rule.matches_state_transition(state, next_state, reward):
                    priority *= rule.priority
                    matched_rules.append(rule.rule_id)
                    self.rule_hit_counts[rule.rule_id] += 1
            except Exception:
                continue

        if len(matched_rules) > 0:
            rarity_bonus = 1.0 + (1.0 / (1.0 + self.rule_hit_counts[matched_rules[0]]))
            priority *= rarity_bonus

        if abs(reward) > 5:
            priority *= 2.0

        self.buffer.append((state, action, reward, next_state, done, matched_rules))
        self.priorities.append(priority)
        self.total_transitions += 1

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return []
        batch_size = min(batch_size, len(self.buffer))
        priorities = np.array(self.priorities, dtype=np.float64)
        if priorities.sum() == 0:
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, rules = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)

    def get_statistics(self):
        return {k: v / max(1, self.total_transitions) for k, v in self.rule_hit_counts.items()}

class GodActRewardShaper:
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules
        self.shaping_functions = {
            'ball_paddle_collision': self._shape_paddle_approach,
            'ball_brick_collision': self._shape_brick_targeting,
            'ball_lost': self._shape_ball_preservation,
        }

    def shape_reward(self, state, action, reward, next_state, env_info=None) -> float:
        shaped_reward = float(reward)
        for rule in self.rules:
            if rule.trigger in self.shaping_functions:
                try:
                    shaped_reward += self.shaping_functions[rule.trigger](state, action, next_state) * rule.confidence
                except Exception:
                    pass
        return shaped_reward

    def _shape_paddle_approach(self, state, action, next_state) -> float:
        ball_x = (state[0] + 1) / 2
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        paddle_x = (state[4] + 1) / 2
        next_paddle_x = (next_state[4] + 1) / 2
        if ball_vy > 0 and ball_y > 0.5:
            prev_dist = abs(paddle_x - ball_x)
            next_dist = abs(next_paddle_x - ball_x)
            if next_dist < prev_dist:
                return 0.5 * (1.0 - ball_y)
        return 0.0

    def _shape_brick_targeting(self, state, action, next_state) -> float:
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        if ball_vy < 0 and 0.3 < ball_y < 0.5:
            return 0.2
        return 0.0

    def _shape_ball_preservation(self, state, action, next_state) -> float:
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        if ball_vy > 0 and ball_y > 0.8:
            return -0.3 * (ball_y - 0.8)
        return 0.0

class GodActCurriculumGenerator:
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules
        self.curriculum_stages = self._generate_stages()
        self.current_stage = 0

    def _generate_stages(self):
        stages = []
        if any(r.trigger == 'ball_paddle_collision' for r in self.rules):
            stages.append({
                'name': 'paddle_collision_training',
                'init_func': self._init_paddle_collision,
                'episodes': 50
            })
        if any(r.trigger == 'ball_brick_collision' for r in self.rules):
            stages.append({
                'name': 'brick_collision_training',
                'init_func': self._init_brick_collision,
                'episodes': 100
            })
        stages.append({
            'name': 'full_game',
            'init_func': None,
            'episodes': 500
        })
        return stages

    def get_current_stage(self):
        return self.curriculum_stages[self.current_stage]

    def advance_stage(self):
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            print(f"[Curriculum] Avanzato a stage: {self.get_current_stage()['name']}")

    def _init_paddle_collision(self):
        return {
            'ball_x': 0.5 + (random.random() - 0.5) * 0.3,
            'ball_y': 0.85,
            'ball_vx': (random.random() - 0.5) * 0.04,
            'ball_vy': 0.03,
            'paddle_x': 0.5 + (random.random() - 0.5) * 0.2
        }

    def _init_brick_collision(self):
        return {
            'ball_x': 0.5 + (random.random() - 0.5) * 0.4,
            'ball_y': 0.35,
            'ball_vx': (random.random() - 0.5) * 0.04,
            'ball_vy': -0.03,
            'paddle_x': 0.5
        }

class GodActEnvWrapper(gym.Wrapper):
    def __init__(self, env, rules: List[GodActRule]):
        super().__init__(env)
        self.reward_shaper = GodActRewardShaper(rules)
        self.curriculum = GodActCurriculumGenerator(rules)
        self.episode_count = 0

    def reset(self, **kwargs):
        stage = self.curriculum.get_current_stage()
        if stage['init_func'] is not None:
            init_params = stage['init_func']()
            if hasattr(self.env, 'reset_with_params'):
                return self.env.reset_with_params(init_params)
        return self.env.reset(**kwargs)

    def step(self, action):
        state = self._get_current_state()
        next_state, reward, done, info = self.env.step(action)
        shaped_reward = self.reward_shaper.shape_reward(state, action, reward, next_state)
        if done:
            self.episode_count += 1
            stage = self.curriculum.get_current_stage()
            if self.episode_count >= stage['episodes']:
                self.curriculum.advance_stage()
                self.episode_count = 0
        return next_state, shaped_reward, done, info

    def _get_current_state(self):
        if hasattr(self.env, 'get_observation'):
            return self.env.get_observation()
        return self.env._get_obs() if hasattr(self.env, '_get_obs') else None

# ---------------------------
# DQN (PyTorch)
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Integrator (carica regole)
# ---------------------------
class GodActDQNIntegrator:
    def __init__(self, rules_json_path: str):
        self.rules = self._load_rules(rules_json_path)
        print(f"[GodActDQN] Caricate {len(self.rules)} regole")
        self._print_rules_summary()

    def _load_rules(self, json_path: str) -> List[GodActRule]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        rules = [GodActRule(r) for r in data.get('rules', [])]
        return rules

    def _print_rules_summary(self):
        print("\n=== Regole Scoperte ===")
        for rule in self.rules:
            print(f"  [{rule.rule_id}] {rule.trigger} → {rule.effect}")
            print(f"    Confidence: {rule.confidence:.2f}, Priority: {rule.priority:.2f}")
        print("=" * 50 + "\n")

    def wrap_environment(self, env):
        return GodActEnvWrapper(env, self.rules)

    def create_replay_buffer(self, capacity: int):
        return GodActPrioritizedReplayBuffer(capacity, self.rules)

    def get_curriculum_generator(self):
        return GodActCurriculumGenerator(self.rules)

# ---------------------------
# Training loop
# ---------------------------
def train(seed=0, total_episodes=400, max_steps_per_episode=2000, render_interval=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Example rules file (creates if not existing)
    rules_path = RULES_PATH
    if not os.path.exists(rules_path):
        # example_rules = {
        #     "rules": [
        #         {"rule_id": "R1", "trigger": "ball_paddle_collision", "effect": "velocity_invert_y", "confidence": 0.95, "priority": 3.0, "reward_modifier": 1.5},
        #         {"rule_id": "R2", "trigger": "ball_brick_collision", "effect": "brick_disappear_and_velocity_invert", "confidence": 0.90, "priority": 2.0, "reward_modifier": 2.0},
        #         {"rule_id": "R3", "trigger": "ball_lost", "effect": "game_over", "confidence": 1.0, "priority": 2.5, "reward_modifier": 1.0}
        #     ]
        # }
        # with open(rules_path, 'w') as f:
        #     json.dump(example_rules, f, indent=2)
         print(f"⚙️  File '{rules_path}'doesn't exist !!")
         break

    integrator = GodActDQNIntegrator(rules_path)

    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(state_dim, action_dim).to(device)
    q_target = QNetwork(state_dim, action_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay_buffer = integrator.create_replay_buffer(capacity=50000)

    gamma = 0.99
    batch_size = 64
    learning_starts = 1000
    train_freq = 4
    polyak = 1.0  # we'll do hard update
    update_target_every = 1000

    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay_episodes = 250

    total_steps = 0
    losses = []

    for ep in range(1, total_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        # epsilon schedule (linear)
        eps = max(epsilon_final, epsilon_start - (epsilon_start - epsilon_final) * (ep / epsilon_decay_episodes))

        while not done and step < max_steps_per_episode:
            step += 1
            total_steps += 1
            s_t = np.array(state, dtype=np.float32)

            if random.random() < eps or len(replay_buffer) < learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t = torch.from_numpy(s_t).unsqueeze(0).to(device)
                    qvals = q_net(t)
                    action = int(torch.argmax(qvals, dim=1).cpu().numpy()[0])

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # training step
            if (total_steps % train_freq == 0) and (len(replay_buffer) >= batch_size):
                batch = replay_buffer.sample(batch_size)
                if len(batch) == 0:
                    continue
                states_b, actions_b, rewards_b, next_states_b, dones_b = batch
                states_t = torch.from_numpy(states_b).to(device)
                actions_t = torch.from_numpy(actions_b).long().to(device)
                rewards_t = torch.from_numpy(rewards_b).to(device)
                next_states_t = torch.from_numpy(next_states_b).to(device)
                dones_t = torch.from_numpy(dones_b.astype(np.uint8)).to(device)

                # compute targets
                with torch.no_grad():
                    next_q = q_target(next_states_t)
                    max_next_q, _ = torch.max(next_q, dim=1)
                    td_target = rewards_t + gamma * (1.0 - dones_t.float()) * max_next_q

                q_values = q_net(states_t)
                q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.mse_loss(q_selected, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # update target
            if total_steps % update_target_every == 0:
                q_target.load_state_dict(q_net.state_dict())

            # render occasionally
            if render_interval > 0 and (ep % render_interval == 0):
                env.render()

        print(f"[Ep {ep}/{total_episodes}] Reward: {ep_reward:.2f} Steps: {step} Eps: {eps:.3f} Buffer: {len(replay_buffer)} Loss(avg): {np.mean(losses[-100:]) if losses else 0:.4f}")

        # small checkpoint
        if ep % 50 == 0:
            torch.save(q_net.state_dict(), f"dqn_checkpoint_ep{ep}.pth")

    # save final model
    torch.save(q_net.state_dict(), "dqn_arkanoid_godacts_final.pth")
    print("Training completo. Modello salvato come dqn_arkanoid_godacts_final.pth")
    env.close()

if __name__ == "__main__":
    train(seed=1, total_episodes=400, max_steps_per_episode=2000, render_interval=25)
