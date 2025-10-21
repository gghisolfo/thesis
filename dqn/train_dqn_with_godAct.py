import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import os
from collections import defaultdict

# Import locali
from arkanoid_game import Game, grid_width, grid_height
from godAct import GodActDQNIntegrator, GodActPopulationIntegrator, GodActPrioritizedReplayBuffer, GodActEnvWrapper

BEST_POPULATION_PATH= "best_population.pkl"
SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Flag per attivare GodAct ===
USE_GODACT = True

# === Ambiente personalizzato Arkanoid ===
class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y

    def reset(self):
        self.game = Game()
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y
        self.done = False
        return self._get_obs()

    def step(self, action):
        prev_bricks = self.game.bricks_alive
        prev_ball_y = self.game.ball_y

        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        self.game.update()
        reward = self._compute_reward(prev_bricks, prev_ball_y)

        if self.game.bricks_alive == 0:
            self.done = True
            reward += 100.0
        if self.game.ball_lost or self.game.ball_y + self.game.ball_radius >= grid_height - 3:
            self.done = True
            reward -= 50.0

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)

    def _compute_reward(self, prev_bricks, prev_ball_y):
        r = 0.0
        if self.game.bricks_alive < prev_bricks:
            r += 10.0 * (prev_bricks - self.game.bricks_alive)
        return r

# === Rete Q ===
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# === Funzione di training con flag ===
def train_dqn(use_godact=True, total_episodes=200, max_steps=2000):
    # Carica popolazione euristica
    with open(BEST_POPULATION_PATH, "rb") as f:
        population = pickle.load(f)

    # Setup ambiente
    env = ArkanoidEnv()
    if use_godact:
        integrator = GodActDQNIntegrator(rules_dict=population)
        env = integrator.wrap_environment(env)
        buffer = integrator.create_replay_buffer(50000)
    else:
        buffer = deque(maxlen=50000)  # Replay buffer standard

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.97

    rewards_history = []

    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_net(s_t)
                    action = int(q_vals.argmax(1).item())

            next_state, reward, done, _ = env.step(action)

            if use_godact:
                buffer.push(state, action, reward, next_state, done)
            else:
                buffer.append((state, action, reward, next_state, done))

            # Training
            if len(buffer) >= 64:
                batch = random.sample(buffer, 64)
                s, a, r, ns, d = zip(*batch)
                s_t = torch.tensor(np.array(s), device=device)
                a_t = torch.tensor(a, device=device).unsqueeze(1)
                r_t = torch.tensor(r, device=device)
                ns_t = torch.tensor(np.array(ns), device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                with torch.no_grad():
                    max_next_q = q_target(ns_t).max(1)[0]
                    target_q = r_t + gamma * (1 - d_t) * max_next_q

                current_q = q_net(s_t).gather(1, a_t).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_reward += reward
            state = next_state
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        if ep % 10 == 0:
            q_target.load_state_dict(q_net.state_dict())
        print(f"[{'GodAct' if use_godact else 'Vanilla'}] Ep {ep} - Reward: {total_reward:.2f}")

    return rewards_history

# === Funzione di confronto ===
def evaluate_godact_vs_vanilla():
    results = defaultdict(list)
    print("=== Training senza GodAct ===")
    results['vanilla'] = train_dqn(use_godact=False, total_episodes=100)
    print("=== Training con GodAct ===")
    results['godact'] = train_dqn(use_godact=True, total_episodes=100)

    avg_vanilla = np.mean(results['vanilla'])
    avg_godact = np.mean(results['godact'])
    print(f"\n✅ Reward medio senza GodAct: {avg_vanilla:.2f}")
    print(f"✅ Reward medio con GodAct: {avg_godact:.2f}")

    return results

if __name__ == "__main__":
    evaluate_godact_vs_vanilla()
