import sys
import os

# Otteniamo la cartella root "thesis" e la aggiungiamo al path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# Ora possiamo importare i moduli
from arkanoid_game import Game

from components.traduzione.godAct.god_act_core import GodActDQNIntegrator
import gym


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os, json

RULES_PATH = "god_acts_rules.json"

# ---------------------------
# Arkanoid Gym Environment
# ---------------------------
class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()  # Niente parametri
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

    def reset(self):
        self.game = Game()  # Reset semplice
        self.done = False
        return self._get_obs()

    def step(self, action):
        # Mappiamo azioni sulla paddle
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        self.game.update()
        reward = self._compute_reward()
        if self.game.bricks_alive == 0:
            self.done = True
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)

    def _compute_reward(self):
        r = 0.1
        # Semplice reward shaping basato su eventi
        if hasattr(self.game, "ball_hit_paddle") and self.game.ball_hit_paddle:
            r += 1.0
        if hasattr(self.game, "brick_destroyed") and self.game.brick_destroyed:
            r += 2.0
        if hasattr(self.game, "ball_lost") and self.game.ball_lost:
            r -= 5.0
        return r

    def render(self, mode="human"):
        pass  # pygame render non implementato qui

    def close(self):
        pass


# ---------------------------
# Q-Network (simple DQN)
# ---------------------------
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

# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train(seed=0, total_episodes=400, render_interval=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(RULES_PATH):
        rules = {
            "rules": [
                {"rule_id": "R1", "trigger": "ball_paddle_collision", "effect": "bounce", "confidence": 0.9, "priority": 2.0},
                {"rule_id": "R2", "trigger": "ball_brick_collision", "effect": "destroy_brick", "confidence": 0.8, "priority": 1.5},
                {"rule_id": "R3", "trigger": "ball_lost", "effect": "lose_life", "confidence": 1.0, "priority": 2.5}
            ]
        }
        with open(RULES_PATH, 'w') as f:
            json.dump(rules, f, indent=2)

    integrator = GodActDQNIntegrator(RULES_PATH)
    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = QNetwork(state_dim, action_dim).to(device)
    q_target = QNetwork(state_dim, action_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = integrator.create_replay_buffer(50000)

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995
    total_steps = 0

    for ep in range(1, total_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    qvals = q_net(s)
                    action = int(torch.argmax(qvals).item())

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(replay) > batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                s = torch.tensor(s).to(device)
                a = torch.tensor(a).unsqueeze(1).to(device)
                r = torch.tensor(r).to(device)
                ns = torch.tensor(ns).to(device)
                d = torch.tensor(d).float().to(device)

                with torch.no_grad():
                    target_q = q_target(ns).max(1)[0]
                    td_target = r + gamma * (1 - d) * target_q

                current_q = q_net(s).gather(1, a).squeeze(1)
                loss = nn.functional.mse_loss(current_q, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % 1000 == 0:
                q_target.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"[Ep {ep}] Reward={ep_reward:.2f} ε={epsilon:.3f}")

        if render_interval and ep % render_interval == 0:
            env.render()

    torch.save(q_net.state_dict(), "dqn_arkanoid_godacts_final.pth")
    print("✅ Training completo. Modello salvato.")
    env.close()

if __name__ == "__main__":
    train(seed=1, total_episodes=400, render_interval=25)
