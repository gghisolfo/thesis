
#run at thesis levels with python -m dqn.train_dqn_from_population
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import os
import pygame

from godAct import GodActDQNIntegrator
from arkanoid_game import Game, grid_width, grid_height, screen_width, screen_height

BEST_POPULATION_PATH= "best_population.pkl"
SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)  # crea la cartella se non esiste

FRAME_RATE = 60 #2

class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        # super().__init__()
        # self.game = Game()  # Niente parametri
        # self.action_space = gym.spaces.Discrete(3)
        # self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        # self.done = False
        super().__init__()
        self.game = Game()  # Niente parametri
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

        # --- Inizializza pygame per il rendering ---
        pygame.init()
        self.screen_width = screen_width#400
        self.screen_height = screen_height#600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Arkanoid RL")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.game = Game()  # Reset semplice
        self.ball_hit_paddle = False ## quiii
        self.brick_destroyed = False
        self.ball_lost = False

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
        # Gestione eventi per chiusura finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        # Aggiornamento logico del gioco
        # self.game.update()  quii

        # Ottieni la griglia RGB come superficie Pygame
        grid_surface = pygame.surfarray.make_surface(self.game.get_grid())

        # Ridimensiona la superficie alla dimensione della finestra
        scaled_surface = pygame.transform.scale(
            grid_surface, (self.screen_width, self.screen_height)
        )

        # Disegna la superficie sulla finestra
        self.screen.blit(scaled_surface, (0, 0))

        # Aggiorna lo schermo
        pygame.display.flip()

        # Imposta il frame rate
        self.clock.tick(FRAME_RATE)#60


    def close(self):
        pygame.quit()

# modello DQN semplice
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


def train_dqn_from_population(
    population_path=BEST_POPULATION_PATH,
    total_episodes=1000, #400
    max_steps_per_episode=2000
):
    # carica popolazione euristica
    with open(population_path, "rb") as f:
        population = pickle.load(f)

    # crea integratore GodActs
    # integrator = GodActDQNIntegrator(population)
    integrator = GodActDQNIntegrator(rules_dict=population)


    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    # setup RL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    buffer = integrator.create_replay_buffer(50000)

    gamma = 0.99
    # epsilon = 1.0#0.2#1.0
    # epsilon_min = 0.02
    # epsilon_decay = 0.995

    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.97  # più rapido: dopo ~200 episodi arriva vicino al minimo


    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_net(s_t)
                    action = int(q_vals.argmax(1).item())

            next_state, reward, done, _ = env.step(action)

            # Reward shaping dal GodActs integrator qui-reward
            # shaped_reward = env.reward_shaper.shape_reward(state, action, reward, next_state)
            # buffer.push(state, action, shaped_reward, next_state, done)
            # total_reward += shaped_reward


            # temporaneamente, usa il reward grezzo
            buffer.push(state, action, reward, next_state, done)
            total_reward += reward

            state = next_state
            
            # training
            if len(buffer.buffer) >= 64:
                s, a, r, ns, d = buffer.sample(64)
                s_t = torch.tensor(s, device=device)
                a_t = torch.tensor(a, device=device).unsqueeze(1)
                r_t = torch.tensor(r, device=device)
                ns_t = torch.tensor(ns, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                with torch.no_grad():
                    max_next_q = q_target(ns_t).max(1)[0]
                    target_q = r_t + gamma * (1 - d_t) * max_next_q

                current_q = q_net(s_t).gather(1, a_t).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Ep {ep} - Reward: {total_reward:.2f} - Eps: {epsilon:.3f}")

        if ep % 10 == 0: #ep % 50 == 0
            q_target.load_state_dict(q_net.state_dict())
            model_path = os.path.join(SAVE_DIR, f"dqn_from_population_ep{ep}.pth")
            torch.save(q_net.state_dict(), model_path)
            # torch.save(q_net.state_dict(), f"dqn_from_population_ep{ep}.pth")

    env.close()
    final_path = os.path.join(SAVE_DIR, "dqn_from_population_final.pth")
    torch.save(q_net.state_dict(), final_path)
    # torch.save(q_net.state_dict(), "dqn_from_population_final.pth")
    print("✅ Training completo, modello salvato.")

if __name__ == "__main__":
    train_dqn_from_population()
