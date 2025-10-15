"""
Versione estesa con render grafico Pygame per ArkanoidEnv + training DQN.
Requisiti: pip install stable-baselines3[extra] gym numpy pygame

File: dqn_arkanoid_pygame.py
- Ambiente Arkanoid semplificato
- Render grafico con Pygame in tempo reale
- Training DQN e possibilità di giocare in manuale
"""
"DQN allena e impara a giocare BASE"

import math
import random
import numpy as np
import gym
from gym import spaces
import pygame

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


if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy

    env = ArkanoidEnv()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=1,
        policy_kwargs={'net_arch': [256, 256]},
    )

    print("Training DQN... (premi Ctrl+C per interrompere)")
    model.learn(total_timesteps=200000)
    model.save("dqn_arkanoid_model")

    # Visualizza una partita renderizzata
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
    env.close()
