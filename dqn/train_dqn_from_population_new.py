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
os.makedirs(SAVE_DIR, exist_ok=True)

FRAME_RATE = 60

class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()
        # self.game.ball_lost = False
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

        # Stato precedente per il reward shaping
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y

    def reset(self):
        self.game = Game()
        # self.game.ball_lost = False
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y
        self.done = False
        return self._get_obs()

    def step(self, action):
        # Salva lo stato precedente
        prev_bricks = self.game.bricks_alive
        prev_ball_y = self.game.ball_y

        # Esegui azione
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        # Aggiorna il gioco
        self.game.update()

        # Calcola reward PRIMA di controllare la terminazione
        reward = self._compute_reward(prev_bricks, prev_ball_y)

        # Controlla condizioni di terminazione
        # 1. Tutti i brick distrutti (VITTORIA)
        if self.game.bricks_alive == 0:
            self.done = True
            reward += 100.0
            print("🎉 VITTORIA! Tutti i brick distrutti! (+100.0)")

        # 2. Palla persa (SCONFITTA)
        if self.game.ball_lost:
            self.done = True
            reward -= 50.0
            # print("💀 GAME OVER! Palla persa! (-50.0)")

        # 3. Palla troppo vicina al bordo inferiore (backup safety check)
        if self.game.ball_y + self.game.ball_radius >= grid_height - 3:
            self.done = True
            reward -= 50.0
            # print("💀 GAME OVER! Palla fuori campo! (-50.0)")

        return self._get_obs(), reward, self.done, {}


    def _get_obs(self):
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)

    def _compute_reward(self, prev_bricks, prev_ball_y):
        r = 0.0  # reward base per ogni step
        
        # 🧱 Reward per brick distrutti
        if self.game.bricks_alive < prev_bricks:
            destroyed = prev_bricks - self.game.bricks_alive
            r += 10.0 * destroyed
            # print(f"🧱 {destroyed} brick distrutti! (+{10.0 * destroyed})")

        # 🏓 Reward per colpo sulla paddle
        if self._check_ball_hits_paddle(prev_ball_y):
            r += 2.0
            # print("✅ Palla colpita dalla paddle! (+2.0)")

        # 📍 Piccolo bonus per mantenere la paddle vicina alla palla (in orizzontale)
        distance_to_paddle = abs(self.game.ball_x - self.game.paddle_x)
        if distance_to_paddle < 10:
            r += 0.1
        
        # ⚠️ Piccola penalità se la palla è molto vicina al fondo
        # (incentiva l'agente a mantenere la palla alta)
        if self.game.ball_y > grid_height - 15:
            r -= 0.2
        
        return r

    def _check_ball_hits_paddle(self, prev_ball_y):
        """
        Rileva se la pallina ha colpito la paddle durante questo step.
        """
        ball = self.game.elements['ball']
        paddle = self.game.elements['paddle_center']

        # Se la palla sta andando verso l'alto dopo essere stata in basso, 
        # probabilmente ha appena colpito la paddle
        if self.game.ball_speed_y < 0 and prev_ball_y > self.game.ball_y:
            # Controlla se la posizione attuale è vicina alla paddle
            if abs(self.game.ball_y - paddle['pos_y']) < 3:
                # Controlla sovrapposizione orizzontale
                overlap_x = (
                    (ball['hitbox_br_x'] >= paddle['hitbox_tl_x']) and
                    (ball['hitbox_tl_x'] <= paddle['hitbox_br_x'])
                )
                if overlap_x:
                    return True
        
        return False

    def render(self, mode="human"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        grid_surface = pygame.surfarray.make_surface(self.game.get_grid())
        scaled_surface = pygame.transform.scale(
            grid_surface, (self.screen_width, self.screen_height)
        )
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(FRAME_RATE)

    def close(self):
        pygame.quit()


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
    total_episodes=1000,
    max_steps_per_episode=2000
):
    # Carica popolazione euristica
    with open(population_path, "rb") as f:
        population = pickle.load(f)

    integrator = GodActDQNIntegrator(rules_dict=population)
    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    # Setup RL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    buffer = integrator.create_replay_buffer(50000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.97 #0.995

    for ep in range(total_episodes): #episodi
        state = env.reset() 
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode: #steps
            # 1️⃣ Scegli un'azione (esplorazione vs. sfruttamento)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_net(s_t)
                    action = int(q_vals.argmax(1).item())

            # 2️⃣ Esegui l’azione nell’ambiente
            # print("episodio", ep, steps)
            next_state, reward, done, _ = env.step(action)

            # 3️⃣ Salva l’esperienza nel buffer
            buffer.push(state, action, reward, next_state, done)

            # 5️⃣ Passa allo stato successivo
            total_reward += reward
            state = next_state
            steps += 1
            
            # Training
            # 4️⃣ Aggiorna la rete (training vero e proprio)
            if len(buffer.buffer) >= 64:
                s, a, r, ns, d = buffer.sample(64)
                s_t = torch.tensor(s, device=device)
                a_t = torch.tensor(a, device=device).unsqueeze(1)
                r_t = torch.tensor(r, device=device)
                ns_t = torch.tensor(ns, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                # quanto la rete crede sia buono fare l'azione a nello stato s
                # Q-target
                with torch.no_grad():
                    max_next_q = q_target(ns_t).max(1)[0]
                    target_q = r_t + gamma * (1 - d_t) * max_next_q

                # Q-current
                current_q = q_net(s_t).gather(1, a_t).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)

                # Backpropagation: calcolo perdita e aggiornamento pesi
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 5️⃣ Passa allo stato successivo
            total_reward += reward
            state = next_state
            steps += 1




        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Ep {ep} - Reward: {total_reward:.2f} - Steps: {steps} - Eps: {epsilon:.3f}")

        if ep % 10 == 0:
            q_target.load_state_dict(q_net.state_dict())# q_net si allena ad ogni step, q_target ogni 10 episodi viene aggiornata
            model_path = os.path.join(SAVE_DIR, f"dqn_from_population_ep{ep}.pth")
            torch.save(q_net.state_dict(), model_path)

    env.close()
    final_path = os.path.join(SAVE_DIR, "dqn_from_population_final.pth")
    torch.save(q_net.state_dict(), final_path)
    print("✅ Training completo, modello salvato.")


if __name__ == "__main__":
    train_dqn_from_population()