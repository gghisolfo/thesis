import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from train_godacts_dqn import ArkanoidEnv
from components.traduzione.godAct.god_act_core import GodActDQNIntegrator

BEST_POPULATION_PATH= "../../best_population.pkl"

# modello DQN semplice
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.net(x)

def train_dqn_from_population(
    population_path=BEST_POPULATION_PATH,
    total_episodes=400,
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
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995

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

            # Reward shaping dal GodActs integrator
            shaped_reward = env.reward_shaper.shape_reward(state, action, reward, next_state)

            buffer.push(state, action, shaped_reward, next_state, done)
            state = next_state
            total_reward += shaped_reward

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

        if ep % 50 == 0:
            q_target.load_state_dict(q_net.state_dict())
            torch.save(q_net.state_dict(), f"dqn_from_population_ep{ep}.pth")

    env.close()
    torch.save(q_net.state_dict(), "dqn_from_population_final.pth")
    print("✅ Training completo, modello salvato.")

if __name__ == "__main__":
    train_dqn_from_population()
