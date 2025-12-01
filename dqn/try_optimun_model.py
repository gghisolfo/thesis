import torch
import numpy as np
import time
import gymnasium as gym
from .train_dqn_from_population import ArkanoidEnv, QNetwork
# python -m dqn.try_optimun_model 

# Ricrea la rete QNetwork (uguale a quella usata in training)
import torch.nn as nn

# Crea ambiente
env = ArkanoidEnv()

# Carica modello PyTorch
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load("./dqn/dqn_models/dqn_generic_symbolic.pth", map_location="cpu"))
model.eval()

# Esegui un episodio
obs = env.reset()
done = False
total_reward = 0

while not done:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_t)
        action = int(torch.argmax(q_values, dim=1).item())

    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.05)

print(f"🏁 Reward totale: {total_reward:.2f}")
env.close()
