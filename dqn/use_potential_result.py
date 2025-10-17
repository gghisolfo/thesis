
#run at thesis level with python -m dqn.use_potential_result
import torch
import numpy as np
from .train_dqn_from_population import ArkanoidEnv, QNetwork

# Ricrea la rete QNetwork (uguale a quella usata in training)
import torch.nn as nn

# Path del modello PT
MODEL_PATH = "./dqn/dqn_models/dqn_from_population_final.pth" #"./dqn/dqn_model/dqn_from_population_final.pth"
# MODEL_PATH = "./dqn/dqn_models/dqn_from_population_ep980.pth"
# Crea ambiente
env = ArkanoidEnv()
obs = env.reset()
done = False


# Inizializza la rete e carica i pesi
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
q_net.eval()  # modalità valutazione

# Gioca la partita
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
    
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
