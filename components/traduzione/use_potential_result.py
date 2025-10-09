from stable_baselines3 import DQN
from dqn_arkanoid_pygame import ArkanoidEnv

# Crea ambiente con render attivo
env = ArkanoidEnv()

# Carica modello già allenato
model = DQN.load("dqn_arkanoid_model", env=env)

obs = env.reset()
done = False

# Mostra la partita in tempo reale
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
