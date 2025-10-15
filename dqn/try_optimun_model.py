from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import time
from dqn_arkanoid_pygame import ArkanoidEnv

# Crea ambiente
env = ArkanoidEnv()

# Carica modello salvato
model = DQN.load("dqn_arkanoid_model", env=env)

# Valutazione quantitativa
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Reward medio: {mean_reward:.2f} ± {std_reward:.2f}")

# Visualizzazione di una partita
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()       # render grafico
    time.sleep(0.05)   # rallenta la partita (slow motion)

print(f"Reward totale in questa partita: {total_reward:.2f}")

env.close()
