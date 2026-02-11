import time
import torch
import numpy as np

from .generic_4_multi_game import QNetwork
from .game_env import ArkanoidEnv


# =====================
# CONFIG
# =====================

MODEL_LIST = [
    "./dqn/dqn_models/WINNING_MODEL.pth",
    "./dqn/dqn_models/generic_4_no_shaping_no_density.pth",
    "./dqn/dqn_models/generic_4_no_shaping_yes_density.pth",
    "./dqn/dqn_models/best_time_pong.pth",
    "./dqn/dqn_models/best_reward_pong.pth",


]



N_EPISODES = 100
TIME_LIMIT = 60  # secondi

# =====================
# SETUP
# =====================

env = ArkanoidEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

def load_model(path):
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# =====================
# EVALUATION
# =====================

all_results = {}

for model_path in MODEL_LIST:

    print(f"\n==============================")
    print(f"TESTING MODEL: {model_path}")
    print(f"==============================")

    q_net = load_model(model_path)

    model_results = []

    for ep in range(N_EPISODES):

        obs = env.reset()
        done = False

        start_time = time.time()
        bricks_start = env.game.bricks_alive

        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            obs, reward, done, _ = env.step(action)

            # STOP dopo 60 secondi
            if time.time() - start_time > TIME_LIMIT:
                break

        elapsed = time.time() - start_time
        time_left = max(0, TIME_LIMIT - elapsed)

        bricks_end = env.game.bricks_alive
        bricks_destroyed = bricks_start - bricks_end

        model_results.append({
            "episode": ep,
            "bricks_destroyed": bricks_destroyed,
            "time_left": time_left,
            "finished": (env.game.bricks_alive == 0)  # SOLO se ha vinto davvero
        })

        print(f"[{model_path.split('/')[-1]}] "
              f"EP {ep:03d} | bricks: {bricks_destroyed:02d} | time_left: {time_left:.2f}s")

    # =====================
    # STATISTICHE MODELLO
    # =====================

    brick_values = [r["bricks_destroyed"] for r in model_results]
    time_values = [r["time_left"] for r in model_results]
    wins = sum(1 for r in model_results if r["finished"])

    summary = {
        "avg_bricks": np.mean(brick_values),
        "max_bricks": np.max(brick_values),
        "min_bricks": np.min(brick_values),
        "avg_time_left": np.mean(time_values),
        "max_time_left": np.max(time_values),
        "min_time_left": np.min(time_values),
        "wins": wins
    }

    all_results[model_path] = summary

# =====================
# COMPARISON
# =====================

print("\n\n==============================")
print("FINAL COMPARISON")
print("==============================")

for model, stats in all_results.items():

    print(f"\nMODEL: {model}")
    print(f" Avg bricks: {stats['avg_bricks']:.2f}")
    print(f" Max bricks: {stats['max_bricks']}")
    print(f" Min bricks: {stats['min_bricks']}")
    print(f" Avg time left: {stats['avg_time_left']:.2f}s")
    print(f" Wins: {stats['wins']}/{N_EPISODES}")
