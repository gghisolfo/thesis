import time
import torch
import numpy as np

from .generic_multi_game import QNetwork
from .game_env import ArkanoidEnv


# =====================
# CONFIG
# =====================

MODEL_LIST = [
    "./dqn/dqn_models/WINNING_MODEL.pth",
    "./dqn/dqn_models/arkanoid_no_shaping_no_density/best_reward_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_no_shaping_no_density/best_time_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_no_shaping_no_density/last_generic_arkanoid.pth",

    "./dqn/dqn_models/arkanoid_yes_shaping_no_density/best_reward_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_yes_shaping_no_density/best_time_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_yes_shaping_no_density/last_generic_arkanoid.pth",

    "./dqn/dqn_models/arkanoid_no_shaping_yes_density/best_reward_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_no_shaping_yes_density/best_time_arkanoid.pth",
    "./dqn/dqn_models/arkanoid_no_shaping_yes_density/last_generic_arkanoid.pth",
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
        steps = 0  # frames

        while not done:

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_values = q_net(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            obs, done, _ = env.step(action)
            steps += 1

            if time.time() - start_time > TIME_LIMIT:
                break

        elapsed = time.time() - start_time
        time_left = max(0, TIME_LIMIT - elapsed)

        bricks_end = env.game.bricks_alive
        bricks_destroyed = bricks_start - bricks_end
        finished = (env.game.bricks_alive == 0)

        model_results.append({
            "episode": ep,
            "bricks_destroyed": bricks_destroyed,
            "time_left": time_left,
            "finished": finished,
            "steps": steps
        })

        print(f"[{model_path.split('/')[-1]}] "
              f"EP {ep:03d} | bricks: {bricks_destroyed:02d} "
              f"| steps: {steps} | win: {finished}")

    # =====================
    # STATISTICHE MODELLO
    # =====================

    brick_values = [r["bricks_destroyed"] for r in model_results]
    time_values = [r["time_left"] for r in model_results]
    step_values = [r["steps"] for r in model_results]
    wins = sum(1 for r in model_results if r["finished"])

    # steps SOLO per episodi vinti
    steps_to_win = [r["steps"] for r in model_results if r["finished"]]

    summary = {
        "avg_bricks": np.mean(brick_values),
        "max_bricks": np.max(brick_values),
        "min_bricks": np.min(brick_values),

        "avg_time_left": np.mean(time_values),

        "avg_steps": np.mean(step_values),
        "min_steps": np.min(step_values),
        "max_steps": np.max(step_values),

        "avg_steps_to_win": np.mean(steps_to_win) if len(steps_to_win) > 0 else None,

        "wins": wins,
        "win_rate": wins / N_EPISODES
    }

    all_results[model_path] = summary


print("\n\n==============================")
print("FINAL COMPARISON")
print("==============================")

for model, stats in all_results.items():

    print(f"\nMODEL: {model}")
    print(f" Avg bricks: {stats['avg_bricks']:.2f}")
    print(f" Wins: {stats['wins']}/{N_EPISODES} "
          f"({stats['win_rate']*100:.1f}%)")

    print(f" Avg steps: {stats['avg_steps']:.1f}")
    print(f" Min steps: {stats['min_steps']}")
    print(f" Max steps: {stats['max_steps']}")

    if stats["avg_steps_to_win"] is not None:
        print(f" Avg steps (only wins): {stats['avg_steps_to_win']:.1f}")
    else:
        print(" Avg steps (only wins): N/A")