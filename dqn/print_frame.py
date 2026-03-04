import os
import time
import pygame
import torch
import numpy as np

from .generic_multi_game import QNetwork
from .game_env import ArkanoidEnv, PongEnv
from arkanoid_game import screen_width, screen_height
from pong_game import LIVES

MODEL_PATH = "./dqn/dqn_models/arkanoid_no_shaping_yes_density/best_time_arkanoid.pth"

# "./dqn/dqn_models/arkanoid_no_shaping_no_density/last_generic_arkanoid.pth",  #99%
#    "./dqn/dqn_models/arkanoid_no_shaping_no_density/best_time_arkanoid.pth", # 81%
#   "./dqn/dqn_models/arkanoid_no_shaping_yes_density/best_time_arkanoid.pth", # 100%

# WINNING_MODEL | generic_multi | generic_4 | generic_4_no_shaping_no_density | best_time_pong

SCREENSHOTS_DIR = "./dqn/PRINT"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

TIME_LIMIT = 20  # secondi

# -----------------------------
# Setup Pygame
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
FRAME_RATE = 60

# -----------------------------
# Scegli il tipo di gioco
# -----------------------------
GAME_TYPE = "arkanoid"  # "arkanoid" o "pong"

if GAME_TYPE == "arkanoid":
    env = ArkanoidEnv()
elif GAME_TYPE == "pong":
    env = PongEnv(lives=LIVES)

obs = env.reset()
done = False

# -----------------------------
# Carica Q-Network
# -----------------------------
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
q_net.eval()

# -----------------------------
# Screenshot setup
# -----------------------------
model_name = MODEL_PATH.split("/")[-1].replace(".pth", "")

# I 4 screenshot avvengono a questi secondi dall'inizio:
#   start  → t = 0
#   mid_1  → t = TIME_LIMIT/2
#   mid_2  → t = TIME_LIMIT/2 + 1
#   mid_3  → t = TIME_LIMIT/2 + 2
t_mid = TIME_LIMIT / 2
checkpoints = {
    "start":  0,
    "mid_1":  t_mid,
    "mid_2":  t_mid + 1,
    "mid_3":  t_mid + 2,
}
taken = {label: False for label in checkpoints}

start_time = time.time()
steps = 0


def save_screenshot(label):
    path = os.path.join(SCREENSHOTS_DIR, f"{model_name}_{label}.png")
    pygame.image.save(screen, path)
    print(f"[screenshot] salvato: {path}")


# -----------------------------
# Loop principale
# -----------------------------
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

    obs, done, _ = env.step(action)

    # Rendering
    grid_surface = pygame.surfarray.make_surface(env.game.get_grid())
    scaled_surface = pygame.transform.scale(grid_surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FRAME_RATE)

    # -----------------------------
    # Screenshot checkpoints
    # -----------------------------
    elapsed = time.time() - start_time

    for label, t in checkpoints.items():
        if not taken[label] and elapsed >= t:
            save_screenshot(label)
            taken[label] = True

    steps += 1

pygame.quit()