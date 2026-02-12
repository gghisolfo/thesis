import pygame
import torch
import numpy as np

from .generic_multi_game import QNetwork

from .game_env import ArkanoidEnv, PongEnv

from arkanoid_game import screen_width, screen_height

from pong_game import LIVES

MODEL_PATH = "./dqn/dqn_models/last_generic_pong.pth" 

# WINNING_MODEL | generic_multi | generic_4 | generic_4_no_shaping_no_density | best_time_pong 

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
GAME_TYPE = "pong"  # "arkanoid" o "pong"

if GAME_TYPE == "arkanoid":
    env = ArkanoidEnv()

elif GAME_TYPE == "pong":
    env = PongEnv(lives=LIVES)  # puoi cambiare il numero di vite


    
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

    # Rendering con debug
    grid = env.game.get_grid()
    # print(f"Grid shape: {grid.shape}, dtype: {grid.dtype}")  # DEBUG
    
    # Trasponi da (H, W, 3) a (W, H, 3)
    grid_transposed = np.transpose(grid, (1, 0, 2))
    # print(f"Grid transposed shape: {grid_transposed.shape}")  # DEBUG

    # Rendering
    # print(env.game.get_grid())
    grid_surface = pygame.surfarray.make_surface(env.game.get_grid())
    scaled_surface = pygame.transform.scale(grid_surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FRAME_RATE)

pygame.quit()