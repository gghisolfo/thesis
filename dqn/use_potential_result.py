import pygame
import torch

from .generic_4_multi_game import QNetwork

from .game_env import CatchEnv, ArkanoidEnv

from arkanoid_game import screen_width, screen_height

MODEL_PATH = "./dqn/dqn_models/generic_multi.pth" # WINNING_MODEL | generic_multi | generic_4 | generic_4_no_shaping_no_density  

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
GAME_TYPE = "catch"  # "arkanoid" o "catch"

if GAME_TYPE == "arkanoid":
    env = ArkanoidEnv()
    
elif GAME_TYPE == "catch":
    env = CatchEnv()
    
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

    # Rendering
    grid_surface = pygame.surfarray.make_surface(env.game.get_grid())
    scaled_surface = pygame.transform.scale(grid_surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FRAME_RATE)

pygame.quit()