import pygame
import torch
import numpy as np
import cv2
import numpy as np
from .generic_multi_game import QNetwork

from .game_env import ArkanoidEnv, PongEnv

from arkanoid_game import screen_width, screen_height

from pong_game import LIVES



MODEL_PATH = ".dqn/dqn_models/pong_yes_shaping_yes_density/best_time_pong.pth" # arkanoid_no_shaping_yes_density/best_time_arkanoid.pth

MODEL_PATH="C:/Users/user/Documents/UNI/TESI/thesis/dqn/dqn_models/pong_no_shaping_yes_density/last_generic_pong.pth"

print
# "./dqn/dqn_models/WINNING_MODEL.pth"
# "./dqn/dqn_models/arkanoid_no_shaping_no_density/last_generic_arkanoid.pth",  #99%
#    "./dqn/dqn_models/arkanoid_no_shaping_no_density/best_time_arkanoid.pth", # 81%
#   "./dqn/dqn_models/arkanoid_no_shaping_yes_density/best_time_arkanoid.pth", # 100%

# WINNING_MODEL | generic_multi | generic_4 | generic_4_no_shaping_no_density | best_time_pong 

# -----------------------------
# Setup Pygame
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
FRAME_RATE = 60


RECORD_VIDEO = True
VIDEO_PATH = "./gameplay_recording.mp4"

# if RECORD_VIDEO:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(VIDEO_PATH, fourcc, FRAME_RATE, (screen_width, screen_height))


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
N = 1  # quante partite registrare

for episode in range(N):
    obs = env.reset()
    done = False

    if RECORD_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"./dqn/RECORDINGS/recording_ep{episode}.mp4", fourcc, FRAME_RATE, (screen_width, screen_height))

    frame_count = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(obs_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        obs, done, _ = env.step(action)

        grid_surface = pygame.surfarray.make_surface(env.game.get_grid())
        scaled_surface = pygame.transform.scale(grid_surface, (screen_width, screen_height))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        # ← parte mancante!
        if RECORD_VIDEO:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frame_count += 1

        clock.tick(FRAME_RATE)

    if RECORD_VIDEO:
        out.release()
        print(f"Episodio {episode} salvato - Frame: {frame_count}")

pygame.quit()