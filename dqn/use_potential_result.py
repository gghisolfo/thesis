import pygame
import torch
from .train_dqn_with_godAct import ArkanoidEnv, QNetwork
from arkanoid_game import Game, grid_width, grid_height, screen_width, screen_height


MODEL_PATH = "./dqn/dqn_models/dqn_with_godAct.pth" # dqn_generic_symbolic | generic_0



# Setup Pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
FRAME_RATE = 60

env = ArkanoidEnv()
obs = env.reset()
done = False

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
q_net.eval()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())

    obs, reward, done, _ = env.step(action)

    # Rendering manuale
    grid_surface = pygame.surfarray.make_surface(env.game.get_grid())
    scaled_surface = pygame.transform.scale(grid_surface, (screen_width, screen_height))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    clock.tick(FRAME_RATE)

pygame.quit()
