import sys
import time
import pygame
import random
import numpy as np
from copy import deepcopy
from datetime import datetime
import pickle

# ---------------- CONFIG ---------------- #

refresh_rate = 0.05
grid_width_C, grid_height_C = 121, 71
screen_width, screen_height = grid_width_C * 10, grid_height_C * 10
MAX_MISS = 1

# IDs
# 0 environment
# 1 falling_object
# 3 agent (paddle)
# 7 wall_top
# 8 wall_bottom
# 5 wall_left
# 6 wall_right


# ================= GAME ================= #

class CatchGame:

    def __init__(self):

        self.elements = {}
        self.event_log = []
        self.event_pending = []

        self.init_grid()
        self.init_environment()
        self.init_walls()
        self.init_agent()
        self.init_object()

        self.caught = 0
        self.missed = 0

    # ---------- GRID ---------- #

    def init_grid(self):
        self.grid = np.zeros((grid_width_C, grid_height_C), dtype=int)
        self.r = np.zeros((grid_width_C, grid_height_C), dtype=int)
        self.g = np.zeros((grid_width_C, grid_height_C), dtype=int)
        self.b = np.zeros((grid_width_C, grid_height_C), dtype=int)

    # ---------- ENV ---------- #

    def init_environment(self):
        self.elements['environment'] = {
            'id': 0,
            'existence': False
        }

    # ---------- WALLS ---------- #

    def init_walls(self):

        # left
        self.grid[0:3, :] = 5
        self.g[0:3, :] = 255

        self.elements['wall_left'] = {'id': 5, 'existence': True}

        # right
        self.grid[grid_width_C-3:grid_width_C, :] = 6
        self.g[grid_width_C-3:grid_width_C, :] = 255

        self.elements['wall_right'] = {'id': 6, 'existence': True}

        # top
        self.grid[:, 0:3] = 7
        self.g[:, 0:3] = 255

        self.elements['wall_top'] = {'id': 7, 'existence': True}

        # bottom
        self.grid[:, grid_height_C-3:grid_height_C] = 8
        self.r[:, grid_height_C-3:grid_height_C] = 255

        self.elements['wall_bottom'] = {'id': 8, 'existence': True}

    # ---------- AGENT ---------- #

    def init_agent(self):

        self.agent_x = 60
        self.agent_y = 65
        self.agent_halfwidth = 5
        self.agent_halfheight = 1
        self.agent_speed = 0
        self.agent_base_speed = 2

        self.draw_agent()

        self.elements['agent'] = {
            'id': 3,
            'pos_x': self.agent_x,
            'pos_y': self.agent_y,
            'existence': True
        }

    def set_paddle_speed(self, v):
        self.agent_speed = v * self.agent_base_speed

    def update_agent(self):
        if self.missed >= MAX_MISS:
            return self.elements, True

        old_x = self.agent_x
        self.agent_x = np.clip(
            self.agent_x + self.agent_speed,
            self.agent_halfwidth + 3,
            grid_width_C - self.agent_halfwidth - 4
        )

        # clear old
        self.b[
            old_x - self.agent_halfwidth:old_x + self.agent_halfwidth + 1,
            self.agent_y - self.agent_halfheight:self.agent_y + self.agent_halfheight + 1
        ] = 0

        self.draw_agent()

        self.elements['agent']['pos_x'] = self.agent_x

    def draw_agent(self):

        self.grid[
            self.agent_x - self.agent_halfwidth:self.agent_x + self.agent_halfwidth + 1,
            self.agent_y - self.agent_halfheight:self.agent_y + self.agent_halfheight + 1
        ] = 3

        self.b[
            self.agent_x - self.agent_halfwidth:self.agent_x + self.agent_halfwidth + 1,
            self.agent_y - self.agent_halfheight:self.agent_y + self.agent_halfheight + 1
        ] = 255

    # ---------- OBJECT ---------- #

    def init_object(self):
        self.obj_x = random.randint(5, 115)
        self.obj_y = 3
        self.obj_radius = 1
        self.obj_speed = random.uniform(0.5, 1.5)

        self.draw_object()

        self.elements['object'] = {
            'id': 1,
            'pos_x': self.obj_x,
            'pos_y': self.obj_y,
            'existence': True
        }

    def respawn_object(self):

        self.r[
            self.obj_x - 1:self.obj_x + 2,
            int(self.obj_y) - 1:int(self.obj_y) + 2
        ] = 0

        self.obj_x = random.randint(5, 115)
        self.obj_y = 3
        self.obj_speed = random.uniform(0.5, 1.5)

    def update_object(self):

        old_y = int(self.obj_y)
        self.obj_y += self.obj_speed

        # clear old
        self.r[
            self.obj_x - 1:self.obj_x + 2,
            old_y - 1:old_y + 2
        ] = 0

        # catch
        if abs(self.obj_y - self.agent_y) < 2:
            if abs(self.obj_x - self.agent_x) < self.agent_halfwidth:
                self.caught += 1
                self.event_log.append({'event': 'caught'})
                self.respawn_object()
                return

        # miss
        if self.obj_y > grid_height_C - 3:
            self.missed += 1
            self.event_log.append({'event': 'missed'})
            self.respawn_object()
            return

        self.draw_object()

        self.elements['object']['pos_y'] = self.obj_y

    def draw_object(self):

        y = int(self.obj_y)
        self.grid[self.obj_x - 1:self.obj_x + 2, y - 1:y + 2] = 1
        self.r[self.obj_x - 1:self.obj_x + 2, y - 1:y + 2] = 255

    # ---------- UPDATE ---------- #

    def update(self):

        self.update_agent()
        self.update_object()

        log = self.event_log
        # print(log)
        self.event_log = []

        return self.elements, log, False  # mai done

    def get_grid(self):
        return np.transpose(np.stack([self.r, self.g, self.b]), (1, 2, 0))


# ================= PYGAME LOOP ================= #

# pygame.init()
# window = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Catch Game")

# game = CatchGame()
# clock = pygame.time.Clock()

# leftHeld = False
# rightHeld = False
# running = True

# while running:
#     clock.tick(60)

#     for e in pygame.event.get():
#         if e.type == pygame.QUIT:
#             running = False

#         if e.type == pygame.KEYDOWN:
#             if e.key == pygame.K_LEFT:
#                 leftHeld = True
#             if e.key == pygame.K_RIGHT:
#                 rightHeld = True

#         if e.type == pygame.KEYUP:
#             if e.key == pygame.K_LEFT:
#                 leftHeld = False
#             if e.key == pygame.K_RIGHT:
#                 rightHeld = False

#     if leftHeld and not rightHeld:
#         game.set_paddle_speed(-1)
#     elif rightHeld and not leftHeld:
#         game.set_paddle_speed(1)
#     else:
#         game.set_paddle_speed(0)

#     _, _, done = game.update()
#     if done:
#         running = False

#     surface = pygame.surfarray.make_surface(game.get_grid())
#     surface = pygame.transform.scale(surface, (screen_width, screen_height))
#     window.blit(surface, (0, 0))

#     pygame.display.flip()

# pygame.quit()
# sys.exit()