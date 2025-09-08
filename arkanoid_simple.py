import sys
import time
import math
import pygame
import pickle
import random
import numpy as np
from datetime import datetime

from copy import deepcopy

refresh_rate = 0.05
grid_width, grid_height = 121, 71
screen_width, screen_height = grid_width * 10, grid_height * 10

# 0 environment
# 1 ball
# 2 paddle_left
# 3 paddle_center
# 4 paddle_right
# 5 wall_left
# 6 wall_right
# 7 wall_top
# 8 wall_bottom
# 9:34 bricks

class Game:

    def __init__(self):

        self.elements = {}
        self.event_log = []

        self.elements['environment'] = {
            'id': 0,
            'pos_x': grid_width // 2,
            'pos_y': grid_height // 2,
            'shape_x': grid_width // 2,
            'shape_y': grid_height // 2,
            'hitbox_tl_x': 0,
            'hitbox_tl_y': 0,
            'hitbox_br_x': grid_width - 1,
            'hitbox_br_y': grid_height - 1,
            'color_r': 0,
            'color_g': 0,
            'color_b': 0,
            'color_state': 0,
            'never_hit': True,
            'existence': False,
        }

        self.init_grid()

        self.init_walls()

#        self.first_brick = 11, 10
#        self.brick_nrow, self.brick_ncol = 3, 8
#        self.brick_distance = 14, 10
#        self.brick_halfwidth, self.brick_halfheight = 5, 2
#
#        self.init_bricks()
#
#        self.paddle_x, self.paddle_y = 60, 60
#        self.paddle_halfwidth, self.paddle_halfheight = 5, 1
#        self.paddle_base_speed = 2
#
#        self.init_paddle()

        self.ball_x, self.ball_y = 40 + random.randint(0, 10), 40 + random.randint(0, 10)
        self.ball_radius = 1
        self.ball_speed_x, self.ball_speed_y = 1, 1

        self.init_ball()

        self.event_pending = []


    def init_grid(self):

        self.grid = np.zeros((grid_width, grid_height), dtype= int)
        self.r = np.zeros((grid_width, grid_height), dtype= int)
        self.g = np.zeros((grid_width, grid_height), dtype= int)
        self.b = np.zeros((grid_width, grid_height), dtype= int)


    def init_walls(self):

        self.grid[0:3, 3:grid_height - 3] = 5 # left wall
        self.r[0:3, 3:grid_height - 3] = 0
        self.g[0:3, 3:grid_height - 3] = 255
        self.b[0:3, 3:grid_height - 3] = 50

        self.elements['wall_left'] = {
            'id': 5,
            'pos_x': 1,
            'pos_y': math.floor(grid_height / 2),
            'shape_x': 1,
            'shape_y': math.floor(grid_height / 2),
            'hitbox_tl_x': 0,
            'hitbox_tl_y': 3,
            'hitbox_br_x': 2,
            'hitbox_br_y': grid_height - 4,
            'color_r': 0,
            'color_g': 255,
            'color_b': 50,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }

        self.grid[grid_width - 3:grid_width, 3:grid_height - 3] = 6 # right wall
        self.r[grid_width - 3:grid_width, 3:grid_height - 3] = 0
        self.g[grid_width - 3:grid_width, 3:grid_height - 3] = 255
        self.b[grid_width - 3:grid_width, 3:grid_height - 3] = 100

        self.elements['wall_right'] = {
            'id': 6,
            'pos_x': grid_width - 2,
            'pos_y': math.floor(grid_height / 2),
            'shape_x': 1,
            'shape_y': math.floor(grid_height / 2),
            'hitbox_tl_x': grid_width - 3,
            'hitbox_tl_y': 3,
            'hitbox_br_x': grid_width - 1,
            'hitbox_br_y': grid_height - 4,
            'color_r': 0,
            'color_g': 255,
            'color_b': 100,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }

        self.grid[3:grid_width - 3, 0:3] = 7 # top wall
        self.r[3:grid_width - 3, 0:3] = 0
        self.g[3:grid_width - 3, 0:3] = 255
        self.b[3:grid_width - 3, 0:3] = 150

        self.elements['wall_top'] = {
            'id': 7,
            'pos_x': math.floor(grid_width / 2),
            'pos_y': 1,
            'shape_x': math.floor(grid_width / 2),
            'shape_y': 1,
            'hitbox_tl_x': 3,
            'hitbox_tl_y': 0,
            'hitbox_br_x': grid_width - 4,
            'hitbox_br_y': 2,
            'color_r': 0,
            'color_g': 255,
            'color_b': 150,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }

        self.grid[3:grid_width - 3, grid_height - 3:grid_height] = 8 # bottom wall
        self.r[3:grid_width - 3, grid_height - 3:grid_height] = 0
        self.g[3:grid_width - 3, grid_height - 3:grid_height] = 255
        self.b[3:grid_width - 3, grid_height - 3:grid_height] = 200

        self.elements['wall_bottom'] = {
            'id': 8,
            'pos_x': math.floor(grid_width / 2),
            'pos_y': grid_height - 2,
            'shape_x': math.floor(grid_width / 2),
            'shape_y': 1,
            'hitbox_tl_x': 3,
            'hitbox_tl_y': grid_height - 3,
            'hitbox_br_x': grid_width - 4,
            'hitbox_br_y': grid_height - 1,
            'color_r': 0,
            'color_g': 255,
            'color_b': 150,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }


    def init_bricks(self):

        self.brick_positions = [(self.first_brick[0] + j * self.brick_distance[0], self.first_brick[1] + i * self.brick_distance[1]) for j in range(self.brick_ncol) for i in range(self.brick_nrow)]
        self.bricks_alive = len(self.brick_positions)

        for i, brick_pos in enumerate(self.brick_positions):

            self.grid[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = i + 9
            self.r[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 255
            self.g[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 255
            self.b[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 255
            
            self.elements[f'brick_{i}'] = {
                'id': i + 9,
                'pos_x': brick_pos[0],
                'pos_y': brick_pos[1],
                'shape_x': self.brick_halfwidth,
                'shape_y': self.brick_halfheight,
                'hitbox_tl_x': brick_pos[0] - self.brick_halfwidth,
                'hitbox_tl_y': brick_pos[1] - self.brick_halfheight,
                'hitbox_br_x': brick_pos[0] + self.brick_halfwidth,
                'hitbox_br_y': brick_pos[1] + self.brick_halfheight,
                'color_r': 255,
                'color_g': 255,
                'color_b': 255,
                'color_state': 0,
                'never_hit': True,
                'existence': True,
            }


    def hit_brick(self, id):

        brick_id = id - 9
        brick_pos = self.brick_positions[brick_id]

        if False:
        #if self.elements[f'brick_{brick_id}']['never_hit']: # first hit change color, the second destroy the brick

            self.r[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 0

            self.elements[f'brick_{brick_id}']['never_hit'] = False
            self.elements[f'brick_{brick_id}']['existence'] = False

        else:
        
            self.grid[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 0
            self.r[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 0
            self.g[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 0
            self.b[brick_pos[0] - self.brick_halfwidth:brick_pos[0] + self.brick_halfwidth + 1, brick_pos[1] - self.brick_halfheight:brick_pos[1] + self.brick_halfheight + 1] = 0
            
            self.elements[f'brick_{brick_id}']['alive'] = False
            self.bricks_alive -= 1

    def hit_wall(self, id):

        match(id):

            case 5: # wall_left
                color_state = self.elements['wall_left']['color_state'] + 1
                if color_state == 3: color_state = 0
                self.elements['wall_left']['color_state'] = color_state

                self.r[0:3, 3:grid_height - 3] = 100 * color_state

            case 6: # wall_right
                color_state = self.elements['wall_right']['color_state'] + 1
                if color_state == 3: color_state = 0
                self.elements['wall_right']['color_state'] = color_state

                self.r[grid_width - 3:grid_width, 3:grid_height - 3] = 100 * color_state

            case 7: # wall_top
                color_state = self.elements['wall_top']['color_state'] + 1
                if color_state == 3: color_state = 0
                self.elements['wall_top']['color_state'] = color_state

                self.r[3:grid_width - 3, 0:3] = 100 * color_state

            case 8: # wall_bottom
                color_state = self.elements['wall_bottom']['color_state'] + 1
                if color_state == 3: color_state = 0
                self.elements['wall_bottom']['color_state'] = color_state

                self.r[3:grid_width - 3, grid_height - 3:grid_height] = 100 * color_state

                self.bricks_alive = 0


    def init_paddle(self):
        
        self.paddle_speed = 0
        self.paddle_old_x, self.paddle_old_y = self.paddle_x, self.paddle_y
        self.draw_paddle()
        self.elements['paddle_center'] = {
            'id': 3,
            'pos_x': self.paddle_x,
            'pos_y': self.paddle_y,
            'shape_x': self.paddle_halfwidth,
            'shape_y': self.paddle_halfheight,
            'hitbox_tl_x': self.paddle_x - self.paddle_halfwidth,
            'hitbox_tl_y': self.paddle_y - self.paddle_halfheight,
            'hitbox_br_x': self.paddle_x + self.paddle_halfwidth,
            'hitbox_br_y': self.paddle_y + self.paddle_halfheight,
            'color_r': 0,
            'color_g': 0,
            'color_b': 255,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }


    def set_paddle_speed(self, value):
        self.paddle_speed = value * self.paddle_base_speed


    def update_paddle(self):

        if self.paddle_speed != 0:
        
            if (self.paddle_x - self.paddle_halfwidth + self.paddle_speed > 2) and (self.paddle_x + self.paddle_halfwidth + self.paddle_speed < grid_width - 3):
                if (self.ball_y + self.ball_radius > self.paddle_y - self.paddle_halfheight - 1 and self.ball_y - self.ball_radius < self.paddle_y + self.paddle_halfheight + 1) and (self.ball_x + self.ball_radius > self.paddle_x - self.paddle_halfwidth + self.paddle_speed and self.ball_x - self.ball_radius < self.paddle_x + self.paddle_halfwidth + self.paddle_speed):
                    pass

                else:
                    self.paddle_old_x = self.paddle_x
                    self.paddle_x += self.paddle_speed

            self.elements['paddle_center']['pos_x'] = self.paddle_x
            self.elements['paddle_center']['pos_y'] = self.paddle_y
            self.elements['paddle_center']['hitbox_tl_x'] = self.paddle_x - self.paddle_halfwidth
            self.elements['paddle_center']['hitbox_tl_y'] = self.paddle_y - self.paddle_halfheight
            self.elements['paddle_center']['hitbox_br_x'] = self.paddle_x + self.paddle_halfwidth
            self.elements['paddle_center']['hitbox_br_y'] = self.paddle_y + self.paddle_halfheight

    def draw_paddle(self):

        self.grid[self.paddle_old_x - self.paddle_halfwidth:self.paddle_old_x + self.paddle_halfwidth + 1, self.paddle_old_y - self.paddle_halfheight:self.paddle_old_y + self.paddle_halfheight + 1] = 0
#        self.r[self.paddle_old_x - self.paddle_halfwidth:self.paddle_old_x + self.paddle_halfwidth + 1, self.paddle_old_y - self.paddle_halfheight:self.paddle_old_y + self.paddle_halfheight + 1] = 0
#        self.g[self.paddle_old_x - self.paddle_halfwidth:self.paddle_old_x + self.paddle_halfwidth + 1, self.paddle_old_y - self.paddle_halfheight:self.paddle_old_y + self.paddle_halfheight + 1] = 0
        self.b[self.paddle_old_x - self.paddle_halfwidth:self.paddle_old_x + self.paddle_halfwidth + 1, self.paddle_old_y - self.paddle_halfheight:self.paddle_old_y + self.paddle_halfheight + 1] = 0

        self.grid[self.paddle_x - self.paddle_halfwidth:self.paddle_x + self.paddle_halfwidth + 1, self.paddle_y - self.paddle_halfheight:self.paddle_y + self.paddle_halfheight + 1] = 3
#        self.r[self.paddle_x - self.paddle_halfwidth:self.paddle_x + self.paddle_halfwidth + 1, self.paddle_y - self.paddle_halfheight:self.paddle_y + self.paddle_halfheight + 1] = 0
#        self.g[self.paddle_x - self.paddle_halfwidth:self.paddle_x + self.paddle_halfwidth + 1, self.paddle_y - self.paddle_halfheight:self.paddle_y + self.paddle_halfheight + 1] = 0
        self.b[self.paddle_x - self.paddle_halfwidth:self.paddle_x + self.paddle_halfwidth + 1, self.paddle_y - self.paddle_halfheight:self.paddle_y + self.paddle_halfheight + 1] = 255


    def init_ball(self):
        
        self.ball_old_x, self.ball_old_y = self.ball_x, self.ball_y
        self.draw_ball()
        self.elements['ball'] = {
            'id': 1,
            'pos_x': self.ball_x,
            'pos_y': self.ball_y,
            'shape_x': self.ball_radius,
            'shape_y': self.ball_radius,
            'hitbox_tl_x': self.ball_x - self.ball_radius,
            'hitbox_tl_y': self.ball_y - self.ball_radius,
            'hitbox_br_x': self.ball_x + self.ball_radius,
            'hitbox_br_y': self.ball_y + self.ball_radius,
            'color_r': 255,
            'color_g': 0,
            'color_b': 0,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }


    def update_ball(self):
        
        invert_speed_x = False
        invert_speed_y = False

        collisions = []
        
        ball_new_x = self.ball_x + self.ball_speed_x
        ball_new_y = self.ball_y + self.ball_speed_y


        if np.any(self.grid[ball_new_x - self.ball_radius: ball_new_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius + 1] != 0):
            invert_speed_x = True
            collisions.extend(set(list(self.grid[ball_new_x - self.ball_radius: ball_new_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius + 1].ravel())))

        if np.any(self.grid[self.ball_x - self.ball_radius: self.ball_x + self.ball_radius + 1, ball_new_y - self.ball_radius:ball_new_y + self.ball_radius + 1] != 0):
            invert_speed_y = True
            collisions.extend(set(list(self.grid[self.ball_x - self.ball_radius: self.ball_x + self.ball_radius + 1, ball_new_y - self.ball_radius:ball_new_y + self.ball_radius + 1].ravel())))

        if (not invert_speed_x) and (not invert_speed_y):
            if np.any(self.grid[ball_new_x - self.ball_radius: ball_new_x + self.ball_radius + 1, ball_new_y - self.ball_radius:ball_new_y + self.ball_radius + 1] != 0):
                invert_speed_x = True
                invert_speed_y = True
                collisions.extend(set(list(self.grid[ball_new_x - self.ball_radius: ball_new_x + self.ball_radius + 1, ball_new_y - self.ball_radius:ball_new_y + self.ball_radius + 1].ravel())))


        ## qui si ha la collisione, si puo mettere "no movimento" e "disappearance del brick" in coda come eventi nel prossimo frame
        
        #if invert_speed_x: self.ball_speed_x = -self.ball_speed_x
        #if invert_speed_y: self.ball_speed_y = -self.ball_speed_y

        if invert_speed_x or invert_speed_y:
            if invert_speed_x:
                #self.event_pending.append((self.bounce_x, None, 1))
                self.ball_speed_x = -self.ball_speed_x
            if invert_speed_y:
                #self.event_pending.append((self.bounce_y, None, 1))
                self.ball_speed_y = -self.ball_speed_y
                
            self.ball_old_x = self.ball_x
            self.ball_old_y = self.ball_y

            self.ball_x += self.ball_speed_x
            self.ball_y += self.ball_speed_y

        else:
            self.ball_old_x = self.ball_x
            self.ball_old_y = self.ball_y

            self.ball_x += self.ball_speed_x
            self.ball_y += self.ball_speed_y
        
        for collision_id in collisions:
            if collision_id != 0:
                if collision_id >= 9:
                    self.event_pending.append((self.hit_brick, collision_id, 1))
                
                elif collision_id >= 5:
                    self.event_pending.append((self.hit_wall, collision_id, 2))

        ######

        self.elements['ball']['pos_x'] = self.ball_x
        self.elements['ball']['pos_y'] = self.ball_y
        self.elements['ball']['hitbox_tl_x'] = self.ball_x - self.ball_radius
        self.elements['ball']['hitbox_tl_y'] = self.ball_y - self.ball_radius
        self.elements['ball']['hitbox_br_x'] = self.ball_x + self.ball_radius
        self.elements['ball']['hitbox_br_y'] = self.ball_y + self.ball_radius

    def draw_ball(self):

        #self.grid[self.ball_old_x - self.ball_radius:self.ball_old_x + self.ball_radius + 1, self.ball_old_y - self.ball_radius:self.ball_old_y + self.ball_radius] = 0
        self.r[self.ball_old_x - self.ball_radius:self.ball_old_x + self.ball_radius + 1, self.ball_old_y - self.ball_radius:self.ball_old_y + self.ball_radius + 1] = 0
#        self.g[self.ball_old_x - self.ball_radius:self.ball_old_x + self.ball_radius + 1, self.ball_old_y - self.ball_radius:self.ball_old_y + self.ball_radius + 1] = 0
#        self.b[self.ball_old_x - self.ball_radius:self.ball_old_x + self.ball_radius + 1, self.ball_old_y - self.ball_radius:self.ball_old_y + self.ball_radius + 1] = 0

        #self.grid[self.ball_x - self.ball_radius:self.ball_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius] = 1
        self.r[self.ball_x - self.ball_radius:self.ball_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius + 1] = 255
#        self.g[self.ball_x - self.ball_radius:self.ball_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius + 1] = 0
#        self.b[self.ball_x - self.ball_radius:self.ball_x + self.ball_radius + 1, self.ball_y - self.ball_radius:self.ball_y + self.ball_radius + 1] = 0

    def bounce_x(self):
        self.ball_speed_x = - self.ball_speed_x

    def bounce_y(self):
        self.ball_speed_y = - self.ball_speed_y

    def resolve_pending(self):
        new_event_pending = []
        for event, param, delay in self.event_pending:
            if delay > 1: new_event_pending.append((event, param, delay - 1))
            else:
                if param is None: event()
                else: event(param)
        
        self.event_pending = new_event_pending

    def update(self):
        self.resolve_pending()
        #self.update_paddle()
        #self.draw_paddle()
        self.update_ball()
        self.draw_ball()

        event_log = self.event_log
        self.event_log = []

        return self.elements, event_log, False#(self.bricks_alive == 0)
    
    def get_log(self): return self.elements, self.event_log
    
    def get_grid(self):
        return np.transpose(np.stack([self.r, self.g, self.b]), (1, 2, 0))


# Main

save_log = True

pygame.init()
window = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Basic Arkanoid")

game = Game()

grid = pygame.surfarray.make_surface(game.get_grid())
screen = pygame.transform.scale(grid, (screen_width, screen_height))
window.blit(screen, (0, 0))

t = time.time()
keys_down = []
keys_up = []
paddle_left = False
paddle_right = False
first_time_run = True
screen_running = True
game_running = False

element_log, event_log = game.get_log()

frame_id = 0
frames = []
frames.append({
    'frame_id': frame_id,
    'commands': [],
    'elements': deepcopy(element_log),
    'events': [{
        'description': 'game_start',
        'subject': 0
    }]
})
frame_id += 1

while screen_running:

    for e in pygame.event.get():

        if e == pygame.QUIT:
            screen_running = False

        if e.type == pygame.KEYDOWN:
            keys_down.append(e.key)
            keydown = True

        if e.type == pygame.KEYUP:
            keys_up.append(e.key)
            keyup = True

    new_t = time.time()
    if new_t - t > refresh_rate:
        
        command_log = []

        if first_time_run:
            if not game_running and len(keys_down) > 0:
                game_running = True

        if pygame.K_q in keys_down:
            screen_running = False

        if pygame.K_s in keys_down:
            game_running = not game_running

        if pygame.K_UP in keys_down:
            refresh_rate -= refresh_rate / 2
            print(refresh_rate)
        if pygame.K_DOWN in keys_down:
            refresh_rate += refresh_rate / 2
            print(refresh_rate)

        if game_running:

            element_log, event_log, end_game = game.update()

            #if first_time_run:
            #    event_log.append({
            #        'description': 'game_start',
            #        'subject': 0
            #    })
            #    first_time_run = False

            if event_log:
                print(frame_id)
                print(event_log)

            frames.append({
                'frame_id': frame_id,
                'commands': [c for c in command_log],
                'elements': deepcopy(element_log),
                'events': event_log
                })
            frame_id += 1

            grid = pygame.surfarray.make_surface(game.get_grid())
            screen = pygame.transform.scale(grid, (screen_width, screen_height))
            window.blit(screen, (0, 0))

            if end_game:
                game_running = False
                screen_running = False
    
        t = new_t
        keys_down = []
        keys_up = []

    # Refresh the display
    pygame.display.flip()

#frames.append({
#    'frame_id': frame_id,
#    'commands': [],
#    'elements': {},
#    'events': [{'description': 'game_end', 'subject': 0}]
#    })

if save_log:
    
    with open(f'logs/arkanoid_logs/arkanoid_log{datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")}.pkl', 'wb') as logfile:
        pickle.dump(frames, logfile)

pygame.quit()
sys.exit()
