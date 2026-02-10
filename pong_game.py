import pygame
import sys
import numpy as np

# =====================
# CONFIG
# =====================
WIDTH = 400
HEIGHT = 300
FPS = 60
LIVES = 1  # Numero di vite iniziali

# =====================
# GAME LOGIC
# =====================
class PongGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.lives = LIVES
        self.ball_radius = 4
        self.paddle_w = 60
        self.paddle_h = 10
        self.reset()
        self.init_grid()

    def reset(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vx = np.random.choice([-3, 3])
        self.ball_vy = -3

        self.player_x = self.width // 2
        self.player_y = self.height - 20
        self.enemy_x = self.width // 2
        self.enemy_y = 20

        self.done = False
        self.lives_left = self.lives

        # paddle action
        self.pending_action = 0

    def set_paddle(self, direction):
        self.pending_action = direction  # -1 left, 0 stop, 1 right

    def update(self):
        # Muovi paddle giocatore
        self.player_x += self.pending_action * 5
        self.player_x = np.clip(self.player_x, self.paddle_w//2, self.width - self.paddle_w//2)

        # Muovi paddle avversario
        if self.ball_x > self.enemy_x:
            self.enemy_x += 3
        else:
            self.enemy_x -= 3
        self.enemy_x = np.clip(self.enemy_x, self.paddle_w//2, self.width - self.paddle_w//2)

        # Muovi palla
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Collisioni pareti
        if self.ball_x - self.ball_radius <= 0 or self.ball_x + self.ball_radius >= self.width:
            self.ball_vx *= -1
        if self.ball_y - self.ball_radius <= 0:
            self.ball_vy *= -1

        # Collisione paddle giocatore
        if (
            self.ball_y + self.ball_radius >= self.player_y and
            abs(self.ball_x - self.player_x) < self.paddle_w // 2
        ):
            self.ball_vy *= -1

        # Collisione paddle avversario
        if (
            self.ball_y - self.ball_radius <= self.enemy_y + self.paddle_h and
            abs(self.ball_x - self.enemy_x) < self.paddle_w // 2
        ):
            self.ball_vy *= -1

        # Perdita palla
        if self.ball_y - self.ball_radius > self.height:
            self.lives_left -= 1
            self.done = True

    def init_grid(self):
        # Griglie RGB
        self.r = np.zeros((self.width, self.height), dtype=np.uint8)
        self.g = np.zeros((self.width, self.height), dtype=np.uint8)
        self.b = np.zeros((self.width, self.height), dtype=np.uint8)

    def get_grid(self):
        # Reset grid
        self.r.fill(0)
        self.g.fill(0)
        self.b.fill(0)

        # paddle giocatore (blu)
        x0 = max(0, int(self.player_x - self.paddle_w//2))
        x1 = min(self.width, int(self.player_x + self.paddle_w//2))
        y0 = max(0, int(self.player_y - self.paddle_h//2))
        y1 = min(self.height, int(self.player_y + self.paddle_h//2))
        self.b[x0:x1, y0:y1] = 255

        # paddle avversario (rosso)
        x0 = max(0, int(self.enemy_x - self.paddle_w//2))
        x1 = min(self.width, int(self.enemy_x + self.paddle_w//2))
        y0 = max(0, int(self.enemy_y - self.paddle_h//2))
        y1 = min(self.height, int(self.enemy_y + self.paddle_h//2))
        self.r[x0:x1, y0:y1] = 255

        # palla (bianca)
        x0 = max(0, int(self.ball_x - self.ball_radius))
        x1 = min(self.width, int(self.ball_x + self.ball_radius))
        y0 = max(0, int(self.ball_y - self.ball_radius))
        y1 = min(self.height, int(self.ball_y + self.ball_radius))
        self.r[x0:x1, y0:y1] = 255
        self.g[x0:x1, y0:y1] = 255
        self.b[x0:x1, y0:y1] = 255

        # Trasponi per avere (H, W, 3)
        # return np.transpose(np.stack([self.r, self.g, self.b]), (0, 1, 2))
        return np.transpose(np.stack([self.r, self.g, self.b]), (2, 1, 0))

    def draw(self, screen):
        screen.fill((0, 0, 0))

        # ball
        pygame.draw.circle(screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), self.ball_radius)

        # paddles
        pygame.draw.rect(screen, (0, 0, 255),
                         (self.player_x - self.paddle_w // 2, self.player_y - self.paddle_h//2, self.paddle_w, self.paddle_h))
        pygame.draw.rect(screen, (255, 0, 0),
                         (self.enemy_x - self.paddle_w // 2, self.enemy_y - self.paddle_h//2, self.paddle_w, self.paddle_h))

        # vite rimanenti
        font = pygame.font.SysFont(None, 24)
        lives_text = font.render(f"Vite: {self.lives_left}", True, (255, 255, 255))
        screen.blit(lives_text, (10, 10))


# # =====================
# # PYGAME LOOP
# # =====================
# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Pong – Single Player")
# clock = pygame.time.Clock()

# game = PongGame(LIVES)

# left = False
# right = False
# running = True

# while running:
#     clock.tick(FPS)

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 left = True
#             if event.key == pygame.K_RIGHT:
#                 right = True
#         elif event.type == pygame.KEYUP:
#             if event.key == pygame.K_LEFT:
#                 left = False
#             if event.key == pygame.K_RIGHT:
#                 right = False

#     if left and not right:
#         game.set_paddle(-1)
#     elif right and not left:
#         game.set_paddle(1)
#     else:
#         game.set_paddle(0)

#     game.update()
#     game.draw(screen)       # draw the current game state

#     # get_grid non prende argomenti
#     grid = game.get_grid()  

#     pygame.display.flip()

#     if game.done:
#         if game.lives_left <= 0:
#             # Game Over
#             screen.fill((0, 0, 0))
#             font = pygame.font.SysFont(None, 48)
#             text = font.render("Sconfitta!", True, (255, 0, 0))
#             screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
#             pygame.display.flip()
#             pygame.time.wait(2000)
#             running = False
#         else:
#             # Reset partita con una vita in meno
#             pygame.time.wait(1000)
#             game.reset()

# pygame.quit()
# sys.exit()