import pygame
import sys
import numpy as np

# =====================
# CONFIG
# =====================
WIDTH = 400
HEIGHT = 300
FPS = 60



# =====================
# GAME LOGIC
# =====================
class PongGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.reset()

    def reset(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vx = np.random.choice([-3, 3])
        self.ball_vy = -3

        self.paddle_w = 60
        self.player_y = self.height - 20
        self.enemy_y = 20

        self.player_x = self.width // 2
        self.enemy_x = self.width // 2

        self.done = False
        self.pending_action = 0

    def set_paddle(self, direction):
        self.pending_action = direction  # -1 left, 0 stop, 1 right

    def update(self):
        # player paddle
        self.player_x += self.pending_action * 5
        self.player_x = np.clip(self.player_x, 0, self.width)

        # enemy AI
        if self.ball_x > self.enemy_x:
            self.enemy_x += 3
        else:
            self.enemy_x -= 3
        self.enemy_x = np.clip(self.enemy_x, 0, self.width)

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # wall collision
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_vx *= -1

        # player paddle collision
        if (
            self.ball_y >= self.player_y - 5 and
            abs(self.ball_x - self.player_x) < self.paddle_w // 2
        ):
            self.ball_vy *= -1

        # enemy paddle collision
        if (
            self.ball_y <= self.enemy_y + 5 and
            abs(self.ball_x - self.enemy_x) < self.paddle_w // 2
        ):
            self.ball_vy *= -1

        # goal
        if self.ball_y > self.height or self.ball_y < 0:
            self.done = True

    def draw(self, screen):
        screen.fill((0, 0, 0))

        # ball
        pygame.draw.circle(
            screen, (255, 255, 255),
            (int(self.ball_x), int(self.ball_y)), 4
        )

        # paddles
        pygame.draw.rect(
            screen, (255, 255, 255),
            (self.player_x - self.paddle_w // 2, self.player_y, self.paddle_w, 5)
        )
        pygame.draw.rect(
            screen, (255, 255, 255),
            (self.enemy_x - self.paddle_w // 2, self.enemy_y, self.paddle_w, 5)
        )


# =====================
# PYGAME LOOP
# =====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong – Single Player")
clock = pygame.time.Clock()

game = PongGame()

left = False
right = False
running = True

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                left = True
            if event.key == pygame.K_RIGHT:
                right = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                left = False
            if event.key == pygame.K_RIGHT:
                right = False

    if left and not right:
        game.set_paddle(-1)
    elif right and not left:
        game.set_paddle(1)
    else:
        game.set_paddle(0)

    game.update()
    game.draw(screen)

    pygame.display.flip()

    if game.done:
        pygame.time.wait(1000)
        game.reset()

pygame.quit()
sys.exit()