import pygame

class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()  # Niente parametri
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

        # --- Inizializza pygame per il rendering ---
        pygame.init()
        self.screen_width = 400
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Arkanoid RL")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.game = Game()  # Reset semplice
        self.done = False
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        self.game.update()
        reward = self._compute_reward()
        if self.game.bricks_alive == 0:
            self.done = True
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        ball_x = self.game.ball_x
        ball_y = self.game.ball_y
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)

    def _compute_reward(self):
        r = 0.1
        if hasattr(self.game, "ball_hit_paddle") and self.game.ball_hit_paddle:
            r += 1.0
        if hasattr(self.game, "brick_destroyed") and self.game.brick_destroyed:
            r += 2.0
        if hasattr(self.game, "ball_lost") and self.game.ball_lost:
            r -= 5.0
        return r

    def render(self, mode="human"):
        # --- gestione eventi per chiusura finestra ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.fill((0, 0, 0))  # sfondo nero

        # Disegna paddle
        paddle_rect = pygame.Rect(
            int(self.game.paddle_x),
            self.screen_height - 20,
            self.game.paddle_width,
            10
        )
        pygame.draw.rect(self.screen, (255, 255, 255), paddle_rect)

        # Disegna palla
        pygame.draw.circle(
            self.screen, (255, 0, 0),
            (int(self.game.ball_x), int(self.game.ball_y)), 5
        )

        # Disegna mattoni
        for brick in self.game.bricks:
            if brick.alive:
                rect = pygame.Rect(brick.x, brick.y, brick.width, brick.height)
                pygame.draw.rect(self.screen, (0, 255, 0), rect)

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def close(self):
        pygame.quit()
