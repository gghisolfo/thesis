import gymnasium as gym
import numpy as np
from catching_game import CatchGame, grid_width, grid_height, MAX_MISS
from arkanoid_game import Game, grid_width, grid_height

# === Ambiente personalizzato CatchGame minimale ===
class CatchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = CatchGame()
        self.action_space = gym.spaces.Discrete(3)  # 0=sinistra, 1=fermo, 2=destra
        # Osservazione: paddle_x, paddle_y, object_x, object_y, object_speed
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

    def reset(self):
        self.game = CatchGame()
        self.done = False
        return self._get_obs()

    def step(self, action):
        # Esegui azione
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        # Aggiorna simulazione
        self.game.update()

        # Check cattura o perdita - Check fine episodio
        # done = self.game.caught or self.game.missed
        self.done = self.game.missed >= MAX_MISS  # Termina dopo 10 oggetti mancati

        return self._get_obs(), self.done, {}

    def _get_obs(self):
        # Normalizza coordinate da -1 a 1
        paddle_x = self.game.agent_x / grid_width
        paddle_y = self.game.agent_y / grid_height
        obj_x = self.game.obj_x / grid_width
        obj_y = self.game.obj_y / grid_height
        obj_speed = self.game.obj_speed / grid_height
        return np.array([paddle_x*2-1, paddle_y*2-1, obj_x*2-1, obj_y*2-1, obj_speed], dtype=np.float32)

class ArkanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()
        # self.game.ball_lost = False
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False

        # Stato precedente per il reward shaping
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y

    def reset(self):
        self.game = Game()
        # self.game.ball_lost = False
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y
        self.done = False
        return self._get_obs()

    def step(self, action):
        # Salva lo stato precedente
        prev_bricks = self.game.bricks_alive
        prev_ball_y = self.game.ball_y

        # Esegui azione
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        # Aggiorna il gioco
        self.game.update()

        
        # Controlla condizioni di terminazione
        # 1. Tutti i brick distrutti (VITTORIA)
        if self.game.bricks_alive == 0:
            self.done = True

        # 2. Palla persa (SCONFITTA)
        if self.game.ball_lost:
            self.done = True

        return self._get_obs(), self.done, {}


    def _get_obs(self):
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)
