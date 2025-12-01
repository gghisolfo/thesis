import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
from collections import deque, defaultdict

#python -m dqn.dqn_sim_env 

# Import locali 
from arkanoid_game import Game, grid_width, grid_height

SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)

class EventChainTracker:
    """
    Traccia la catena di eventi fisici triggerati da un'azione.
    Ogni evento nella sequenza incrementa il reward.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.events = []
        self.chain_length = 0
    
    def add_event(self, event_type, details=None):
        """
        Registra un evento fisico.
        event_type: str - tipo di evento ('wall_collision', 'object_destroyed', 'paddle_hit', etc.)
        details: dict - dettagli opzionali sull'evento
        """
        self.events.append({
            'type': event_type,
            'details': details or {},
            'index': len(self.events)
        })
        self.chain_length += 1
    
    def get_chain_reward(self):
        """
        Calcola il reward basato sulla lunghezza della catena di eventi.
        Più eventi in sequenza = reward maggiore (con bonus esponenziale).
        """
        if self.chain_length == 0:
            return 0.0
        
        # Reward base per ogni evento
        base_reward = self.chain_length
        
        # Bonus esponenziale per catene lunghe
        chain_bonus = 0.0
        if self.chain_length >= 2:
            chain_bonus = (self.chain_length - 1) ** 1.5
        
        return base_reward + chain_bonus


class SymbolicPhysicsEnv(gym.Env):
    """
    Ambiente che usa il modulo simbolico per simulare la fisica
    e traccia gli eventi per calcolare il reward.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.game = Game()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.done = False
        self.event_tracker = EventChainTracker()
        
        # Stato precedente per rilevare eventi
        self._prev_state = self._capture_state()

    def reset(self):
        self.game = Game()
        self.done = False
        self.event_tracker.reset()
        self._prev_state = self._capture_state()
        return self._get_obs()

    def step(self, action):
        # Reset tracker eventi per questo step
        self.event_tracker.reset()
        
        # Cattura stato prima dell'azione
        prev_state = self._prev_state
        
        # Esegui azione
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)

        # Aggiorna il gioco (simula fisica simbolica)
        self.game.update()
        
        # Cattura nuovo stato
        current_state = self._capture_state()
        
        # Rileva eventi fisici triggerati dall'azione
        self._detect_physics_events(prev_state, current_state)
        
        # Calcola reward basato sulla catena di eventi
        reward = self.event_tracker.get_chain_reward()
        
        # Aggiorna stato precedente
        self._prev_state = current_state
        
        # Controlla condizioni di terminazione
        if self.game.bricks_alive == 0:
            self.done = True
            reward += 100.0
            self.event_tracker.add_event('victory')
            print(f"🎉 VITTORIA! Catena finale: {self.event_tracker.chain_length} eventi")

        if self.game.ball_lost or self.game.ball_y + self.game.ball_radius >= grid_height - 3:
            self.done = True
            reward -= 50.0
            self.event_tracker.add_event('ball_lost')

        # Debug output per catene interessanti
        if self.event_tracker.chain_length > 2:
            print(f"⛓️  Catena di {self.event_tracker.chain_length} eventi → Reward: {reward:.2f}")
            for evt in self.event_tracker.events:
                print(f"   {evt['index']+1}. {evt['type']}")

        return self._get_obs(), reward, self.done, {
            'events': self.event_tracker.events,
            'chain_length': self.event_tracker.chain_length
        }

    def _capture_state(self):
        """Cattura lo stato fisico completo del gioco."""
        return {
            'ball_x': self.game.ball_x,
            'ball_y': self.game.ball_y,
            'ball_vx': self.game.ball_speed_x,
            'ball_vy': self.game.ball_speed_y,
            'paddle_x': self.game.paddle_x,
            'bricks_alive': self.game.bricks_alive,
            'ball_lost': self.game.ball_lost
        }

    def _detect_physics_events(self, prev_state, current_state):
        """
        Rileva gli eventi fisici confrontando stato precedente e corrente.
        Ogni evento rilevato viene aggiunto al tracker.
        """
        
        # 1. Collisione con muri (cambio direzione)
        if prev_state['ball_vx'] * current_state['ball_vx'] < 0:
            self.event_tracker.add_event('wall_collision_horizontal', {
                'position': current_state['ball_x']
            })
        
        if prev_state['ball_vy'] * current_state['ball_vy'] < 0:
            # Distingui tra muro superiore e paddle
            if current_state['ball_y'] < grid_height / 2:
                self.event_tracker.add_event('wall_collision_top', {
                    'position': current_state['ball_y']
                })
            else:
                # Probabile collisione con paddle
                self.event_tracker.add_event('paddle_collision', {
                    'paddle_x': current_state['paddle_x'],
                    'ball_x': current_state['ball_x']
                })
        
        # 2. Distruzione di brick (evento più importante!)
        bricks_destroyed = prev_state['bricks_alive'] - current_state['bricks_alive']
        if bricks_destroyed > 0:
            for _ in range(bricks_destroyed):
                self.event_tracker.add_event('brick_destroyed', {
                    'remaining': current_state['bricks_alive']
                })
        
        # 3. Movimento significativo della palla (per catene lunghe)
        distance_traveled = np.sqrt(
            (current_state['ball_x'] - prev_state['ball_x'])**2 +
            (current_state['ball_y'] - prev_state['ball_y'])**2
        )
        if distance_traveled > 5:  # Soglia arbitraria
            self.event_tracker.add_event('ball_movement', {
                'distance': distance_traveled
            })

    def _get_obs(self):
        """Normalizza lo stato del gioco per la rete neurale."""
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)


class QNetwork(nn.Module):
    """Rete Q-Network per l'apprendimento."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_event_chain_dqn(total_episodes=1000, max_steps=2000):
    """
    Training DQN con reward basato su catene di eventi.
    """
    population = None

    # Setup ambiente
    env = SymbolicPhysicsEnv()
    
    buffer = deque(maxlen=50000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995

    rewards_history = []
    chain_lengths_history = []

    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        total_chain_events = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_net(s_t)
                    action = int(q_vals.argmax(1).item())

            next_state, reward, done, info = env.step(action)

            # Accumula statistiche sugli eventi
            total_chain_events += info.get('chain_length', 0)

            # Aggiungi al buffer

            buffer.append((state, action, reward, next_state, done))

            # Training step
            if len(buffer) >= 64:
                
                batch = random.sample(buffer, 64)
                s, a, r, ns, d = zip(*batch)
                
                s_t = torch.tensor(np.array(s), device=device)
                a_t = torch.tensor(a, device=device).unsqueeze(1)
                r_t = torch.tensor(r, device=device)
                ns_t = torch.tensor(np.array(ns), device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                with torch.no_grad():
                    max_next_q = q_target(ns_t).max(1)[0]
                    target_q = r_t + gamma * (1 - d_t) * max_next_q

                current_q = q_net(s_t).gather(1, a_t).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_reward += reward
            state = next_state
            steps += 1

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network
        if ep % 10 == 0:
            q_target.load_state_dict(q_net.state_dict())
        
        rewards_history.append(total_reward)
        chain_lengths_history.append(total_chain_events)
        
        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_chains = np.mean(chain_lengths_history[-10:])
            print(f"[Ep {ep}] Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f} | Eventi totali: {total_chain_events} | Avg eventi: {avg_chains:.2f}")

    # Salva il modello
    model_name = "dqn_event_chain.pth"
    final_path = os.path.join(SAVE_DIR, model_name)
    torch.save(q_net.state_dict(), final_path)
    
    print(f"\n✅ Training completo! Modello salvato in {final_path}")
    print(f"📊 Reward medio: {np.mean(rewards_history):.2f}")
    print(f"⛓️  Eventi medi per episodio: {np.mean(chain_lengths_history):.2f}")
    
    return rewards_history, chain_lengths_history


if __name__ == "__main__":
    print("🚀 Avvio training DQN con reward basato su catene di eventi fisici")
    print("=" * 70)
    
    rewards, chains = train_event_chain_dqn(
        total_episodes=1000,
        max_steps=2000
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward massimo: {max(rewards):.2f}")
    print(f"   Catena più lunga: {max(chains)} eventi")
    print(f"   Performance negli ultimi 100 episodi: {np.mean(rewards[-100:]):.2f}")