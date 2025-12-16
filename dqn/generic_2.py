import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
from collections import deque, defaultdict
from typing import Any, Dict, List, Tuple, Callable

# python -m dqn.generic_2

# Import locali 
from arkanoid_game import Game, grid_width, grid_height

SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)


class GenericStateExtractor:
    """
    Estrae automaticamente tutti gli attributi osservabili da uno stato.
    Completamente agnostico al dominio.
    """
    def __init__(self, state_object):
        self.tracked_attributes = self._discover_attributes(state_object)
    
    def _discover_attributes(self, obj) -> List[str]:
        """Scopre automaticamente tutti gli attributi numerici/booleani."""
        attrs = []
        for attr in dir(obj):
            if not attr.startswith('_'):
                try:
                    value = getattr(obj, attr)
                    # Traccia solo attributi numerici o booleani
                    if isinstance(value, (int, float, bool, np.number)):
                        attrs.append(attr)
                except:
                    pass
        return attrs
    
    def extract(self, obj) -> Dict[str, Any]:
        """Estrae lo stato completo come dizionario."""
        state = {}
        for attr in self.tracked_attributes:
            try:
                state[attr] = getattr(obj, attr)
            except:
                state[attr] = None
        return state


class AgnosticEventDetector:
    """
    Rileva eventi SENZA priorità o classificazione.
    Tutti i cambiamenti significativi hanno uguale importanza.
    
    ZERO conoscenza del dominio: non sa cosa sia un "rimbalzo" o un "brick".
    """
    def __init__(self, 
                 threshold_absolute: float = 0.1,
                 threshold_relative: float = 0.05):
        """
        Args:
            threshold_absolute: Soglia assoluta per cambiamenti numerici
            threshold_relative: Soglia relativa (5% minimo)
        """
        self.threshold_abs = threshold_absolute
        self.threshold_rel = threshold_relative
    
    def detect_events(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        """
        Rileva TUTTI i cambiamenti significativi.
        Ogni attributo genera AL PIÙ un evento per step.
        
        IMPORTANTE: Nessuna priorità o classificazione degli eventi.
        """
        events = []
        
        for key in prev_state.keys():
            if key not in curr_state:
                continue
            
            prev_val = prev_state[key]
            curr_val = curr_state[key]
            
            # Salta valori None
            if prev_val is None or curr_val is None:
                continue
            
            # Rileva cambiamento generico (senza classificazione)
            event = self._detect_change(key, prev_val, curr_val)
            if event:
                events.append(event)
        
        return events
    
    def _detect_change(self, key: str, prev: Any, curr: Any) -> Dict:
        """
        Rileva cambiamento generico senza conoscere la natura.
        Ritorna evento SOLO se significativo.
        """
        # Case 1: Valori numerici (float, int grandi)
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            delta = abs(curr - prev)
            
            # Soglia assoluta
            if delta < self.threshold_abs:
                return None
            
            # Soglia relativa (5% minimo)
            relative = delta / (abs(prev) + 1e-9)
            if relative < self.threshold_rel:
                return None
            
            return {
                'attribute': key,
                'prev': prev,
                'curr': curr,
                'delta': delta,
                'timestamp': None
            }
        
        # Case 2: Valori discreti (bool, int piccoli)
        elif isinstance(prev, (bool, int)) and isinstance(curr, (bool, int)):
            if prev != curr:
                return {
                    'attribute': key,
                    'prev': prev,
                    'curr': curr,
                    'delta': abs(curr - prev) if isinstance(prev, int) else 1,
                    'timestamp': None
                }
        
        return None


class AgnosticRewardCalculator:
    """
    Calcola reward SOLO basandosi su:
    1. Numero di eventi (più eventi = stato più "interessante")
    2. Sopravvivenza (più step vivo = meglio)
    
    ZERO conoscenza sulla semantica degli eventi.
    """
    def __init__(self, 
                 survival_reward_per_step: float = 0.01,
                 event_reward_base: float = 1.0,
                 chain_exponent: float = 1.3):
        """
        Args:
            survival_reward_per_step: Piccolo bonus per ogni step vivo
            event_reward_base: Reward base per evento
            chain_exponent: Esponente per crescita super-lineare
        """
        self.survival_reward = survival_reward_per_step
        self.event_base = event_reward_base
        self.chain_exp = chain_exponent
    
    def calculate_step_reward(self, num_events: int, is_terminal: bool) -> float:
        """
        Calcola reward per uno step.
        
        Formula:
        - R_survival = 0.01 (se vivo)
        - R_events = base * (num_events ^ exponent)
        - R_death = -10.0 (se terminato)
        - R_total = R_survival + R_events + R_death
        """
        # Componente sopravvivenza
        survival = 0 if is_terminal else self.survival_reward
        
        # Componente eventi (crescita super-lineare)
        events = self.event_base * (num_events ** self.chain_exp) if num_events > 0 else 0
        
        # Penalità morte
        death_penalty = -10.0 if is_terminal else 0.0
        
        return survival + events + death_penalty


class EventTracker:
    """
    Tracker agnostico che conta eventi senza classificarli.
    """
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.reset()
    
    def reset(self):
        """Reset completo."""
        self.event_history = deque(maxlen=self.history_size)
        self.step_counter = 0
        self.total_events = 0
        self.events_per_step = []
        self.event_distribution = defaultdict(int)  # Per chain length
        
    def add_events(self, events: List[Dict]) -> int:
        """
        Aggiunge eventi e ritorna il numero rilevato.
        """
        num_events = len(events)
        
        # Assegna timestamp
        for event in events:
            event['timestamp'] = self.step_counter
            self.event_history.append(event)
        
        self.total_events += num_events
        self.events_per_step.append(num_events)
        self.event_distribution[num_events] += 1
        self.step_counter += 1
        
        return num_events
    
    def get_statistics(self) -> Dict:
        """Statistiche aggregate."""
        return {
            'total_events': self.total_events,
            'total_steps': self.step_counter,
            'avg_events_per_step': self.total_events / max(self.step_counter, 1),
            'max_events_in_step': max(self.events_per_step) if self.events_per_step else 0,
            'event_frequency': np.mean(self.events_per_step) if self.events_per_step else 0,
            'event_distribution': dict(self.event_distribution)
        }


class GenericSymbolicEnv(gym.Env):
    """
    Ambiente completamente generico con reward agnostico.
    Funziona con QUALSIASI simulazione fisica senza conoscenza del dominio.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_object, action_map: Dict[int, Callable], 
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 survival_reward: float = 0.01,
                 event_base_reward: float = 1.0,
                 chain_exponent: float = 1.3):
        """
        Args:
            sim_object: Oggetto simulazione (es. Game())
            action_map: Dizionario {action_id: lambda sim: ...}
            observation_extractor: Funzione per estrarre osservazione
            termination_check: Funzione per controllare terminazione
            survival_reward: Bonus per ogni step vivo
            event_base_reward: Reward base per evento
            chain_exponent: Esponente per catene di eventi
        """
        super().__init__()
        self.sim = sim_object
        self.action_map = action_map
        self.observation_extractor = observation_extractor or self._default_obs_extractor
        self.termination_check = termination_check or (lambda sim: False)
        
        # Configurazione spazio azioni/osservazioni
        self.action_space = gym.spaces.Discrete(len(action_map))
        obs_sample = self.observation_extractor(self.sim)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_sample.shape, dtype=np.float32
        )
        
        # Componenti AGNOSTICI
        self.state_extractor = GenericStateExtractor(sim_object)
        self.event_detector = AgnosticEventDetector(
            threshold_absolute=0.1,
            threshold_relative=0.05
        )
        self.event_tracker = EventTracker()
        self.reward_calculator = AgnosticRewardCalculator(
            survival_reward_per_step=survival_reward,
            event_reward_base=event_base_reward,
            chain_exponent=chain_exponent
        )
        
        self._prev_state = None
        self.done = False
    
    def reset(self):
        # Ricrea o resetta simulazione
        if hasattr(self.sim, 'reset'):
            self.sim.reset()
        else:
            self.sim = type(self.sim)()
        
        self.done = False
        self.event_tracker.reset()
        self._prev_state = self.state_extractor.extract(self.sim)
        
        return self.observation_extractor(self.sim)
    
    def step(self, action: int):
        # Cattura stato precedente
        prev_state = self._prev_state
        
        # Esegui azione
        if action in self.action_map:
            self.action_map[action](self.sim)
        
        # Aggiorna simulazione
        if hasattr(self.sim, 'update'):
            self.sim.update()
        
        # Cattura nuovo stato
        current_state = self.state_extractor.extract(self.sim)
        
        # Rileva eventi (SENZA priorità o classificazione)
        detected_events = self.event_detector.detect_events(prev_state, current_state)
        
        # Traccia eventi (solo conteggio)
        num_events = self.event_tracker.add_events(detected_events)
        
        # Controlla terminazione
        self.done = self.termination_check(self.sim)
        
        # Calcola reward AGNOSTICO
        reward = self.reward_calculator.calculate_step_reward(
            num_events=num_events,
            is_terminal=self.done
        )
        
        # Aggiorna stato
        self._prev_state = current_state
        
        stats = self.event_tracker.get_statistics()
        
        return (
            self.observation_extractor(self.sim),
            reward,
            self.done,
            {
                'events': detected_events,
                'num_events': num_events,
                'step_reward': reward,
                **stats
            }
        )
    
    def _default_obs_extractor(self, sim):
        """Estrae osservazione di default."""
        state = self.state_extractor.extract(sim)
        values = [v for v in state.values() if isinstance(v, (int, float))]
        return np.array(values[:10], dtype=np.float32)


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


def create_arkanoid_env():
    """Factory per creare l'ambiente Arkanoid con configurazione agnostica."""
    
    # Mappa azioni generiche
    action_map = {
        0: lambda game: game.set_paddle_speed(-1),  # Sinistra
        1: lambda game: game.set_paddle_speed(0),   # Fermo
        2: lambda game: game.set_paddle_speed(1),   # Destra
    }
    
    # Estrazione osservazione personalizzata
    def obs_extractor(game):
        ball_x = game.ball_x / grid_width
        ball_y = game.ball_y / grid_height
        vx = game.ball_speed_x / 10.0
        vy = game.ball_speed_y / 10.0
        paddle_x = game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)
    
    # Controllo terminazione
    def termination_check(game):
        return game.ball_lost or game.bricks_alive == 0
    
    return GenericSymbolicEnv(
        sim_object=Game(),
        action_map=action_map,
        observation_extractor=obs_extractor,
        termination_check=termination_check,
        survival_reward=0.01,      # Piccolo bonus per sopravvivere
        event_base_reward=1.0,     # Reward base per evento
        chain_exponent=1.3         # Crescita moderata per catene
    )


def train_generic_dqn(env_factory: Callable, total_episodes=1000, max_steps=2000):
    """
    Training DQN con reward completamente agnostico.
    """
    env = env_factory()
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
    stats_history = []

    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_net(s_t)
                    action = int(q_vals.argmax(1).item())

            next_state, reward, done, info = env.step(action)
            
            buffer.append((state, action, reward, next_state, done))

            # Training
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

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if ep % 10 == 0:
            q_target.load_state_dict(q_net.state_dict())
        
        rewards_history.append(total_reward)
        stats = env.event_tracker.get_statistics()
        stats_history.append(stats)
        
        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_events = np.mean([s['total_events'] for s in stats_history[-10:]])
            avg_freq = np.mean([s['event_frequency'] for s in stats_history[-10:]])
            max_events = max([s['max_events_in_step'] for s in stats_history[-10:]])
            
            print(f"[Ep {ep}] R: {total_reward:.2f} | Avg R: {avg_reward:.2f} | "
                  f"Events: {stats['total_events']} | Freq: {avg_freq:.2f} | "
                  f"Max: {max_events}")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "generic_2.pth")
    torch.save(q_net.state_dict(), model_path)
    
    print(f"\n✅ Training completo! Modello: {model_path}")
    print(f"📊 Reward medio: {np.mean(rewards_history):.2f}")
    
    # Statistiche eventi
    final_stats = stats_history[-1]
    print(f"\n📈 Statistiche eventi:")
    print(f"   Eventi totali: {final_stats['total_events']}")
    print(f"   Step totali: {final_stats['total_steps']}")
    print(f"   Frequenza media: {final_stats['event_frequency']:.2f} eventi/step")
    print(f"   Max eventi/step: {final_stats['max_events_in_step']}")
    print(f"   Distribuzione: {final_stats['event_distribution']}")
    
    return rewards_history, stats_history


if __name__ == "__main__":
    print("🚀 Training DQN con Reward COMPLETAMENTE Agnostico")
    print("=" * 70)
    print("PRINCIPIO: Massimizza eventi + sopravvivenza")
    print("           ZERO conoscenza sulla natura degli eventi")
    print("=" * 70)
    
    rewards, stats = train_generic_dqn(
        env_factory=create_arkanoid_env,
        total_episodes=5000,
        max_steps=5000
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward max: {max(rewards):.2f}")
    print(f"   Performance ultimi 100: {np.mean(rewards[-100:]):.2f}")