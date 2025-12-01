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

# python -m dqn.dqn_sim_env_generic 

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


class GenericEventDetector:
    """
    Rileva automaticamente QUALSIASI cambiamento di stato come un evento.
    Non richiede conoscenza del dominio.
    """
    def __init__(self, threshold_for_change: float = 1e-6):
        self.threshold = threshold_for_change
        self.event_types = {
            'value_change': self._detect_value_change,
            'sign_flip': self._detect_sign_flip,
            'threshold_cross': self._detect_threshold_cross,
            'discrete_change': self._detect_discrete_change,
        }
    
    def detect_events(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        """
        Rileva tutti gli eventi confrontando due stati.
        Ritorna una lista di eventi rilevati.
        """
        events = []
        
        for key in prev_state.keys():
            if key not in curr_state:
                continue
            
            prev_val = prev_state[key]
            curr_val = curr_state[key]
            
            # Salta se uno dei due è None
            if prev_val is None or curr_val is None:
                continue
            
            # Applica tutti i rilevatori di eventi
            for event_type, detector in self.event_types.items():
                detected = detector(key, prev_val, curr_val)
                if detected:
                    events.append(detected)
        
        return events
    
    def _detect_value_change(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva qualsiasi cambiamento significativo di valore."""
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            delta = abs(curr - prev)
            if delta > self.threshold:
                return {
                    'type': 'value_change',
                    'attribute': key,
                    'delta': delta,
                    'prev': prev,
                    'curr': curr,
                    'magnitude': delta / (abs(prev) + 1e-9)  # Cambiamento relativo
                }
        return None
    
    def _detect_sign_flip(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva inversioni di segno (indica collisioni, rimbalzi, etc.)."""
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            if prev * curr < 0:  # Segni opposti
                return {
                    'type': 'sign_flip',
                    'attribute': key,
                    'prev': prev,
                    'curr': curr,
                    'importance': 'high'  # I sign flip sono tipicamente eventi fisici importanti
                }
        return None
    
    def _detect_threshold_cross(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva attraversamenti di soglie specifiche (0, min, max)."""
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            # Attraversamento dello zero
            if (prev < 0 < curr) or (prev > 0 > curr):
                return {
                    'type': 'zero_crossing',
                    'attribute': key,
                    'prev': prev,
                    'curr': curr
                }
        return None
    
    def _detect_discrete_change(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva cambiamenti in variabili discrete (contatori, flag)."""
        if isinstance(prev, (int, bool)) and isinstance(curr, (int, bool)):
            if prev != curr:
                return {
                    'type': 'discrete_change',
                    'attribute': key,
                    'prev': prev,
                    'curr': curr,
                    'direction': 'increase' if curr > prev else 'decrease'
                }
        return None


class EventChainTracker:
    """
    Traccia catene di eventi con pesi configurabili.
    """
    def __init__(self, event_weights: Dict[str, float] = None):
        self.event_weights = event_weights or {
            'sign_flip': 2.0,           # Eventi fisici importanti
            'discrete_change': 3.0,     # Cambiamenti di stato (es. oggetti distrutti)
            'value_change': 0.5,        # Cambiamenti generici
            'zero_crossing': 1.5,       # Attraversamenti di confini
        }
        self.reset()
    
    def reset(self):
        self.events = []
        self.chain_length = 0
        self.weighted_reward = 0.0
    
    def add_events(self, events: List[Dict]):
        """Aggiunge una lista di eventi rilevati."""
        for event in events:
            if event:  # Salta eventi None
                self.events.append(event)
                self.chain_length += 1
                
                # Calcola peso dell'evento
                event_type = event.get('type', 'value_change')
                weight = self.event_weights.get(event_type, 1.0)
                
                # Aggiungi bonus per importanza
                if event.get('importance') == 'high':
                    weight *= 2.0
                
                self.weighted_reward += weight
    
    def get_chain_reward(self) -> float:
        """
        Calcola reward basato su:
        1. Somma dei pesi degli eventi
        2. Bonus esponenziale per catene lunghe
        3. Bonus per diversità di eventi
        """
        if self.chain_length == 0:
            return 0.0
        
        # Reward pesato base
        base_reward = self.weighted_reward
        
        # Bonus esponenziale per catene lunghe
        chain_bonus = 0.0
        if self.chain_length >= 2:
            chain_bonus = (self.chain_length - 1) ** 1.3
        
        # Bonus per diversità (tipi di eventi diversi)
        unique_types = len(set(e.get('type') for e in self.events))
        diversity_bonus = unique_types * 0.5
        
        return base_reward + chain_bonus + diversity_bonus


class GenericSymbolicEnv(gym.Env):
    """
    Ambiente completamente generico che funziona con QUALSIASI simulazione fisica.
    Estrae automaticamente lo stato e rileva eventi senza conoscenza del dominio.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_object, action_map: Dict[int, Callable], 
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 event_weights: Dict[str, float] = None):
        """
        Args:
            sim_object: Oggetto simulazione (es. Game())
            action_map: Dizionario {action_id: lambda sim: ...} per eseguire azioni
            observation_extractor: Funzione per estrarre osservazione (opzionale)
            termination_check: Funzione per controllare terminazione (opzionale)
            event_weights: Pesi personalizzati per tipi di eventi
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
        
        # Estrazione stato e rilevamento eventi
        self.state_extractor = GenericStateExtractor(sim_object)
        self.event_detector = GenericEventDetector(threshold_for_change=0.1)
        self.event_tracker = EventChainTracker(event_weights)
        
        self._prev_state = None
        self.done = False
    
    def reset(self):
        # Ricrea simulazione (o resetta se ha metodo reset)
        if hasattr(self.sim, 'reset'):
            self.sim.reset()
        else:
            # Se non ha reset, ricrea l'oggetto
            self.sim = type(self.sim)()
        
        self.done = False
        self.event_tracker.reset()
        self._prev_state = self.state_extractor.extract(self.sim)
        
        return self.observation_extractor(self.sim)
    
    def step(self, action: int):
        # Reset tracker eventi
        self.event_tracker.reset()
        
        # Cattura stato precedente
        prev_state = self._prev_state
        
        # Esegui azione sulla simulazione
        if action in self.action_map:
            self.action_map[action](self.sim)
        
        # Aggiorna simulazione (assume che abbia metodo update)
        if hasattr(self.sim, 'update'):
            self.sim.update()
        
        # Cattura nuovo stato
        current_state = self.state_extractor.extract(self.sim)
        
        # Rileva TUTTI gli eventi automaticamente
        detected_events = self.event_detector.detect_events(prev_state, current_state)
        self.event_tracker.add_events(detected_events)
        
        # Calcola reward dalla catena di eventi
        reward = self.event_tracker.get_chain_reward()
        
        # Aggiorna stato precedente
        self._prev_state = current_state
        
        # Controlla terminazione
        self.done = self.termination_check(self.sim)
        
        # Debug output per catene interessanti
        if self.event_tracker.chain_length > 50: #2
            print("event")
            # print(f"⛓️  {self.event_tracker.chain_length} eventi → R: {reward:.2f}")
            # for i, evt in enumerate(self.event_tracker.events[:5]):  # Primi 5
            #     print(f"   {i+1}. {evt['type']}: {evt.get('attribute', '?')}")
        
        return (
            self.observation_extractor(self.sim),
            reward,
            self.done,
            {
                'events': self.event_tracker.events,
                'chain_length': self.event_tracker.chain_length,
                'weighted_reward': self.event_tracker.weighted_reward
            }
        )
    
    def _default_obs_extractor(self, sim):
        """Estrae osservazione di default (primi N attributi numerici)."""
        state = self.state_extractor.extract(sim)
        values = [v for v in state.values() if isinstance(v, (int, float))]
        return np.array(values[:10], dtype=np.float32)  # Limita a 10 per semplicità


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
    """Factory per creare l'ambiente Arkanoid con configurazione generica."""
    
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
    
    # Controllo terminazione (opzionale, può essere None)
    def termination_check(game):
        return game.ball_lost or game.bricks_alive == 0
    
    # Pesi eventi personalizzati per Arkanoid
    event_weights = {
        'sign_flip': 3.0,        # Collisioni/rimbalzi
        'discrete_change': 5.0,  # Brick distrutti
        'value_change': 0.3,     # Movimenti generici
        'zero_crossing': 2.0,    # Attraversamenti confini
    }
    
    return GenericSymbolicEnv(
        sim_object=Game(),
        action_map=action_map,
        observation_extractor=obs_extractor,
        termination_check=termination_check,
        event_weights=event_weights
    )


def train_generic_dqn(env_factory: Callable, total_episodes=1000, max_steps=2000):
    """
    Training DQN completamente generico.
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
    chain_lengths_history = []

    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        total_chain_events = 0
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
            total_chain_events += info.get('chain_length', 0)
            
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
        chain_lengths_history.append(total_chain_events)
        
        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_chains = np.mean(chain_lengths_history[-10:])
            print(f"[Ep {ep}] R: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eventi: {total_chain_events} | Avg: {avg_chains:.2f}")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "dqn_generic_symbolic.pth")
    torch.save(q_net.state_dict(), model_path)
    
    print(f"\n✅ Training completo! Modello: {model_path}")
    print(f"📊 Reward medio: {np.mean(rewards_history):.2f}")
    print(f"⛓️  Eventi medi: {np.mean(chain_lengths_history):.2f}")
    
    return rewards_history, chain_lengths_history


if __name__ == "__main__":
    print("🚀 Training DQN generico con rilevamento automatico eventi")
    print("=" * 70)
    
    rewards, chains = train_generic_dqn(
        env_factory=create_arkanoid_env,
        total_episodes=1000,
        max_steps=2000
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward max: {max(rewards):.2f}")
    print(f"   Catena max: {max(chains)} eventi")
    print(f"   Performance ultimi 100: {np.mean(rewards[-100:]):.2f}")