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

# python -m dqn.generic_0 

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
    def __init__(self, 
                 threshold_for_change: float = 0.5,  # Aumentato da 1e-6
                 min_relative_change: float = 0.05):  # 5% minimo
        self.threshold = threshold_for_change
        self.min_relative_change = min_relative_change
        self.event_types = {
            'sign_flip': self._detect_sign_flip,           # Prima i più importanti
            'discrete_change': self._detect_discrete_change,
            'threshold_cross': self._detect_threshold_cross,
            'value_change': self._detect_value_change,     # Ultimo (più comune)
        }
        self.last_event_per_attr = {}  # Anti-spam per attributo
    
    def detect_events(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        """
        Rileva tutti gli eventi confrontando due stati.
        Ritorna una lista di eventi rilevati (SENZA duplicati).
        """
        events = []
        seen_attributes = set()  # Previeni eventi multipli per stesso attributo
        
        for key in prev_state.keys():
            if key not in curr_state or key in seen_attributes:
                continue
            
            prev_val = prev_state[key]
            curr_val = curr_state[key]
            
            # Salta se uno dei due è None
            if prev_val is None or curr_val is None:
                continue
            
            # Applica rilevatori in ordine di priorità (ritorna al primo match)
            for event_type, detector in self.event_types.items():
                detected = detector(key, prev_val, curr_val)
                if detected:
                    events.append(detected)
                    seen_attributes.add(key)  # Blocca altri eventi per questo attributo
                    break  # IMPORTANTE: un solo evento per attributo
        
        return events
    
    def _detect_value_change(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva SOLO cambiamenti significativi (assoluti E relativi)."""
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            delta = abs(curr - prev)
            
            # Richiede ENTRAMBI:
            # 1. Soglia assoluta
            if delta < self.threshold:
                return None
            
            # 2. Soglia relativa (5% del valore precedente)
            relative_change = delta / (abs(prev) + 1e-9)
            if relative_change < self.min_relative_change:
                return None
            
            return {
                'type': 'value_change',
                'attribute': key,
                'delta': delta,
                'prev': prev,
                'curr': curr,
                'timestamp': None
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
                    'timestamp': None
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
                    'curr': curr,
                    'timestamp': None
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
                    'direction': 'increase' if curr > prev else 'decrease',
                    'timestamp': None
                }
        return None


class CausalEventChainTracker:
    """
    Traccia catene causali di eventi.
    
    PRINCIPIO: Tutti gli eventi hanno peso uguale (1.0), ma eventi che 
    scatenano altri eventi ricevono reward esponenziale in base alla 
    lunghezza della catena causale che generano.
    
    Esempio Arkanoid:
    - Rimbalzo muro (1 evento) → reward = 1.0
    - Rimbalzo + cambio velocità (2 eventi simultanei) → reward = 2^1.2 ≈ 2.3
    - Rimbalzo + velocità + brick distrutto (3 eventi) → reward = 3^1.2 ≈ 3.7
    """
    
    def __init__(self, 
                 causal_window: int = 3,
                 base_reward: float = 1.0,
                 chain_exponent: float = 1.2):  # Ridotto da 1.5 a 1.2
        """
        Args:
            causal_window: Numero di step in cui eventi sono considerati causali
            base_reward: Reward base per ogni evento singolo
            chain_exponent: Esponente per reward catena (1.2 = crescita moderata)
        """
        self.causal_window = causal_window
        self.base_reward = base_reward
        self.chain_exponent = chain_exponent
        self.reset()
    
    def reset(self):
        """Reset completo del tracker."""
        self.event_history = []  # Lista di (timestamp, evento)
        self.step_counter = 0
        self.total_reward = 0.0
        self.chain_stats = defaultdict(int)  # Statistiche per lunghezza catena
    
    def add_events(self, events: List[Dict]):
        """
        Aggiunge eventi rilevati in questo step e calcola reward causale.
        """
        if not events:
            self.step_counter += 1
            return
        
        # Assegna timestamp a tutti gli eventi
        for event in events:
            event['timestamp'] = self.step_counter
            self.event_history.append(event)
        
        # Calcola reward per questa catena causale
        chain_length = len(events)
        chain_reward = self._calculate_chain_reward(chain_length)
        
        self.total_reward += chain_reward
        self.chain_stats[chain_length] += 1
        
        # Debug per catene interessanti
        # if chain_length >= 2:
        #     print(f"⛓️  Catena causale: {chain_length} eventi → R: {chain_reward:.2f}")
        #     for evt in events:
        #         print(f"   - {evt['type']}: {evt['attribute']}")
        
        self.step_counter += 1
    
    def _calculate_chain_reward(self, chain_length: int) -> float:
        """
        Calcola reward per una catena causale.
        
        Formula: R = base_reward * (chain_length ^ exponent)
        
        Esempi con exponent=1.5:
        - 1 evento  → 1.0^1.5 = 1.0
        - 2 eventi  → 2.0^1.5 ≈ 2.8  (quasi 3x)
        - 3 eventi  → 3.0^1.5 ≈ 5.2  (5x)
        - 5 eventi  → 5.0^1.5 ≈ 11.2 (11x)
        """
        if chain_length == 0:
            return 0.0
        
        return self.base_reward * (chain_length ** self.chain_exponent)
    
    def get_total_reward(self) -> float:
        """Ritorna il reward totale accumulato."""
        return self.total_reward
    
    def get_statistics(self) -> Dict:
        """Ritorna statistiche sulle catene."""
        total_events = sum(length * count for length, count in self.chain_stats.items())
        total_chains = sum(self.chain_stats.values())
        
        return {
            'total_reward': self.total_reward,
            'total_events': total_events,
            'total_chains': total_chains,
            'avg_chain_length': total_events / max(total_chains, 1),
            'chain_distribution': dict(self.chain_stats),
            'max_chain': max(self.chain_stats.keys()) if self.chain_stats else 0
        }


class GenericSymbolicEnv(gym.Env):
    """
    Ambiente completamente generico che funziona con QUALSIASI simulazione fisica.
    Estrae automaticamente lo stato e rileva eventi senza conoscenza del dominio.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_object, action_map: Dict[int, Callable], 
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 causal_window: int = 3,
                 chain_exponent: float = 1.5):
        """
        Args:
            sim_object: Oggetto simulazione (es. Game())
            action_map: Dizionario {action_id: lambda sim: ...} per eseguire azioni
            observation_extractor: Funzione per estrarre osservazione (opzionale)
            termination_check: Funzione per controllare terminazione (opzionale)
            causal_window: Finestra temporale per eventi causali
            chain_exponent: Esponente per reward catena
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
        self.event_tracker = CausalEventChainTracker(
            causal_window=causal_window,
            chain_exponent=chain_exponent
        )
        
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
        
        # Aggiungi eventi al tracker (calcola reward causale automaticamente)
        self.event_tracker.add_events(detected_events)
        
        # Il reward è calcolato dalla catena causale
        reward = len(detected_events) ** self.event_tracker.chain_exponent if detected_events else 0.0
        
        # Aggiorna stato precedente
        self._prev_state = current_state
        
        # Controlla terminazione
        self.done = self.termination_check(self.sim)
        
        stats = self.event_tracker.get_statistics()
        
        return (
            self.observation_extractor(self.sim),
            reward,
            self.done,
            {
                'events': detected_events,
                'chain_length': len(detected_events),
                'step_reward': reward,
                **stats
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
    
    # Controllo terminazione
    def termination_check(game):
        return game.ball_lost or game.bricks_alive == 0
    
    return GenericSymbolicEnv(
        sim_object=Game(),
        action_map=action_map,
        observation_extractor=obs_extractor,
        termination_check=termination_check,
        causal_window=3,
        chain_exponent=1.5  # Crescita super-lineare per catene
    )


def train_generic_dqn(env_factory: Callable, total_episodes=1000, max_steps=2000):
    """
    Training DQN completamente generico con reward causale.
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
    chain_stats_history = []

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
        chain_stats_history.append(stats)
        
        if ep % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_chains = np.mean([s['total_chains'] for s in chain_stats_history[-10:]])
            avg_chain_len = np.mean([s['avg_chain_length'] for s in chain_stats_history[-10:]])
            max_chain = max([s['max_chain'] for s in chain_stats_history[-10:]])
            
            print(f"[Ep {ep}] R: {total_reward:.2f} | Avg R: {avg_reward:.2f} | "
                  f"Chains: {stats['total_chains']} | Avg len: {avg_chain_len:.2f} | "
                  f"Max: {max_chain}")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "generic_0.pth")
    torch.save(q_net.state_dict(), model_path)
    
    print(f"\n✅ Training completo! Modello: {model_path}")
    print(f"📊 Reward medio: {np.mean(rewards_history):.2f}")
    
    # Statistiche catene
    final_stats = chain_stats_history[-1]
    print(f"\n⛓️  Statistiche catene causali:")
    print(f"   Eventi totali: {final_stats['total_events']}")
    print(f"   Catene totali: {final_stats['total_chains']}")
    print(f"   Lunghezza media: {final_stats['avg_chain_length']:.2f}")
    print(f"   Catena massima: {final_stats['max_chain']}")
    print(f"   Distribuzione: {final_stats['chain_distribution']}")
    
    return rewards_history, chain_stats_history


if __name__ == "__main__":
    print("🚀 Training DQN con Causal Event Chains")
    print("=" * 70)
    print("PRINCIPIO: Tutti gli eventi hanno peso uguale,")
    print("           ma catene causali hanno reward esponenziale")
    print("=" * 70)
    
    rewards, stats = train_generic_dqn(
        env_factory=create_arkanoid_env,
        total_episodes=1000,
        max_steps=2000
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward max: {max(rewards):.2f}")
    print(f"   Performance ultimi 100: {np.mean(rewards[-100:]):.2f}")