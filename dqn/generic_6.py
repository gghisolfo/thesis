import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
import copy
from collections import deque, defaultdict
from typing import Any, Dict, List, Tuple, Callable

# Import locali 
from arkanoid_game import Game, grid_width, grid_height


# python -m dqn.generic_6


SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)
PRINT_MODE = False


class StateVisitationTracker:
    """
    Traccia la frequenza di visita degli stati per incentivare l'esplorazione
    di stati rari e penalizzare stati che portano rapidamente alla morte.
    """
    def __init__(self, 
                 discretization_bins: int = 10,
                 rare_bonus_scale: float = 2.0,
                 death_memory_size: int = 100,
                 death_penalty: float = -5.0,
                 death_proximity_threshold: float = 0.3):
        """
        Args:
            discretization_bins: Numero di bin per discretizzare ogni dimensione
            rare_bonus_scale: Scala del bonus per stati rari (reward = scale / sqrt(visit_count))
            death_memory_size: Quanti stati pre-morte ricordare
            death_penalty: Penalità per stati simili a quelli che hanno portato a morte
            death_proximity_threshold: Soglia di distanza per considerare uno stato "vicino" a morte
        """
        self.bins = discretization_bins
        self.rare_bonus_scale = rare_bonus_scale
        self.death_penalty = death_penalty
        self.death_threshold = death_proximity_threshold
        
        # Contatori
        self.visit_counts = defaultdict(int)  # state_hash -> count
        self.death_states = deque(maxlen=death_memory_size)  # Lista di stati pre-morte
        
        # Statistiche
        self.total_visits = 0
        self.unique_states = 0
        self.death_penalties_given = 0
        self.rare_bonuses_given = 0
    
    def _discretize_state(self, state_vector: np.ndarray) -> tuple:
        """
        Discretizza uno stato continuo in bin per creare un hash.
        Usa percentili invece di range fissi per robustezza.
        """
        # Normalizza in [0, 1] usando tanh (gestisce meglio outliers)
        normalized = (np.tanh(state_vector) + 1) / 2
        # Discretizza in bin
        discretized = (normalized * self.bins).astype(int)
        # Clamp per sicurezza
        discretized = np.clip(discretized, 0, self.bins - 1)
        return tuple(discretized)
    
    def _state_distance(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calcola distanza normalizzata tra due stati."""
        return np.linalg.norm(state1 - state2) / (np.sqrt(len(state1)) + 1e-9)
    
    def record_visit(self, state_vector: np.ndarray) -> float:
        """
        Registra la visita di uno stato e calcola il rare state bonus.
        
        Returns:
            rare_bonus: Bonus positivo per stati rari, 0 per stati comuni
        """
        state_hash = self._discretize_state(state_vector)
        self.visit_counts[state_hash] += 1
        self.total_visits += 1
        
        if self.visit_counts[state_hash] == 1:
            self.unique_states += 1
        
        # Calcola bonus: inversamente proporzionale a sqrt(visit_count)
        # Stati mai visti: bonus massimo
        # Stati visti molte volte: bonus minimo
        visit_count = self.visit_counts[state_hash]
        rare_bonus = self.rare_bonus_scale / np.sqrt(visit_count)
        
        if rare_bonus > 0.5:  # Soglia arbitraria per "raro"
            self.rare_bonuses_given += 1
        
        return rare_bonus
    
    def check_death_proximity(self, state_vector: np.ndarray) -> float:
        """
        Controlla se lo stato è vicino a stati che hanno portato a morte.
        
        Returns:
            penalty: Penalità negativa se vicino a morte, 0 altrimenti
        """
        if not self.death_states:
            return 0.0
        
        # Calcola distanza minima da tutti gli stati di morte
        min_distance = min(
            self._state_distance(state_vector, death_state)
            for death_state in self.death_states
        )
        
        # Se troppo vicino, applica penalità
        if min_distance < self.death_threshold:
            # Penalità proporzionale alla vicinanza
            proximity_factor = 1.0 - (min_distance / self.death_threshold)
            penalty = self.death_penalty * proximity_factor
            self.death_penalties_given += 1
            return penalty
        
        return 0.0
    
    def record_death(self, state_vector: np.ndarray):
        """Registra uno stato che ha portato a morte del gioco."""
        self.death_states.append(state_vector.copy())
    
    def get_statistics(self) -> Dict:
        """Ritorna statistiche sull'esplorazione."""
        exploration_ratio = self.unique_states / max(self.total_visits, 1)
        
        return {
            'total_visits': self.total_visits,
            'unique_states': self.unique_states,
            'exploration_ratio': exploration_ratio,
            'death_states_recorded': len(self.death_states),
            'death_penalties_given': self.death_penalties_given,
            'rare_bonuses_given': self.rare_bonuses_given,
            'avg_revisit_rate': self.total_visits / max(self.unique_states, 1)
        }
    
    def reset_episode(self):
        """Reset per nuovo episodio (mantiene memoria tra episodi)."""
        # Non resettiamo visit_counts né death_states - vogliamo memoria persistente!
        pass


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

    def to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Converte un dizionario di stato in vettore numpy numerico."""
        vec = []
        for k in self.tracked_attributes:
            v = state_dict.get(k, 0.0)
            # Normalizza booleans a 0/1
            if isinstance(v, bool):
                vec.append(1.0 if v else 0.0)
            elif v is None:
                vec.append(0.0)
            else:
                try:
                    vec.append(float(v))
                except:
                    vec.append(0.0)
        return np.array(vec, dtype=np.float32)


class GenericEventDetector:
    """
    Rileva automaticamente QUALSIASI cambiamento di stato come un evento.
    Non richiede conoscenza del dominio.
    """
    def __init__(self, 
                 threshold_for_change: float = 0.3,
                 min_relative_change: float = 0.03):
        self.threshold = threshold_for_change
        self.min_relative_change = min_relative_change

    def detect_events(self, prev_state: Dict, curr_state: Dict) -> List[Dict]:
        """Rileva TUTTI i cambiamenti significativi."""
        events = []
        
        for key in prev_state.keys():
            if key not in curr_state:
                continue
            
            prev_val = prev_state[key]
            curr_val = curr_state[key]
            
            if prev_val is None or curr_val is None:
                continue
            
            event = self._detect_change(key, prev_val, curr_val)
            if event:
                events.append(event)
        
        return events
    
    def _detect_change(self, key: str, prev: Any, curr: Any) -> Dict:
        """Rileva cambiamento generico senza conoscere la natura."""
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)):
            delta = abs(curr - prev)
            
            if delta < self.threshold:
                return None
            
            relative = delta / (abs(prev) + 1e-9)
            if relative < self.min_relative_change:
                return None
            
            return {
                'attribute': key,
                'prev': prev,
                'curr': curr,
                'delta': delta,
                'timestamp': None
            }
        
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


class ForwardModel(nn.Module):
    """Forward model per curiosity-driven learning."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class CausalEventChainTracker:
    """Traccia catene causali di eventi."""
    def __init__(self, 
                 causal_window: int = 3,
                 base_reward: float = 1.0,
                 chain_exponent: float = 1.2):
        self.causal_window = causal_window
        self.base_reward = base_reward
        self.chain_exponent = chain_exponent
        self.reset()
    
    def reset(self):
        self.event_history = []
        self.step_counter = 0
        self.total_reward = 0.0
        self.chain_stats = defaultdict(int)
    
    def add_events(self, events: List[Dict]):
        if not events:
            self.step_counter += 1
            return
        
        for event in events:
            event['timestamp'] = self.step_counter
            self.event_history.append(event)
        
        chain_length = len(events)
        chain_reward = self._calculate_chain_reward(chain_length)
        
        self.total_reward += chain_reward
        self.chain_stats[chain_length] += 1
        
        if chain_length >= 2 and PRINT_MODE:
            print(f"⛓️ Catena lunga: {chain_length} eventi → Reward: {chain_reward:.2f}")
            for e in events:
                print(f"  - {e['attribute']} : {e['prev']} -> {e['curr']}")
        
        self.step_counter += 1
    
    def _calculate_chain_reward(self, chain_length: int) -> float:
        if chain_length == 0:
            return 0.0
        return self.base_reward * (chain_length ** self.chain_exponent)
    
    def get_total_reward(self) -> float:
        return self.total_reward
    
    def get_statistics(self) -> Dict:
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
    Ambiente completamente generico con supporto per rare state exploration.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_object, action_map: Dict[int, Callable], 
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 causal_window: int = 3,
                 chain_exponent: float = 1.5,
                 w_causal: float = 1.0,
                 w_curiosity: float = 0.5,
                 w_density: float = 0.5,
                 w_rare: float = 1.0,  # NUOVO: peso per rare state bonus
                 forward_lr: float = 1e-3,
                 rare_state_config: Dict = None):  # NUOVO: config per rare states
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
        self.event_detector = GenericEventDetector(
            threshold_for_change=0.3,
            min_relative_change=0.03
        )
        self.event_tracker = CausalEventChainTracker(
            causal_window=causal_window,
            chain_exponent=chain_exponent
        )
        
        # NUOVO: Tracker per stati rari
        rare_config = rare_state_config or {}
        self.state_tracker = StateVisitationTracker(
            discretization_bins=rare_config.get('bins', 10),
            rare_bonus_scale=rare_config.get('rare_bonus_scale', 2.0),
            death_memory_size=rare_config.get('death_memory', 100),
            death_penalty=rare_config.get('death_penalty', -5.0),
            death_proximity_threshold=rare_config.get('death_threshold', 0.3)
        )
        
        # Pesi componenti reward
        self.w_causal = w_causal
        self.w_curiosity = w_curiosity
        self.w_density = w_density
        self.w_rare = w_rare  # NUOVO

        # Forward model (curiosity)
        self._forward_model = None
        self._forward_opt = None
        self._forward_lr = forward_lr

        self._prev_state = None
        self._prev_state_vec = None
        self.done = False

        self._maybe_init_forward_model()
    
    def _maybe_init_forward_model(self):
        state_dim = len(self.state_extractor.tracked_attributes)
        action_dim = len(self.action_map)
        if state_dim > 0 and action_dim > 0 and self._forward_model is None:
            self._forward_model = ForwardModel(state_dim=state_dim, action_dim=action_dim, hidden=128)
            self._forward_opt = optim.Adam(self._forward_model.parameters(), lr=self._forward_lr)

    def reset(self):
        if hasattr(self.sim, 'reset'):
            self.sim.reset()
        else:
            self.sim = type(self.sim)()
        
        self.done = False
        self.event_tracker.reset()
        self.state_tracker.reset_episode()  # NUOVO
        
        self._prev_state = self.state_extractor.extract(self.sim)
        self._prev_state_vec = self.state_extractor.to_vector(self._prev_state)
        self._maybe_init_forward_model()
        
        return self.observation_extractor(self.sim)
    
    def step(self, action: int):
        # Cattura stato precedente
        prev_state = self._prev_state
        prev_state_vec = self._prev_state_vec.copy() if self._prev_state_vec is not None else None
        
        # Esegui azione
        if action in self.action_map:
            self.action_map[action](self.sim)
        
        if hasattr(self.sim, 'update'):
            self.sim.update()
        
        # Cattura nuovo stato
        current_state = self.state_extractor.extract(self.sim)
        current_state_vec = self.state_extractor.to_vector(current_state)
        
        # Rileva eventi
        detected_events = self.event_detector.detect_events(prev_state, current_state)
        self.event_tracker.add_events(detected_events)
        
        # === REWARD COMPONENTS ===
        
        # 1) Eventi (catene causali)
        event_reward = len(detected_events) ** self.event_tracker.chain_exponent if detected_events else 0.0

        # 2) Impatto causale (no-op baseline)
        causal_bonus = 0.0
        try:
            sim_copy = copy.deepcopy(self.sim) if not hasattr(self.sim, 'clone') else self.sim.clone()
            if hasattr(sim_copy, 'update'):
                sim_copy.update()
            noop_state = self.state_extractor.extract(sim_copy)
            noop_vec = self.state_extractor.to_vector(noop_state)
            causal_impact = float(np.linalg.norm(current_state_vec - noop_vec, ord=1))
            causal_bonus = self.w_causal * causal_impact
        except Exception as e:
            if PRINT_MODE:
                print("Causal baseline failed:", e)

        # 3) Curiosity (forward model)
        curiosity_bonus = 0.0
        if self._forward_model is not None and prev_state_vec is not None:
            try:
                state_t = torch.tensor(prev_state_vec, dtype=torch.float32).unsqueeze(0)
                a_onehot = np.zeros(len(self.action_map), dtype=np.float32)
                a_onehot[action] = 1.0
                act_t = torch.tensor(a_onehot, dtype=torch.float32).unsqueeze(0)
                inp = torch.cat([state_t, act_t], dim=1)
                
                self._forward_model.train()
                pred = self._forward_model(inp)
                target = torch.tensor(current_state_vec, dtype=torch.float32).unsqueeze(0)
                loss = nn.functional.mse_loss(pred, target, reduction='none').mean(1)
                err = float(loss.item())
                
                self._forward_opt.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self._forward_model.parameters(), 1.0)
                self._forward_opt.step()
                
                curiosity_bonus = self.w_curiosity * err
            except Exception as e:
                if PRINT_MODE:
                    print("Curiosity failed:", e)

        # 4) Event density
        density_bonus = 0.0
        try:
            density = float(np.sum(np.abs(current_state_vec - prev_state_vec)))
            density_bonus = self.w_density * density
        except Exception:
            pass

        # === NUOVO: 5) RARE STATE EXPLORATION ===
        rare_state_bonus = self.state_tracker.record_visit(current_state_vec) * self.w_rare
        death_proximity_penalty = self.state_tracker.check_death_proximity(current_state_vec)
        
        # 6) Reward shaping (opzionale)
        shaping_reward = 0.0
        if hasattr(self.sim, 'ball_y') and hasattr(self.sim, 'paddle_x'):
            ball_x = getattr(self.sim, 'ball_x', 0)
            paddle_x = getattr(self.sim, 'paddle_x', 0)
            distance = abs(ball_x - paddle_x)
            max_distance = getattr(self.sim, 'grid_width', 100)
            proximity_bonus = 0.1 * (1.0 - min(distance / max_distance, 1.0))
            shaping_reward = proximity_bonus
        
        # === REWARD TOTALE ===
        reward = (event_reward + 
                  shaping_reward * 0.1 + 
                  causal_bonus + 
                  curiosity_bonus + 
                  density_bonus +
                  rare_state_bonus +  # NUOVO
                  death_proximity_penalty)  # NUOVO
        
        # Debug printing
        if PRINT_MODE and (rare_state_bonus > 0.5 or death_proximity_penalty < -1.0):
            print(f"🔍 RARE STATE: bonus={rare_state_bonus:.2f} | death_penalty={death_proximity_penalty:.2f}")
        
        # Aggiorna stato precedente
        self._prev_state = current_state
        self._prev_state_vec = current_state_vec.copy()
        
        # Info per debug
        extra_info = {
            'events': detected_events,
            'chain_length': len(detected_events),
            'event_reward': event_reward,
            'shaping_reward': shaping_reward,
            'causal_bonus': causal_bonus,
            'curiosity_bonus': curiosity_bonus,
            'density_bonus': density_bonus,
            'rare_state_bonus': rare_state_bonus,  # NUOVO
            'death_proximity_penalty': death_proximity_penalty,  # NUOVO
            'step_reward': reward,
            **self.event_tracker.get_statistics(),
            **self.state_tracker.get_statistics()  # NUOVO
        }
        
        # Controlla terminazione
        self.done = self.termination_check(self.sim)
        
        # NUOVO: Se il gioco è finito per morte, registra lo stato pre-morte
        if self.done and hasattr(self.sim, 'ball_lost') and self.sim.ball_lost:
            self.state_tracker.record_death(current_state_vec)
            if PRINT_MODE:
                print(f"💀 Morte registrata! Stati di morte totali: {len(self.state_tracker.death_states)}")
        
        return (
            self.observation_extractor(self.sim),
            reward,
            self.done,
            extra_info
        )
    
    def _default_obs_extractor(self, sim):
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
    """Factory per creare l'ambiente Arkanoid con rare state exploration."""
    
    action_map = {
        0: lambda game: game.set_paddle_speed(-1),
        1: lambda game: game.set_paddle_speed(0),
        2: lambda game: game.set_paddle_speed(1),
    }
    
    def obs_extractor(game):
        ball_x = game.ball_x / grid_width
        ball_y = game.ball_y / grid_height
        vx = game.ball_speed_x / 10.0
        vy = game.ball_speed_y / 10.0
        paddle_x = game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)
    
    def termination_check(game):
        return game.ball_lost or game.bricks_alive == 0
    
    # NUOVO: Configurazione rare state exploration
    rare_config = {
        'bins': 15,  # Più granularità per distinguere meglio gli stati
        'rare_bonus_scale': 3.0,  # Bonus significativo per stati rari
        'death_memory': 200,  # Ricorda molti stati di morte
        'death_penalty': -8.0,  # Penalità forte per avvicinarsi a morte
        'death_threshold': 0.25  # Distanza sotto cui scatta la penalità
    }
    
    return GenericSymbolicEnv(
        sim_object=Game(),
        action_map=action_map,
        observation_extractor=obs_extractor,
        termination_check=termination_check,
        causal_window=3,
        chain_exponent=1.2,
        w_causal=0.8,
        w_curiosity=0.5,
        w_density=0.3,
        w_rare=1.5,  # NUOVO: Peso alto per incentivare esplorazione
        forward_lr=1e-3,
        rare_state_config=rare_config  # NUOVO
    )


def train_generic_dqn(env_factory: Callable, total_episodes=10000, max_steps=3000):
    """Training DQN con rare state exploration."""
    env = env_factory()
    buffer = deque(maxlen=100000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=5e-5)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995
    
    # Metriche
    rewards_history = []
    chain_stats_history = []
    survival_times = []
    exploration_stats = []  # NUOVO
    best_survival = 0

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
            if len(buffer) >= 128:
                batch = random.sample(buffer, 128)
                s, a, r, ns, d = zip(*batch)
                
                s_t = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device)
                ns_t = torch.tensor(np.array(ns), dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)


                with torch.no_grad():
                    max_next_q = q_target(ns_t).max(1)[0]
                    target_q = r_t + gamma * (1 - d_t) * max_next_q

                current_q = q_net(s_t).gather(1, a_t).squeeze(1)
                loss = nn.functional.mse_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

            total_reward += reward
            state = next_state
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network
        if ep < 1000 and ep % 5 == 0:
            q_target.load_state_dict(q_net.state_dict())
        elif ep % 20 == 0:
            q_target.load_state_dict(q_net.state_dict())
        
        # Traccia statistiche
        survival_time = steps / 60.0
        survival_times.append(survival_time)
        if survival_time > best_survival:
            best_survival = survival_time
            if PRINT_MODE:
                print(f"🏆 Nuovo record! Sopravvissuto {best_survival:.1f}s (ep {ep})")
        
        rewards_history.append(total_reward)
        chain_stats_history.append(env.event_tracker.get_statistics())
        exploration_stats.append(env.state_tracker.get_statistics())  # NUOVO
        
        # Logging dettagliato
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if rewards_history else 0.0
            avg_survival = np.mean(survival_times[-50:]) if survival_times else 0.0
            
            # NUOVO: Statistiche esplorazione
            recent_exploration = exploration_stats[-50:] if len(exploration_stats) >= 50 else exploration_stats
            avg_unique_states = np.mean([s['unique_states'] for s in recent_exploration]) if recent_exploration else 0
            avg_exploration_ratio = np.mean([s['exploration_ratio'] for s in recent_exploration]) if recent_exploration else 0
            total_death_states = exploration_stats[-1]['death_states_recorded'] if exploration_stats else 0
            
            print(f"[Ep {ep:5d}] ε={epsilon:.3f} | R: {total_reward:6.1f} (avg {avg_reward:6.1f}) | "
                  f"Survived: {survival_time:4.1f}s (avg {avg_survival:4.1f}s) | "
                  f"Explored: {avg_unique_states:.0f} states ({avg_exploration_ratio:.2%}) | "
                  f"Deaths: {total_death_states}")
        
        # Milestone checks
        if ep == 1000:
            stats = exploration_stats[-1]
            print(f"\n📊 Milestone 1000 episodi:")
            print(f"   Sopravvivenza media: {np.mean(survival_times[-100:]):.1f}s")
            print(f"   Stati unici scoperti: {stats['unique_states']}")
            print(f"   Esplorazione ratio: {stats['exploration_ratio']:.2%}")
            print(f"   Stati di morte memorizzati: {stats['death_states_recorded']}")
        
        if ep == 5000:
            avg_surv = np.mean(survival_times[-100:])
            stats = exploration_stats[-1]
            print(f"\n📊 Milestone 5000 episodi:")
            print(f"   Sopravvivenza media: {avg_surv:.1f}s")
            print(f"   Stati unici scoperti: {stats['unique_states']}")
            print(f"   Tasso di rivisitazione: {stats['avg_revisit_rate']:.1f}x")
            if avg_surv > 60.0:
                print("🎉 Obiettivo 1 minuto raggiunto!")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "generic_6.pth")
    torch.save(q_net.state_dict(), model_path)
    
    print(f"\n✅ Training completo! Modello: {model_path}")
    print(f"📊 Reward medio finale: {np.mean(rewards_history[-100:]):.2f}")
    print(f"⏱️  Sopravvivenza media finale: {np.mean(survival_times[-100:]):.1f}s")
    print(f"🏆 Record sopravvivenza: {best_survival:.1f}s")
    
    # NUOVO: Statistiche esplorazione finali
    final_exploration = exploration_stats[-1]
    print(f"\n🔍 Statistiche esplorazione:")
    print(f"   Visite totali: {final_exploration['total_visits']}")
    print(f"   Stati unici: {final_exploration['unique_states']}")
    print(f"   Exploration ratio: {final_exploration['exploration_ratio']:.2%}")
    print(f"   Stati di morte: {final_exploration['death_states_recorded']}")
    print(f"   Penalità morte date: {final_exploration['death_penalties_given']}")
    print(f"   Bonus rari dati: {final_exploration['rare_bonuses_given']}")
    
    return rewards_history, chain_stats_history, survival_times, exploration_stats


if __name__ == "__main__":
    total_episodes = 5000

    rewards, stats, survival, exploration = train_generic_dqn(
        env_factory=create_arkanoid_env,
        total_episodes=total_episodes,
        max_steps=3600
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward max: {max(rewards):.2f}")
    print(f"   Performance ultimi 100: {np.mean(rewards[-100:]):.2f}")
    print(f"   Sopravvivenza media ultimi 100: {np.mean(survival[-100:]):.1f}s")
    print(f"   Record assoluto: {max(survival):.1f}s")
    print(f"   Stati unici esplorati: {exploration[-1]['unique_states']}")