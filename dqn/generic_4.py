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


# python -m dqn.generic_4


SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)
PRINT_MODE = False
SHAPING = False
EVENT_DENSITY = False


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
        """Converte un dizionario di stato (basato su tracked_attributes) in vettore numpy numerico."""
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
                 threshold_for_change: float = 0.3,    # Soglia intermedia
                 min_relative_change: float = 0.03):   # 3% minimo
        self.threshold = threshold_for_change
        self.min_relative_change = min_relative_change

    
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
            if delta < self.threshold:
                return None
            
            # Soglia relativa 
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

# ----------------------------
# Forward model (curiosity)
# ----------------------------
class ForwardModel(nn.Module):
    """
    Piccolo modello che predice next_state_vector a partire da (state_vector, action_onehot).
    Allenato online con passo SGD ogni transizione osservata.
    """
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
    """
    Traccia catene causali di eventi.
    
    PRINCIPIO: Tutti gli eventi hanno peso uguale (1.0), ma eventi che 
    scatenano altri eventi ricevono reward esponenziale in base alla 
    lunghezza della catena causale che generano.
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
        # print("chain_length:", chain_length)
        
        chain_reward = self._calculate_chain_reward(chain_length)
        
        self.total_reward += chain_reward
        self.chain_stats[chain_length] += 1
        
        # Debug per catene interessanti
        if chain_length >= 2 and PRINT_MODE:
            print(f"⛓️ Catena lunga: {chain_length} eventi → Reward: {chain_reward:.2f}")
            for e in events:
                print(f"  - {e['attribute']} : {e['prev']} -> {e['curr']}")

        
        self.step_counter += 1
    
    def _calculate_chain_reward(self, chain_length: int) -> float:
        """
        Calcola reward per una catena causale.
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
    Inoltre fornisce reward generico basato su:
      - catene di eventi (già presente)
      - impatto causale (no-op baseline)
      - curiosity (forward model)
      - event density (magnitudo dei cambi nello stato)
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, sim_object, action_map: Dict[int, Callable], 
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 causal_window: int = 3,
                 chain_exponent: float = 1.5,
                 # nuovi iperparametri per i bonus generici
                 w_causal: float = 1.0,
                 w_curiosity: float = 0.5,
                 w_density: float = 0.5,
                 forward_lr: float = 1e-3):
        """
        Args:
            sim_object: Oggetto simulazione (es. Game())
            action_map: Dizionario {action_id: lambda sim: ...} per eseguire azioni
            observation_extractor: Funzione per estrarre osservazione (opzionale)
            termination_check: Funzione per controllare terminazione (opzionale)
            causal_window: Finestra temporale per eventi causali
            chain_exponent: Esponente per reward catena
            w_causal/w_curiosity/w_density: pesi per i componenti aggiuntivi
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
        self.event_detector = GenericEventDetector(
            threshold_for_change=0.3,      # Soglia intermedia (era 1.0, troppo alta)
            min_relative_change=0.03       # 3% minimo (era 0.1, troppo alto)
        )
        self.event_tracker = CausalEventChainTracker(
            causal_window=causal_window,
            chain_exponent=chain_exponent
        )
        
        # Nuove componenti per reward generico
        self.w_causal = w_causal
        self.w_curiosity = w_curiosity
        self.w_density = w_density

        # forward model (curiosity) - inizializzato dopo che conosciamo state_dim
        self._forward_model = None
        self._forward_opt = None
        self._forward_lr = forward_lr

        self._prev_state = None
        self._prev_state_vec = None
        self.done = False

        # se vogliamo inizializzare forward model subito:
        self._maybe_init_forward_model()
    
    def _maybe_init_forward_model(self):
        # inizializza il forward model se possibile (dipende dalla lunghezza degli attributi osservati)
        state_dim = len(self.state_extractor.tracked_attributes)
        action_dim = len(self.action_map)
        if state_dim > 0 and action_dim > 0 and self._forward_model is None:
            self._forward_model = ForwardModel(state_dim=state_dim, action_dim=action_dim, hidden=128)
            self._forward_opt = optim.Adam(self._forward_model.parameters(), lr=self._forward_lr)

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
        self._prev_state_vec = self.state_extractor.to_vector(self._prev_state)
        # eventualmente inizializza forward model (se non fatto)
        self._maybe_init_forward_model()
        
        return self.observation_extractor(self.sim)
    
    def step(self, action: int):
        # Cattura stato precedente
        prev_state = self._prev_state
        prev_state_vec = self._prev_state_vec.copy() if self._prev_state_vec is not None else None
        
        # Esegui azione sulla simulazione
        if action in self.action_map:
            self.action_map[action](self.sim)
        
        # Aggiorna simulazione (assume che abbia metodo update)
        if hasattr(self.sim, 'update'):
            self.sim.update()
        
        # Cattura nuovo stato
        current_state = self.state_extractor.extract(self.sim)
        current_state_vec = self.state_extractor.to_vector(current_state)
        
        # Rileva TUTTI gli eventi automaticamente
        detected_events = self.event_detector.detect_events(prev_state, current_state)
        
        # Aggiungi eventi al tracker (calcola reward causale automaticamente)
        self.event_tracker.add_events(detected_events)
        
        # REWARD: catena eventi (tua logica originale)
        event_reward = len(detected_events) ** self.event_tracker.chain_exponent if detected_events else 0.0

        # ---------------------------
        # 1) IMPATTO CAUSALE (no-op baseline)
        # ---------------------------
        causal_bonus = 0.0
        try:
            # Proviamo a simulare "no-op": copia lo stato, non applicare azione, chiami update() e leggi next state.
            sim_copy = None
            if hasattr(self.sim, 'clone'):  # se Game implementa clone, usalo
                sim_copy = self.sim.clone()
            else:
                # fallback generico: deepcopy (potrebbe essere lento ma è generico)
                sim_copy = copy.deepcopy(self.sim)
            
            # non eseguire la action_map su sim_copy (no-op)
            if hasattr(sim_copy, 'update'):
                sim_copy.update()
            noop_state = self.state_extractor.extract(sim_copy)
            noop_vec = self.state_extractor.to_vector(noop_state)
            # causal impact = differenza fra stato ottenuto con action vs no-op
            causal_impact = float(np.linalg.norm(current_state_vec - noop_vec, ord=1))  # L1
            causal_bonus = self.w_causal * causal_impact
        except Exception as e:
            # se la simulazione non supporta deepcopy o è pesante, ignoriamo causal baseline (0.0)
            if PRINT_MODE:
                print("Causal baseline failed:", e)
            causal_bonus = 0.0

        # ---------------------------
        # 2) CURIOSITY (forward model prediction error)
        # ---------------------------
        curiosity_bonus = 0.0
        if self._forward_model is not None and prev_state_vec is not None:
            try:
                # build input = [prev_state_vec, action_onehot]
                state_t = torch.tensor(prev_state_vec, dtype=torch.float32).unsqueeze(0)
                a_onehot = np.zeros(len(self.action_map), dtype=np.float32)
                a_onehot[action] = 1.0
                act_t = torch.tensor(a_onehot, dtype=torch.float32).unsqueeze(0)
                inp = torch.cat([state_t, act_t], dim=1)
                
                self._forward_model.train()
                pred = self._forward_model(inp)  # predicts next state vector
                target = torch.tensor(current_state_vec, dtype=torch.float32).unsqueeze(0)
                loss = nn.functional.mse_loss(pred, target, reduction='none').mean(1)  # per-batch error
                err = float(loss.item())
                
                # piccolo training step online (1 step)
                self._forward_opt.zero_grad()
                loss.mean().backward()
                # gradient clipping to be safe
                torch.nn.utils.clip_grad_norm_(self._forward_model.parameters(), 1.0)
                self._forward_opt.step()
                
                curiosity_bonus = self.w_curiosity * err
            except Exception as e:
                if PRINT_MODE:
                    print("Curiosity failed:", e)
                curiosity_bonus = 0.0

        # ---------------------------
        # 3) EVENT DENSITY (magnitudo di cambiamento di stato) 
        # ---------------------------
        density_bonus = 0.0
        if EVENT_DENSITY:
            try:
                density = float(np.sum(np.abs(current_state_vec - prev_state_vec)))
                density_bonus = self.w_density * density
            except Exception:
                density_bonus = 0.0


        # REWARD SHAPING (opzionale): piccolo bonus per comportamento base
        shaping_reward = 0.0
        if SHAPING:
            
            if hasattr(self.sim, 'ball_y') and hasattr(self.sim, 'paddle_x'):
                # Piccolo bonus per essere vicino alla palla (solo asse X)
                ball_x = getattr(self.sim, 'ball_x', 0)
                paddle_x = getattr(self.sim, 'paddle_x', 0)
                distance = abs(ball_x - paddle_x)
                
                # Normalizza e scala: max 0.1 quando perfettamente allineato
                max_distance = getattr(self.sim, 'grid_width', 100)
                proximity_bonus = 0.1 * (1.0 - min(distance / max_distance, 1.0))
                shaping_reward = proximity_bonus
            # Reward totale: eventi (dominante) + shaping (minore) + nuovi bonus generici
            reward = event_reward + shaping_reward * 0.1 + causal_bonus + curiosity_bonus + density_bonus
        
        
        # Reward totale: eventi (dominante) + shaping (minore) + nuovi bonus generici
        reward = event_reward + causal_bonus + curiosity_bonus + density_bonus
        
        # (Debug printing)
        if PRINT_MODE:
            print(f"R_comp: events={event_reward:.3f} causal={causal_bonus:.3f} curious={curiosity_bonus:.3f} density={density_bonus:.3f} shaping={shaping_reward*0.1:.3f} => total={reward:.3f}")
        

        # Aggiorna stato precedente
        self._prev_state = current_state
        self._prev_state_vec = current_state_vec.copy()
        
        # Aggiungi info per debug / replay
        extra_info = {
            'events': detected_events,
            'chain_length': len(detected_events),
            'event_reward': event_reward,
            'shaping_reward': shaping_reward,
            'causal_bonus': causal_bonus,
            'curiosity_bonus': curiosity_bonus,
            'density_bonus': density_bonus,
            'step_reward': reward,
            **self.event_tracker.get_statistics()
        }
        
        # Controlla terminazione
        self.done = self.termination_check(self.sim)
        
        return (
            self.observation_extractor(self.sim),
            reward,
            self.done,
            extra_info
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
        chain_exponent=1.2,  # Esponente più basso per reward moderati
        w_causal=0.8,        # peso per causal impact
        w_curiosity=0.5,     # peso per curiosity
        w_density=0.3,       # peso per event density
        forward_lr=1e-3
    )


def train_generic_dqn(env_factory: Callable, total_episodes=10000, max_steps=3000):
    """
    Training DQN completamente generico con reward causale.
    """
    env = env_factory()
    buffer = deque(maxlen=100000)  # Buffer più grande

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=5e-5)  # Learning rate più basso

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05  # Epsilon minimo più alto per continuare esplorazione
    epsilon_decay = 0.9995  # Decay più lento
    
    # Metriche per tracking
    rewards_history = []
    chain_stats_history = []
    survival_times = []  # Nuovo: traccia quanto sopravvive
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

            # for evt in info['events']:
            #     print(f"Evento: {evt['attribute']} | prev: {evt['prev']} -> curr: {evt['curr']}")
            
            buffer.append((state, action, reward, next_state, done))

            # Training - batch più grande e più frequente
            if len(buffer) >= 128:
                batch = random.sample(buffer, 128)  # Batch size aumentato
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
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)  # Gradient clipping
                optimizer.step()

            total_reward += reward
            state = next_state
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network più frequentemente all'inizio
        if ep < 1000 and ep % 5 == 0:
            q_target.load_state_dict(q_net.state_dict())
        elif ep % 20 == 0:
            q_target.load_state_dict(q_net.state_dict())
        
        # Traccia survival time (steps = tempo in vita)
        survival_time = steps / 60.0  # Converti in secondi (60 fps)
        survival_times.append(survival_time)
        if survival_time > best_survival:
            best_survival = survival_time
            if PRINT_MODE:
                print(f"🏆 Nuovo record! Sopravvissuto {best_survival:.1f}s (ep {ep})")
        
        rewards_history.append(total_reward)
        stats = env.event_tracker.get_statistics()
        chain_stats_history.append(stats)
        
        # Logging più dettagliato
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if rewards_history else 0.0
            avg_survival = np.mean(survival_times[-50:]) if survival_times else 0.0
            avg_chains = np.mean([s['total_chains'] for s in chain_stats_history[-50:]]) if chain_stats_history else 0.0
            avg_chain_len = np.mean([s['avg_chain_length'] for s in chain_stats_history[-50:]]) if chain_stats_history else 0.0
            
            print(f"[Ep {ep:5d}] ε={epsilon:.3f} | R: {total_reward:6.1f} (avg {avg_reward:6.1f}) | "
                  f"Survived: {survival_time:4.1f}s (avg {avg_survival:4.1f}s) | "
                  f"Chains: {stats['total_chains']:3d} (len {avg_chain_len:.1f})")
        
        # Milestone checks
        if ep == 1000:
            avg_surv_1k = np.mean(survival_times[-100:]) if len(survival_times) >= 100 else np.mean(survival_times)
            print(f"\n📊 Milestone 1000 episodi: Sopravvivenza media = {avg_surv_1k:.1f}s")
            if avg_surv_1k < 5.0:
                print("⚠️  Agente ancora debole. Potrebbe servire più training.")
        
        if ep == 5000:
            avg_surv_5k = np.mean(survival_times[-100:]) if len(survival_times) >= 100 else np.mean(survival_times)
            print(f"\n📊 Milestone 5000 episodi: Sopravvivenza media = {avg_surv_5k:.1f}s")
            if avg_surv_5k > 60.0:
                print("🎉 Obiettivo 1 minuto raggiunto!")
            else:
                print(f"   Mancano ~{60 - avg_surv_5k:.1f}s per 1 minuto")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "generic_4.pth")
    torch.save(q_net.state_dict(), model_path)
    
    print(f"\n✅ Training completo! Modello: {model_path}")
    print(f"📊 Reward medio finale: {np.mean(rewards_history[-100:]):.2f}")
    print(f"⏱️  Sopravvivenza media finale: {np.mean(survival_times[-100:]):.1f}s")
    print(f"🏆 Record sopravvivenza: {best_survival:.1f}s")
    
    # Statistiche catene
    final_stats = chain_stats_history[-1]
    print(f"\n⛓️  Statistiche catene causali:")
    print(f"   Eventi totali: {final_stats['total_events']}")
    print(f"   Catene totali: {final_stats['total_chains']}")
    print(f"   Lunghezza media: {final_stats['avg_chain_length']:.2f}")
    print(f"   Catena massima: {final_stats['max_chain']}")
    
    return rewards_history, chain_stats_history, survival_times


if __name__ == "__main__":
    total_episodes = 5000 #10000

    rewards, stats, survival = train_generic_dqn(
        env_factory=create_arkanoid_env,
        total_episodes=total_episodes,  # Default per 1 minuto
        max_steps=3600  # 60 secondi * 60 fps
    )
    
    print("\n" + "=" * 70)
    print("📈 Statistiche finali:")
    print(f"   Reward max: {max(rewards):.2f}")
    print(f"   Performance ultimi 100: {np.mean(rewards[-100:]):.2f}")
    print(f"   Sopravvivenza media ultimi 100: {np.mean(survival[-100:]):.1f}s")
    print(f"   Record assoluto: {max(survival):.1f}s")
