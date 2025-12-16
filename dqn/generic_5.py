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
import copy
import pickle
from collections import Counter

# Import locali 
from arkanoid_game import Game, grid_width, grid_height


# python -m dqn.generic_3 


SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)
PRINT_MODE = False
# Inserisci queste opzioni globali vicino ai tuoi hyperparams
USE_ACTION_ATTRIBUTION = True    # abilita il confronto action vs no-op
NOOP_ACTION_ID = 1               # id dell'azione "fermo" nella mappa azioni
ATTRIBUTION_P_COSTLY = True      # True = prova pickle, se fallisce fallback



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
            
            # Soglia relativa (5% minimo)
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
        
        Formula: R = base_reward * (chain_length ^ exponent)
        
        Esempi con exponent=1.2:
        - 1 evento  → 1.0^1.2 = 1.0
        - 2 eventi  → 2.0^1.2 ≈ 2.3  
        - 3 eventi  → 3.0^1.2 ≈ 3.7  
        - 5 eventi  → 5.0^1.2 ≈ 6.9  
        - 10 eventi → 10.0^1.2 ≈ 15.8
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
        self.event_detector = GenericEventDetector(
            threshold_for_change=0.3,      # Soglia intermedia (era 1.0, troppo alta)
            min_relative_change=0.03       # 3% minimo (era 0.1, troppo alto)
        )
        self.event_tracker = CausalEventChainTracker(
            causal_window=causal_window,
            chain_exponent=chain_exponent
        )
        
        self._prev_state = None
        self.done = False

        # per eventi che avvengono sempre
        self.attr_stats = defaultdict(lambda: {'when_action':0, 'when_noop':0, 'total':0})
        self.ACTION_CONTROL_THRESHOLD = 0.9  # se >90% dei cambi quando agente agisce -> rumore

    
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
    
    # --- sostituisci il corpo di GenericSymbolicEnv.step con questa versione ---
    # def step(self, action: int):
    #     prev_state = self._prev_state

    #     # --- (A) Prova a creare uno snapshot della simulazione ---
    #     snapshot = None
    #     snapshot_via_pickle = False
    #     if USE_ACTION_ATTRIBUTION:
    #         try:
    #             snapshot = pickle.dumps(self.sim)   # prova rapide snapshot via pickle
    #             snapshot_via_pickle = True
    #         except Exception:
    #             try:
    #                 snapshot = copy.deepcopy(self.sim)  # fallback a deepcopy (potrebbe fallire o essere pesante)
    #                 snapshot_via_pickle = False
    #             except Exception:
    #                 snapshot = None
    #                 # non possiamo fare attribution per questo step

    #     # --- (B) Esegui l'azione scelta sulla simulazione reale ---
    #     if action in self.action_map:
    #         print("ACTION:", action)
    #         self.action_map[action](self.sim)

    #     if hasattr(self.sim, 'update'):
    #         self.sim.update()

    #     # Cattura nuovo stato + eventi dopo l'azione
    #     current_state = self.state_extractor.extract(self.sim)
    #     detected_events_action = self.event_detector.detect_events(prev_state, current_state)



    #     # 'action_taken' bool: se l'azione non è NOOP_ACTION_ID
    #     action_taken = (action != NOOP_ACTION_ID)

    #     # aggiorna contatori per ogni attributo che è cambiato nell'azione/noop
    #     for e in detected_events_action:
    #         a = e['attribute']
    #         self.attr_stats[a]['total'] += 1
    #         if action_taken:
    #             self.attr_stats[a]['when_action'] += 1
    #         else:
    #             self.attr_stats[a]['when_noop'] += 1

    #     # ora calcola attributi action-controlled
    #     action_controlled_attrs = {
    #         a for a, st in self.attr_stats.items()
    #         if st['total'] >= 5 and (st['when_action'] / st['total']) >= self.ACTION_CONTROL_THRESHOLD
    #     }
    #     # filtra gli eventi
    #     filtered_action_events = [e for e in detected_events_action if e['attribute'] not in action_controlled_attrs]
    #     filtered_noop_events   = [e for e in detected_events_noop   if e['attribute'] not in action_controlled_attrs]


    #     # --- (C) Se possibile, esegui no-op sulla copia per vedere cosa succede senza la nostra azione ---
    #     detected_events_noop = []
    #     if USE_ACTION_ATTRIBUTION and snapshot is not None:
    #         try:
    #             # Ripristina la copia
    #             if snapshot_via_pickle:
    #                 sim_copy = pickle.loads(snapshot)
    #             else:
    #                 sim_copy = copy.deepcopy(snapshot) if not isinstance(snapshot, bytes) else pickle.loads(snapshot)
    #         except Exception:
    #             sim_copy = None

    #         if sim_copy is not None:
    #             # Esegui la no-op definita (se presente), altrimenti niente
    #             if NOOP_ACTION_ID in self.action_map:
    #                 # la no-op è la stessa lambda ma applicata a sim_copy
    #                 self.action_map[NOOP_ACTION_ID](sim_copy)
    #             # se non c'è no-op, semplicemente non applicare alcuna azione (lascia sim_copy avanzare "naturally")
    #             if hasattr(sim_copy, 'update'):
    #                 sim_copy.update()

    #             # stato dopo noop
    #             noop_state = self.state_extractor.extract(sim_copy)
    #             detected_events_noop = self.event_detector.detect_events(prev_state, noop_state)
    #         else:
    #             detected_events_noop = []

    #         # --- (D) Attribuzione causale: differenza tra i due insiemi (genericamente sulle attribute) ---
    #         # Rappresentiamo gli eventi come multiset di attributi (così ripetizioni contano)
    #         attrs_action = [e['attribute'] for e in detected_events_action]
    #         attrs_noop = [e['attribute'] for e in detected_events_noop]
    #         counter_action = Counter(attrs_action)
    #         counter_noop = Counter(attrs_noop)

    #         # E_caused = elementi in action con count > noop count
    #         caused_counter = counter_action - counter_noop
    #         caused_count = sum(caused_counter.values())

    #         # (Se snapshot non disponibile, fallback ad usare detected_events_action come before)
    #         if snapshot is None:
    #             # fallback (meno preciso): consideriamo tutto ciò che è cambiato come potenzialmente causale
    #             caused_count = len(detected_events_action)

    #         # --- (E) Calcolo reward dalla causality (nessun uso di tipi) ---
    #         # Usa chain exponent lineare o leggermente >1 ma con cap
    #         chain_length = caused_count
    #         event_reward = (chain_length ** self.event_tracker.chain_exponent) if chain_length > 0 else 0.0
    #         if hasattr(self, 'MAX_CHAIN_REWARD'):
    #             event_reward = min(event_reward, self.MAX_CHAIN_REWARD)

    #         # --- (F) opzionali: attenuation per ripetizione (mantieni genericità) ---
    #         # qui puoi implementare la logica di recent_event_counts se vuoi, ma non è per forza necessaria

    #         # REWARD SHAPING (survival + proximity)
    #         shaping_reward = 0.0
    #         if hasattr(self.sim, 'ball_y') and hasattr(self.sim, 'paddle_x'):
    #             ball_x = getattr(self.sim, 'ball_x', 0)
    #             paddle_x = getattr(self.sim, 'paddle_x', 0)
    #             distance = abs(ball_x - paddle_x)
    #             max_distance = getattr(self.sim, 'grid_width', 100)
    #             proximity_bonus = 0.1 * (1.0 - min(distance / max_distance, 1.0))
    #             shaping_reward = proximity_bonus

    #         survival_bonus = getattr(self, 'SURVIVAL_BONUS_PER_STEP', 0.0)
    #         terminal_penalty = 0.0
    #         if getattr(self.sim, 'ball_lost', False):
    #             terminal_penalty = getattr(self, 'BALL_LOST_PENALTY', 0.0)

    #         reward = event_reward + survival_bonus + shaping_reward * 0.1 + terminal_penalty

    #         # Mantieni stato precedente e tracker
    #         self._prev_state = current_state
    #         # registra eventi (usato per statistiche / accumulo)
    #         self.event_tracker.add_events(detected_events_action)

    #         # Statistiche
    #         stats = self.event_tracker.get_statistics()
    #         info = {
    #             'events': detected_events_action,
    #             'events_noop': detected_events_noop,
    #             'caused_count': caused_count,
    #             'chain_length': len(detected_events_action),
    #             'event_reward': event_reward,
    #             'shaping_reward': shaping_reward,
    #             'step_reward': reward,
    #             **stats
    #         }

    #         # Terminazione
    #         self.done = self.termination_check(self.sim)

    #         return self.observation_extractor(self.sim), reward, self.done, info
    



    def step(self, action: int):
        prev_state = self._prev_state

        # --- (A) Prova a creare uno snapshot della simulazione ---
        snapshot = None
        snapshot_via_pickle = False
        if USE_ACTION_ATTRIBUTION:
            try:
                snapshot = pickle.dumps(self.sim)
                snapshot_via_pickle = True
            except Exception:
                try:
                    snapshot = copy.deepcopy(self.sim)
                    snapshot_via_pickle = False
                except Exception:
                    snapshot = None

        # --- (B) Esegui l'azione scelta sulla simulazione reale ---
        if action in self.action_map:
            # print("ACTION:", action)
            self.action_map[action](self.sim)

        if hasattr(self.sim, 'update'):
            self.sim.update()

        # Cattura nuovo stato + eventi dopo l'azione
        current_state = self.state_extractor.extract(self.sim)
        detected_events_action = self.event_detector.detect_events(prev_state, current_state)

        # --- (C) Se possibile, esegui no-op sulla copia per vedere cosa succede senza la nostra azione ---
        detected_events_noop = []
        if USE_ACTION_ATTRIBUTION and snapshot is not None:
            try:
                if snapshot_via_pickle:
                    sim_copy = pickle.loads(snapshot)
                else:
                    # se snapshot non è bytes, è già deepcopy; altrimenti carica bytes
                    sim_copy = copy.deepcopy(snapshot) if not isinstance(snapshot, bytes) else pickle.loads(snapshot)
            except Exception:
                sim_copy = None

            if sim_copy is not None:
                if NOOP_ACTION_ID in self.action_map:
                    # applica la no-op alla copia
                    self.action_map[NOOP_ACTION_ID](sim_copy)
                if hasattr(sim_copy, 'update'):
                    sim_copy.update()

                noop_state = self.state_extractor.extract(sim_copy)
                detected_events_noop = self.event_detector.detect_events(prev_state, noop_state)
            else:
                detected_events_noop = []
        else:
            detected_events_noop = []

        # --- (D) AGGIORNA STATISTICHE ATTRIBUTO per auto-detect (dopo aver ottenuto entrambi gli insiemi) ---
        action_taken = (action != NOOP_ACTION_ID)
        for e in detected_events_action:
            a = e['attribute']
            st = self.attr_stats[a]
            st['total'] += 1
            if action_taken:
                st['when_action'] += 1
            else:
                st['when_noop'] += 1

        # Identifica attributi probabilmente controllati dall'azione (solo dopo alcuni campioni)
        action_controlled_attrs = {
            a for a, st in self.attr_stats.items()
            if st['total'] >= 8 and (st['when_action'] / (st['total'] + 1e-9)) >= getattr(self, 'ACTION_CONTROL_THRESHOLD', 0.9)
        }

        # Filtra eventi rumorosi (es. paddle_x, paddle_speed) automaticamente
        filtered_action_events = [e for e in detected_events_action if e['attribute'] not in action_controlled_attrs]
        filtered_noop_events   = [e for e in detected_events_noop   if e['attribute'] not in action_controlled_attrs]

        # --- (E) Attribuzione causale: differenza tra i due insiemi (sulle attribute), in modo conservativo ---
        attrs_action = [e['attribute'] for e in filtered_action_events]
        attrs_noop   = [e['attribute'] for e in filtered_noop_events]
        counter_action = Counter(attrs_action)
        counter_noop = Counter(attrs_noop)

        if snapshot is None:
            # fallback conservativo: al massimo 1 attributo causale per step se non possiamo testare noop
            caused_count = min(1, len(set(attrs_action)))
        else:
            caused_counter = counter_action - counter_noop
            # conteggia attributi unici effettivamente in più rispetto al noop
            caused_count = len([k for k, v in caused_counter.items() if v > 0])

        # Calcola chain_length come numero di attributi unici cambiati nello step (con cap)
        chain_length = min(len(set(attrs_action)), getattr(self, 'MAX_PER_STEP_CHAIN', 6))
        # chain_length = min(len(set(attrs_action)), getattr(self, 'MAX_PER_STEP_CHAIN', 5))
        # print(self.MAX_PER_STEP_CHAIN)


        # --- (F) Calcolo reward dalla causality (con cap opzionale)
        event_reward = (chain_length ** self.event_tracker.chain_exponent) if chain_length > 0 else 0.0
        if hasattr(self, 'MAX_CHAIN_REWARD'):
            event_reward = min(event_reward, self.MAX_CHAIN_REWARD)
        else:
            print("no MAX_CHAIN_REWARD")

        # REWARD SHAPING (survival + proximity)
        shaping_reward = 0.0
        if hasattr(self.sim, 'ball_y') and hasattr(self.sim, 'paddle_x'):
            ball_x = getattr(self.sim, 'ball_x', 0)
            paddle_x = getattr(self.sim, 'paddle_x', 0)
            distance = abs(ball_x - paddle_x)
            max_distance = getattr(self.sim, 'grid_width', 100)
            proximity_bonus = 0.1 * (1.0 - min(distance / max_distance, 1.0))
            shaping_reward = proximity_bonus

        survival_bonus = getattr(self, 'SURVIVAL_BONUS_PER_STEP', 0.0)
        terminal_penalty = 0.0
        if getattr(self.sim, 'ball_lost', False):
            terminal_penalty = getattr(self, 'BALL_LOST_PENALTY', 0.0)

        reward = event_reward + survival_bonus + shaping_reward * 0.1 + terminal_penalty

        # Mantieni stato precedente e tracker (registra gli eventi filtrati, non il rumore)
        self._prev_state = current_state
        self.event_tracker.add_events(filtered_action_events)

        # Statistiche e info
        stats = self.event_tracker.get_statistics()
        info = {
            'events': filtered_action_events,
            'events_noop': filtered_noop_events,
            'caused_count': caused_count,
            'chain_length': chain_length,
            'event_reward': event_reward,
            'shaping_reward': shaping_reward,
            'step_reward': reward,
            **stats
        }

        # Terminazione
        self.done = self.termination_check(self.sim)
        

        return self.observation_extractor(self.sim), reward, self.done, info

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
    env = GenericSymbolicEnv(
        sim_object=Game(),
        action_map={
            0: lambda game: game.set_paddle_speed(-1),
            1: lambda game: game.set_paddle_speed(0),
            2: lambda game: game.set_paddle_speed(1),
        },
        observation_extractor=lambda game: np.array([
            game.ball_x/grid_width*2-1,
            game.ball_y/grid_height*2-1,
            game.ball_speed_x/10.0,
            game.ball_speed_y/10.0,
            game.paddle_x/grid_width*2-1
        ], dtype=np.float32),
        termination_check=lambda game: game.ball_lost or game.bricks_alive==0,
        causal_window=3,
        chain_exponent=1.2
    )

    # Limite massimo per catena di eventi per step
    env.MAX_PER_STEP_CHAIN = 5  
    env.MAX_CHAIN_REWARD = 5.0  


    return env



def train_generic_dqn(env_factory: Callable, total_episodes=10000, max_steps=3000):
    """
    Training DQN completamente generico con reward causale.
    
    Per Arkanoid con reward solo da eventi:
    - 5,000-10,000 episodi per comportamento base (non perdere subito)
    - 10,000-20,000 episodi per giocare decentemente (>1 minuto)
    - 50,000+ episodi per padronanza completa
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
            avg_reward = np.mean(rewards_history[-50:])
            avg_survival = np.mean(survival_times[-50:])
            avg_chains = np.mean([s['total_chains'] for s in chain_stats_history[-50:]])
            avg_chain_len = np.mean([s['avg_chain_length'] for s in chain_stats_history[-50:]])
            
            print(f"[Ep {ep:5d}] ε={epsilon:.3f} | R: {total_reward:6.1f} (avg {avg_reward:6.1f}) | "
                  f"Survived: {survival_time:4.1f}s (avg {avg_survival:4.1f}s) | "
                  f"total_chains: {stats['total_chains']:3d} (avg_chain_len {avg_chain_len:.1f})")
        
        # Milestone checks
        if ep == 1000:
            avg_surv_1k = np.mean(survival_times[-100:])
            print(f"\n📊 Milestone 1000 episodi: Sopravvivenza media = {avg_surv_1k:.1f}s")
            if avg_surv_1k < 5.0:
                print("⚠️  Agente ancora debole. Potrebbe servire più training.")
        
        if ep == 5000:
            avg_surv_5k = np.mean(survival_times[-100:])
            print(f"\n📊 Milestone 5000 episodi: Sopravvivenza media = {avg_surv_5k:.1f}s")
            if avg_surv_5k > 60.0:
                print("🎉 Obiettivo 1 minuto raggiunto!")
            else:
                print(f"   Mancano ~{60 - avg_surv_5k:.1f}s per 1 minuto")

    # Salva modello
    model_path = os.path.join(SAVE_DIR, "generic_5.pth")
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
    # print("🚀 Training DQN con Causal Event Chains")
    # print("=" * 70)
    # print("PRINCIPIO: Tutti gli eventi hanno peso uguale,")
    # print("           ma catene causali hanno reward esponenziale")
    # print("=" * 70)
    # print("\n📋 Piano di training:")
    # print("   - 1,000 ep:   Apprendimento base (aspettati ~5-10s sopravvivenza)")
    # print("   - 5,000 ep:   Comportamento decente (~20-40s)")
    # print("   - 10,000 ep:  Target 1 minuto (60s+)")
    # print("   - 20,000 ep:  Padronanza completa")
    # print("=" * 70 + "\n")
    
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
    
    # Analisi milestone
    if len(survival) >= 1000:
        print(f"\n📊 Progressione apprendimento:")
        print(f"   Primi 100 ep:    {np.mean(survival[:100]):.1f}s")
        print(f"   1000 ep:         {np.mean(survival[900:1000]):.1f}s")
        if len(survival) >= 5000:
            print(f"   5000 ep:         {np.mean(survival[4900:5000]):.1f}s")
        print(f"   Ultimi 100 ep:   {np.mean(survival[-100:]):.1f}s")
    
    # Suggerimenti
    # avg_final = np.mean(survival[-100:])
    # if avg_final < 60:
    #     additional_eps = int((60 - avg_final) / avg_final * 10000)
    #     print(f"\n💡 Suggerimento: Prova altri {additional_eps:,} episodi per raggiungere 1 minuto")