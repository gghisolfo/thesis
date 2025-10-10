"""
Modulo di integrazione: God Acts → DQN
Prende in input le regole estratte dai video e le usa per accelerare il training DQN

Input: File JSON con gli atti di Dio e regole inferite
Output: DQN configurata dinamicamente in base alle regole scoperte
"""

import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import List, Dict, Tuple, Callable
import gym


# ============================================================================
# STRUTTURE DATI PER GLI ATTI DI DIO
# ============================================================================

class GodActRule:
    """Rappresenta una regola inferita dagli atti di Dio"""
    
    def __init__(self, rule_data: dict):
        self.rule_id = rule_data['rule_id']
        self.trigger = rule_data['trigger']  # Es: 'ball_paddle_collision'
        self.effect = rule_data['effect']    # Es: 'velocity_invert_y'
        self.confidence = rule_data['confidence']
        self.priority = rule_data.get('priority', 1.0)
        self.reward_modifier = rule_data.get('reward_modifier', 1.0)
        
        # Condizioni per riconoscere quando la regola si applica
        self.conditions = rule_data.get('conditions', {})
    
    def matches_state_transition(self, state, next_state, reward) -> bool:
        """Controlla se questa regola si applica alla transizione osservata"""
        # Implementa la logica specifica per ogni tipo di trigger
        if self.trigger == 'ball_paddle_collision':
            return self._check_ball_paddle_collision(state, next_state)
        elif self.trigger == 'ball_brick_collision':
            return self._check_ball_brick_collision(state, next_state, reward)
        elif self.trigger == 'ball_wall_collision':
            return self._check_ball_wall_collision(state, next_state)
        elif self.trigger == 'ball_lost':
            return reward < -5  # Penalità grande = palla persa
        return False
    
    def _check_ball_paddle_collision(self, state, next_state) -> bool:
        """Rileva collisione palla-paddle"""
        # Assumendo state = [ball_x, ball_y, ball_vx, ball_vy, paddle_x]
        ball_y = (state[1] + 1) / 2  # Denormalizza
        next_ball_vy = next_state[3]
        ball_vy = state[3]
        
        # Palla in zona paddle + inversione velocità verticale
        return ball_y > 0.85 and ball_vy > 0 and next_ball_vy < 0
    
    def _check_ball_brick_collision(self, state, next_state, reward) -> bool:
        """Rileva collisione palla-mattone"""
        ball_y = (state[1] + 1) / 2
        # Zona mattoni + reward positivo alto
        return 0.05 < ball_y < 0.3 and reward > 5
    
    def _check_ball_wall_collision(self, state, next_state) -> bool:
        """Rileva collisione con muro laterale"""
        ball_vx = state[2]
        next_ball_vx = next_state[2]
        # Inversione velocità orizzontale
        return abs(ball_vx + next_ball_vx) < 0.01


# ============================================================================
# REPLAY BUFFER CON PRIORITÀ DINAMICA DA GOD ACTS
# ============================================================================

class GodActPrioritizedReplayBuffer:
    """
    Replay buffer che usa le regole degli atti di Dio per assegnare priorità dinamiche
    """
    
    def __init__(self, capacity: int, rules: List[GodActRule]):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.rules = rules
        
        # Statistiche per adattare le priorità
        self.rule_hit_counts = {rule.rule_id: 0 for rule in rules}
        self.total_transitions = 0
    
    def push(self, state, action, reward, next_state, done):
        """Aggiunge transizione con priorità calcolata dalle regole"""
        
        # Calcola priorità base
        priority = 1.0
        matched_rules = []
        
        # Controlla quali regole si applicano
        for rule in self.rules:
            if rule.matches_state_transition(state, next_state, reward):
                priority *= rule.priority
                matched_rules.append(rule.rule_id)
                self.rule_hit_counts[rule.rule_id] += 1
        
        # Bonus per transizioni rare (esplorazione)
        if len(matched_rules) > 0:
            rarity_bonus = 1.0 + (1.0 / (1.0 + self.rule_hit_counts[matched_rules[0]]))
            priority *= rarity_bonus
        
        # Bonus per transizioni con reward estremo
        if abs(reward) > 5:
            priority *= 2.0
        
        self.buffer.append((state, action, reward, next_state, done, matched_rules))
        self.priorities.append(priority)
        self.total_transitions += 1
    
    def sample(self, batch_size: int) -> Tuple:
        """Campiona batch con probabilità proporzionale alla priorità"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=priorities, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, rules_matched = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def get_statistics(self) -> Dict:
        """Ritorna statistiche sulle regole applicate"""
        return {
            rule_id: count / max(1, self.total_transitions)
            for rule_id, count in self.rule_hit_counts.items()
        }
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# REWARD SHAPING DINAMICO DA GOD ACTS
# ============================================================================

class GodActRewardShaper:
    """
    Modifica i reward in base alle regole scoperte per dare feedback anticipato
    """
    
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules
        
        # Mappa trigger → funzione di reward shaping
        self.shaping_functions = {
            'ball_paddle_collision': self._shape_paddle_approach,
            'ball_brick_collision': self._shape_brick_targeting,
            'ball_lost': self._shape_ball_preservation,
        }
    
    def shape_reward(self, state, action, reward, next_state, env_info=None) -> float:
        """Applica reward shaping basato sulle regole"""
        shaped_reward = reward
        
        for rule in self.rules:
            if rule.trigger in self.shaping_functions:
                shaping_func = self.shaping_functions[rule.trigger]
                shaped_reward += shaping_func(state, action, next_state) * rule.confidence
        
        return shaped_reward
    
    def _shape_paddle_approach(self, state, action, next_state) -> float:
        """Reward per avvicinarsi alla palla quando sta scendendo"""
        ball_x = (state[0] + 1) / 2
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        paddle_x = (state[4] + 1) / 2
        next_paddle_x = (next_state[4] + 1) / 2
        
        # Solo se la palla sta scendendo
        if ball_vy > 0 and ball_y > 0.5:
            prev_dist = abs(paddle_x - ball_x)
            next_dist = abs(next_paddle_x - ball_x)
            
            # Ricompensa per essersi avvicinato
            if next_dist < prev_dist:
                return 0.5 * (1.0 - ball_y)  # Più urgente quando palla è in basso
        
        return 0.0
    
    def _shape_brick_targeting(self, state, action, next_state) -> float:
        """Reward per mirare ai mattoni"""
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        
        # Se la palla sta salendo verso i mattoni
        if ball_vy < 0 and 0.3 < ball_y < 0.5:
            return 0.2
        
        return 0.0
    
    def _shape_ball_preservation(self, state, action, next_state) -> float:
        """Penalità per lasciar avvicinare troppo la palla al bordo inferiore"""
        ball_y = (state[1] + 1) / 2
        ball_vy = state[3]
        
        # Palla sta scendendo pericolosamente
        if ball_vy > 0 and ball_y > 0.8:
            return -0.3 * (ball_y - 0.8)  # Penalità crescente
        
        return 0.0


# ============================================================================
# CURRICULUM LEARNING GENERATOR DA GOD ACTS
# ============================================================================

class GodActCurriculumGenerator:
    """
    Genera stati di partenza interessanti basati sugli atti di Dio osservati
    """
    
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules
        self.curriculum_stages = self._generate_stages()
        self.current_stage = 0
    
    def _generate_stages(self) -> List[Dict]:
        """Genera stages curriculum dalle regole"""
        stages = []
        
        # Stage 1: Impara collisioni paddle (atto di Dio più critico)
        if any(r.trigger == 'ball_paddle_collision' for r in self.rules):
            stages.append({
                'name': 'paddle_collision_training',
                'init_func': self._init_paddle_collision,
                'episodes': 200
            })
        
        # Stage 2: Impara collisioni mattoni
        if any(r.trigger == 'ball_brick_collision' for r in self.rules):
            stages.append({
                'name': 'brick_collision_training',
                'init_func': self._init_brick_collision,
                'episodes': 300
            })
        
        # Stage 3: Gioco completo
        stages.append({
            'name': 'full_game',
            'init_func': None,  # Reset normale
            'episodes': 500
        })
        
        return stages
    
    def get_current_stage(self) -> Dict:
        """Ritorna lo stage corrente"""
        return self.curriculum_stages[self.current_stage]
    
    def advance_stage(self):
        """Passa allo stage successivo"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            print(f"[Curriculum] Avanzato a stage: {self.get_current_stage()['name']}")
    
    def _init_paddle_collision(self) -> Dict:
        """Inizializzazione per allenare collisioni paddle"""
        import random
        return {
            'ball_x': 0.5 + (random.random() - 0.5) * 0.3,
            'ball_y': 0.85,
            'ball_vx': (random.random() - 0.5) * 0.04,
            'ball_vy': 0.03,
            'paddle_x': 0.5 + (random.random() - 0.5) * 0.2
        }
    
    def _init_brick_collision(self) -> Dict:
        """Inizializzazione per allenare collisioni mattoni"""
        import random
        return {
            'ball_x': 0.5 + (random.random() - 0.5) * 0.4,
            'ball_y': 0.35,
            'ball_vx': (random.random() - 0.5) * 0.04,
            'ball_vy': -0.03,
            'paddle_x': 0.5
        }


# ============================================================================
# WRAPPER PER GYM ENVIRONMENT CON GOD ACTS
# ============================================================================

class GodActEnvWrapper(gym.Wrapper):
    """
    Wrapper che integra reward shaping e curriculum learning basati su God Acts
    """
    
    def __init__(self, env, rules: List[GodActRule]):
        super().__init__(env)
        self.reward_shaper = GodActRewardShaper(rules)
        self.curriculum = GodActCurriculumGenerator(rules)
        self.episode_count = 0
    
    def reset(self, **kwargs):
        """Reset con curriculum learning"""
        stage = self.curriculum.get_current_stage()
        
        # Usa funzione init custom se disponibile
        if stage['init_func'] is not None:
            init_params = stage['init_func']()
            # Applica i parametri all'environment (devi implementare questo nel tuo env)
            if hasattr(self.env, 'reset_with_params'):
                return self.env.reset_with_params(init_params)
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step con reward shaping"""
        state = self._get_current_state()
        next_state, reward, done, info = self.env.step(action)
        
        # Applica reward shaping
        shaped_reward = self.reward_shaper.shape_reward(state, action, reward, next_state)
        
        # Avanza curriculum se necessario
        if done:
            self.episode_count += 1
            stage = self.curriculum.get_current_stage()
            if self.episode_count >= stage['episodes']:
                self.curriculum.advance_stage()
                self.episode_count = 0
        
        return next_state, shaped_reward, done, info
    
    def _get_current_state(self):
        """Ritorna lo stato corrente (devi adattarlo al tuo environment)"""
        if hasattr(self.env, 'get_observation'):
            return self.env.get_observation()
        return self.env._get_obs() if hasattr(self.env, '_get_obs') else None


# ============================================================================
# INTEGRAZIONE COMPLETA: CARICA REGOLE E CONFIGURA DQN
# ============================================================================

class GodActDQNIntegrator:
    """
    Classe principale per integrare God Acts in una DQN esistente
    """
    
    def __init__(self, rules_json_path: str):
        self.rules = self._load_rules(rules_json_path)
        print(f"[GodActDQN] Caricate {len(self.rules)} regole")
        self._print_rules_summary()
    
    def _load_rules(self, json_path: str) -> List[GodActRule]:
        """Carica regole da file JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        rules = []
        for rule_data in data['rules']:
            rules.append(GodActRule(rule_data))
        
        return rules
    
    def _print_rules_summary(self):
        """Stampa sommario delle regole caricate"""
        print("\n=== Regole Scoperte ===")
        for rule in self.rules:
            print(f"  [{rule.rule_id}] {rule.trigger} → {rule.effect}")
            print(f"    Confidence: {rule.confidence:.2f}, Priority: {rule.priority:.2f}")
        print("=" * 50 + "\n")
    
    def wrap_environment(self, env):
        """Wrappa l'environment con le funzionalità God Acts"""
        return GodActEnvWrapper(env, self.rules)
    
    def create_replay_buffer(self, capacity: int):
        """Crea replay buffer prioritizzato con God Acts"""
        return GodActPrioritizedReplayBuffer(capacity, self.rules)
    
    def get_curriculum_generator(self):
        """Ritorna il generatore di curriculum"""
        return GodActCurriculumGenerator(self.rules)


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Esempio di file JSON con regole estratte
    example_rules = {
        "rules": [
            {
                "rule_id": "R1",
                "trigger": "ball_paddle_collision",
                "effect": "velocity_invert_y",
                "confidence": 0.95,
                "priority": 3.0,
                "reward_modifier": 1.5
            },
            {
                "rule_id": "R2",
                "trigger": "ball_brick_collision",
                "effect": "brick_disappear_and_velocity_invert",
                "confidence": 0.90,
                "priority": 2.0,
                "reward_modifier": 2.0
            },
            {
                "rule_id": "R3",
                "trigger": "ball_lost",
                "effect": "game_over",
                "confidence": 1.0,
                "priority": 2.5,
                "reward_modifier": 1.0
            }
        ]
    }
    
    # Salva esempio
    with open('god_acts_rules.json', 'w') as f:
        json.dump(example_rules, f, indent=2)
    
    print("Creato file di esempio: god_acts_rules.json")
    print("\n=== Come usare questo modulo ===")

    # 1. Carica le regole scoperte dal video
    integrator = GodActDQNIntegrator('god_acts_rules.json')

    # 2. Wrappa il tuo environment
    from dqn_arkanoid_pygame import ArkanoidEnv
    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    # 3. Usa il replay buffer prioritizzato
    replay_buffer = integrator.create_replay_buffer(capacity=50000)

    # 4. Training normale, ma accelerato dalle regole!
    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = select_action(state)  # La tua policy
            next_state, reward, done, _ = env.step(action)  # Reward già shaped!
            replay_buffer.push(state, action, reward, next_state, done)  # Priorità automatica!
            # ... resto del training ...
