import json
import random
import numpy as np
from collections import deque
from typing import List
import torch
import torch.nn as nn
import gym


# i print con numero commentati e perchè sono certamente usati
# ==================================================
#  GodAct Core Classes
# ==================================================

class GodActRule:
    def __init__(self, rule_data: dict):
        #print("---1")
        self.rule_id = rule_data['rule_id']
        self.trigger = rule_data['trigger'] #ogni regola è attivata da un trigger
        self.effect = rule_data.get('effect', '')
        self.confidence = rule_data.get('confidence', 1.0)
        self.priority = rule_data.get('priority', 1.0)
        self.reward_modifier = rule_data.get('reward_modifier', 1.0)
        self.conditions = rule_data.get('conditions', {})

    # verifica se la regola si applica a una transizione
    def matches_state_transition(self, state, next_state, reward) -> bool:
        #print("---2")
        if state is None or next_state is None:
            return False
        if self.trigger == 'ball_paddle_collision': #esempi di trigger
            return self._check_ball_paddle_collision(state, next_state)
        elif self.trigger == 'ball_brick_collision':
            return reward > 1.0
        elif self.trigger == 'ball_lost':
            return reward < -4.0
        elif self.trigger == 'ball_wall_collision':
            return abs(state[2] + next_state[2]) < 1e-3
        return False

    def _check_ball_paddle_collision(self, state, next_state):
        print("---3")
        ball_y = (state[1] + 1) / 2
        prev_vy, next_vy = state[3], next_state[3]
        return prev_vy > 0 and next_vy < 0 and ball_y > 0.7

#Assegna priorità maggiore alle transizioni che attivano una regola o hanno reward estremo
class GodActPrioritizedReplayBuffer:
    def __init__(self, capacity: int, rules: List[GodActRule]):
        print("---4")
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.rules = rules
        self.rule_hit_counts = {rule.rule_id: 0 for rule in rules}
        self.total_transitions = 0

    def push(self, state, action, reward, next_state, done):
        print("---5")
        priority = 1.0
        matched = []
        for r in self.rules:
            if r.matches_state_transition(state, next_state, reward):
                priority *= r.priority
                matched.append(r.rule_id)
                self.rule_hit_counts[r.rule_id] += 1

        if matched:
            rarity_bonus = 1.0 + (1.0 / (1.0 + self.rule_hit_counts[matched[0]]))
            priority *= rarity_bonus
        if abs(reward) > 5:
            priority *= 2.0

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        self.total_transitions += 1

    def sample(self, batch_size: int):
        #print("---6")
        batch_size = min(batch_size, len(self.buffer))
        p = np.array(self.priorities, dtype=np.float64)
        probs = p / (p.sum() + 1e-8)
        idx = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        #print("---7")
        return len(self.buffer)

# Modifica il reward dell’agente in base alle regole.
class GodActRewardShaper:
    def __init__(self, rules: List[GodActRule]):
        print("---8")
        self.rules = rules

    def shape_reward(self, state, action, reward, next_state):
        #print("---9")
        shaped = float(reward)
        for r in self.rules:
            if r.trigger == 'ball_paddle_collision':
                ball_x = (state[0] + 1) / 2
                paddle_x = (state[4] + 1) / 2
                next_paddle_x = (next_state[4] + 1) / 2
                if state[3] > 0 and abs(next_paddle_x - ball_x) < abs(paddle_x - ball_x):
                    shaped += 0.3 * r.confidence
            elif r.trigger == 'ball_brick_collision' and reward > 1:
                shaped += 0.5 * r.confidence
            elif r.trigger == 'ball_lost' and reward < -4:
                shaped -= 0.2 * r.confidence
        return shaped

# Permette un curriculum learning: inizia con compiti semplici e aumenta la difficoltà.

class GodActCurriculumGenerator:
    def __init__(self, rules: List[GodActRule]):
        print("---10")
        self.rules = rules
        self.stages = self._generate_stages()
        self.current_stage = 0

    def _generate_stages(self):
        print("---11")
        return [
            {'name': 'paddle_focus', 'episodes': 50},
            {'name': 'brick_focus', 'episodes': 100},
            {'name': 'full_game', 'episodes': 500},
        ]

    def get_current_stage(self):
        print("---12")
        return self.stages[self.current_stage]

    # passa allo stage successivo quando completati gli episodi richiesti.
    def advance_stage(self):
        print("---13")
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"[Curriculum] Passato a stage: {self.get_current_stage()['name']}")

# wrapper che applica reward shaping e curriculum learning.
class GodActEnvWrapper(gym.Wrapper):
    def __init__(self, env, rules: List[GodActRule]):
        super().__init__(env)
        self.reward_shaper = GodActRewardShaper(rules) #qui1
        self.curriculum = GodActCurriculumGenerator(rules) #qui2 
        self.episode_count = 0

    def reset(self, **kwargs):
        print("---14")
        return self.env.reset(**kwargs)

    def step(self, action):
        #print("---15")
        state = self.env._get_obs()
        next_state, reward, done, info = self.env.step(action)
        shaped_reward = self.reward_shaper.shape_reward(state, action, reward, next_state)
        if done:
            self.episode_count += 1
            stage = self.curriculum.get_current_stage()
            if self.episode_count >= stage['episodes']:
                self.curriculum.advance_stage()
                self.episode_count = 0
        return next_state, shaped_reward, done, info

# Integrazione delle regole GodAct in DQN
class GodActDQNIntegrator:
    def __init__(self, rules_json_path=None, rules_dict=None):
        print("---16")
        if rules_dict is not None:
            # la popolazione caricata da pickle
            self.rules = [GodActRule(r) for r in rules_dict.get('rules', [])]
            print(1)
        elif rules_json_path is not None:
            self.rules = self._load_rules(rules_json_path)
            print(2)
        else:
            self.rules = []
            print(3)
        print(f"[GodAct] Caricate {len(self.rules)} regole.")

    # Carica regole da file JSON o dizionario Python.
    def _load_rules(self, path: str):
        print("---17")
        with open(path, 'r') as f:
            data = json.load(f)
        return [GodActRule(r) for r in data.get('rules', [])]

    def wrap_environment(self, env):
        print("---18")
        return GodActEnvWrapper(env, self.rules)

    def create_replay_buffer(self, capacity: int):
        print("---19")
        return GodActPrioritizedReplayBuffer(capacity, self.rules)


class GodActPopulationIntegrator:
    """
    Usa una popolazione euristica (best_population.pkl) come fonte di God Acts.
    """
    def __init__(self, population):
        print("--20")
        self.population = population
        # genera lista di dict
        rules_dicts = self._extract_rules_from_population(population)
        # print(rules_dicts)
        # **converti in GodActRule**
        self.rules = [GodActRule(r) for r in rules_dicts]
        print(len(self.rules))
        print("prontooo")
        self.reward_shaper = GodActRewardShaper(self.rules)

    def wrap_environment(self, env):
        print("---18")
        return GodActEnvWrapper(env, self.rules)
    def create_replay_buffer(self, capacity: int):
        print("---19")
        return GodActPrioritizedReplayBuffer(capacity, self.rules)


    def _extract_rules_from_population(self, population):
        print("---21 (filtered causes & effects)")
        rules = []

        def simple_numeric_priority(rule_obj) -> float:
            n = len(getattr(rule_obj, "causes", []) or []) + len(getattr(rule_obj, "effects", []) or [])
            return float(max(1.0, n))

        def infer_trigger_from_rule(rule_obj) -> str:
            texts = []
            for side in ('causes', 'effects'):
                vals = getattr(rule_obj, side, []) or []
                for v in vals:
                    try:
                        texts.append(repr(v).lower())
                    except Exception:
                        pass
            joined = " ".join(texts)
            if "paddle" in joined:
                return "ball_paddle_collision"
            if "brick" in joined or "block" in joined:
                return "ball_brick_collision"
            if "lost" in joined or "life" in joined:
                return "ball_lost"
            if "wall" in joined:
                return "ball_wall_collision"
            return "custom"

        def rule_to_dict(rule_obj, ind_id, obj_id, idx):
            try:
                rule_hash = getattr(rule_obj, "my_hash", lambda: None)()
            except Exception:
                rule_hash = None

            causes = getattr(rule_obj, "causes", []) or []
            effects = getattr(rule_obj, "effects", []) or []

            # filtriamo regole senza causes o effects
            if not causes or not effects:
                return None

            trigger = infer_trigger_from_rule(rule_obj)
            causes_list = []
            for c in causes:
                try:
                    causes_list.append({"type": type(c).__name__, "repr": repr(c), "hash": getattr(c, "my_hash", lambda: None)()})
                except Exception:
                    causes_list.append({"type": type(c).__name__, "repr": str(c)})

            effects_list = []
            for e in effects:
                try:
                    effects_list.append({"type": type(e).__name__, "repr": repr(e), "hash": getattr(e, "my_hash", lambda: None)()})
                except Exception:
                    effects_list.append({"type": type(e).__name__, "repr": str(e)})

            reward_modifier = 1.0
            joined_text = " ".join([c.get("repr","") for c in causes_list + effects_list])
            if "brick" in joined_text: reward_modifier = 1.2
            if "lost" in joined_text: reward_modifier = 0.5

            rule_id = f"{ind_id}_{obj_id}_{idx}"
            if rule_hash:
                rule_id += f"_{str(rule_hash)[:8]}"

            return {
                "rule_id": rule_id,
                "trigger": trigger,
                "effect": "; ".join([e.get("repr","") for e in effects_list]),
                "confidence": 1.0,
                "priority": simple_numeric_priority(rule_obj),
                "reward_modifier": float(reward_modifier),
                "conditions": {
                    "cause_offset": getattr(rule_obj, "cause_offset", None),
                    "causes": causes_list,
                    "effects": effects_list
                }
            }

        for ind_id, individual in population.items():
            if not hasattr(individual, "rules"):
                continue
            for obj_id, rule_list in individual.rules.items():
                if rule_list is None:
                    continue
                if not isinstance(rule_list, (list, tuple)):
                    rule_list = [rule_list]
                for idx, r in enumerate(rule_list):
                    try:
                        rd = rule_to_dict(r, ind_id, obj_id, idx)
                        if rd is not None:  # aggiungiamo solo regole con causes ed effects
                            rules.append(rd)
                    except Exception:
                        continue
        print("dentro:")
        print(len(rules))
        return rules



    