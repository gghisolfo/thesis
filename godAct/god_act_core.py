import json
import random
import numpy as np
from collections import deque
from typing import List
import torch
import torch.nn as nn
import gym

# ==================================================
#  GodAct Core Classes
# ==================================================

class GodActRule:
    def __init__(self, rule_data: dict):
        self.rule_id = rule_data['rule_id']
        self.trigger = rule_data['trigger']
        self.effect = rule_data.get('effect', '')
        self.confidence = rule_data.get('confidence', 1.0)
        self.priority = rule_data.get('priority', 1.0)
        self.reward_modifier = rule_data.get('reward_modifier', 1.0)
        self.conditions = rule_data.get('conditions', {})

    def matches_state_transition(self, state, next_state, reward) -> bool:
        if state is None or next_state is None:
            return False
        if self.trigger == 'ball_paddle_collision':
            return self._check_ball_paddle_collision(state, next_state)
        elif self.trigger == 'ball_brick_collision':
            return reward > 1.0
        elif self.trigger == 'ball_lost':
            return reward < -4.0
        elif self.trigger == 'ball_wall_collision':
            return abs(state[2] + next_state[2]) < 1e-3
        return False

    def _check_ball_paddle_collision(self, state, next_state):
        ball_y = (state[1] + 1) / 2
        prev_vy, next_vy = state[3], next_state[3]
        return prev_vy > 0 and next_vy < 0 and ball_y > 0.7


class GodActPrioritizedReplayBuffer:
    def __init__(self, capacity: int, rules: List[GodActRule]):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.rules = rules
        self.rule_hit_counts = {rule.rule_id: 0 for rule in rules}
        self.total_transitions = 0

    def push(self, state, action, reward, next_state, done):
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
        return len(self.buffer)


class GodActRewardShaper:
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules

    def shape_reward(self, state, action, reward, next_state):
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


class GodActCurriculumGenerator:
    def __init__(self, rules: List[GodActRule]):
        self.rules = rules
        self.stages = self._generate_stages()
        self.current_stage = 0

    def _generate_stages(self):
        return [
            {'name': 'paddle_focus', 'episodes': 50},
            {'name': 'brick_focus', 'episodes': 100},
            {'name': 'full_game', 'episodes': 500},
        ]

    def get_current_stage(self):
        return self.stages[self.current_stage]

    def advance_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"[Curriculum] Passato a stage: {self.get_current_stage()['name']}")


class GodActEnvWrapper(gym.Wrapper):
    def __init__(self, env, rules: List[GodActRule]):
        super().__init__(env)
        self.reward_shaper = GodActRewardShaper(rules)
        self.curriculum = GodActCurriculumGenerator(rules)
        self.episode_count = 0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
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


class GodActDQNIntegrator:
    def __init__(self, rules_json_path=None, rules_dict=None):
        if rules_dict is not None:
            # la popolazione caricata da pickle
            self.rules = [GodActRule(r) for r in rules_dict.get('rules', [])]
        elif rules_json_path is not None:
            self.rules = self._load_rules(rules_json_path)
        else:
            self.rules = []
        print(f"[GodAct] Caricate {len(self.rules)} regole.")

    def _load_rules(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return [GodActRule(r) for r in data.get('rules', [])]

    def wrap_environment(self, env):
        return GodActEnvWrapper(env, self.rules)

    def create_replay_buffer(self, capacity: int):
        return GodActPrioritizedReplayBuffer(capacity, self.rules)


class GodActPopulationIntegrator:
    """
    Usa una popolazione euristica (best_population.pkl) come fonte di God Acts.
    """
    def __init__(self, population):
        self.population = population
        self.rules = self._extract_rules_from_population(population)
        self.reward_shaper = GodActRewardShaper(self.rules)

    def _extract_rules_from_population(self, population):
        rules = []
        for ind_id, individual in population.items():
            if hasattr(individual, "rules"):
                for obj_id, rule_list in individual.rules.items():
                    for r in rule_list:
                        # converte l’oggetto euristico in GodActRule “classico”
                        try:
                            rules.append(GodActRule({
                                "rule_id": f"{ind_id}_{obj_id}_{r.get('name', 'rule')}",
                                "trigger": r.get("trigger", "custom"),
                                "effect": r.get("effect", ""),
                                "confidence": float(r.get("confidence", 1.0)),
                                "priority": float(r.get("priority", 1.0)),
                                "reward_modifier": float(r.get("reward_modifier", 1.0))
                            }))
                        except Exception:
                            continue
        return rules

    def wrap_environment(self, env):
        """
        Aggiunge reward shaping e replay buffer all’ambiente RL.
        """
        env.reward_shaper = self.reward_shaper
        return env

    def create_replay_buffer(self, capacity=50000):
        return GodActPrioritizedReplayBuffer(capacity, self.rules)
