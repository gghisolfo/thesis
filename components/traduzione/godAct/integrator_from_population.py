import torch
from god_act_core import GodActRule, GodActRewardShaper
from god_act_core import GodActPrioritizedReplayBuffer

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
