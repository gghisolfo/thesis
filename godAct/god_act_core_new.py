# Nel tuo modulo GodAct (es. godact_core.py)

class GodActRule:
    """Rappresenta una regola GodAct con trigger, priorità e modificatori di reward."""

    def __init__(self, rule_data):
        # Ipotizziamo che rule_data contenga 'name', 'trigger', 'confidence', 'priority', 'reward_modifier'
        self.name = rule_data.get('name', 'UnknownRule')
        self.trigger = rule_data.get('trigger')  # Questo sarà il nome tradotto (es. 'vertical_bounce')
        self.confidence = rule_data.get('confidence', 1.0)
        self.priority = rule_data.get('priority', 1.0)
        self.reward_modifier = rule_data.get('reward_modifier', 0.0)

        # Mappa dei nomi dei trigger alle funzioni di check
        self.trigger_map = {
            'vertical_bounce': self._check_vertical_bounce,
            'side_bounce': self._check_side_bounce,
            'paddle_approaching_ball': self._check_paddle_approach, # Per reward shaping
            'brick_destroyed': self._check_brick_destroyed
        }
        
        if self.trigger not in self.trigger_map:
             print(f"ATTENZIONE: Trigger '{self.trigger}' non mappato.")


    def _check_vertical_bounce(self, state, next_state, action, reward) -> bool:
        """Verifica l'inversione di vy (colpo paddle/brick/muro alto).
           Corrisponde alla logica: Contact_With_Something_B o T -> vy(i+1) = -1 * vy(i)"""
        
        prev_vy, next_vy = state[3], next_state[3]
        
        # 1. Deve esserci un'inversione di segno
        is_bounce = (prev_vy * next_vy < 0)
        
        # 2. La velocità non deve essere troppo bassa (per escludere rumore)
        is_fast_enough = abs(prev_vy) > 0.05
        
        # 3. La palla deve essere in una posizione ragionevole (escludi rimbalzi esterni)
        is_relevant_pos = (state[1] > -0.9) and (state[1] < 0.9)
        
        return is_bounce and is_fast_enough and is_relevant_pos

    def _check_side_bounce(self, state, next_state, action, reward) -> bool:
        """Verifica l'inversione di vx (colpo su muro laterale).
           Corrisponde alla logica: Contact_With_Something_L o R -> vx(i+1) = -1 * vx(i)"""
        
        prev_vx, next_vx = state[2], next_state[2]
        
        # 1. Deve esserci un'inversione di segno
        is_bounce = (prev_vx * next_vx < 0)
        
        # 2. La palla deve essere vicina ai bordi laterali
        is_near_side = (abs(state[0]) > 0.9)
        
        return is_bounce and is_near_side

    def _check_paddle_approach(self, state, next_state, action, reward) -> bool:
        """Verifica se l'azione è un movimento predittivo corretto.
           Non è una regola fisica estratta, ma un'euristica di comportamento."""
        
        ball_x_norm, vy_norm, paddle_x_norm = state[0], state[3], state[4]
        next_paddle_x_norm = next_state[4]

        # Se la palla sta scendendo (vy > 0 o normalizzato positivo)
        if vy_norm > 0:
            # Calcola la distanza tra paddle e palla
            dist_prev = abs(paddle_x_norm - ball_x_norm)
            dist_next = abs(next_paddle_x_norm - ball_x_norm)
            
            # Se la nuova posizione della paddle è più vicina alla palla
            return dist_next < dist_prev
            
        return False # Non è rilevante se la palla sta salendo
        
    def _check_brick_destroyed(self, state, next_state, action, reward) -> bool:
        """Controlla se un brick è stato distrutto (reward positivo)"""
        return reward > 1.0 # Assumendo che il reward per un brick sia > 1

    def matches_state_transition(self, state, next_state, action, reward) -> bool:
        """Il metodo principale richiamato dal Replay Buffer e Reward Shaper."""
        check_func = self.trigger_map.get(self.trigger)
        if check_func:
            return check_func(state, next_state, action, reward)
        return False

class GodActPopulationIntegrator:

    # ... (altri metodi)

    def _extract_rules_from_population(self, population):
        rules = []
        
        # Mappa la Logica Pura Estratta al Trigger GodAct per l'analisi dello stato
        translation_map = {
            'Contact_With_Something_B': 'vertical_bounce', 
            'Contact_With_Something_T': 'vertical_bounce', 
            'Contact_With_Something_L': 'side_bounce',
            'Contact_With_Something_R': 'side_bounce',
        }

        # Iterazione sulla struttura del file di Core Knowledge (population.pkl)
        for ind_id, individual in population.items():
            if 'rules' in individual: # Cerchiamo il dizionario delle regole
                for obj_id, rule_list in individual['rules'].items():
                    for r in rule_list:
                        raw_trigger = r.get("causes", ["Unknown"])[0] # Estrai il Causa, es. 'Contact_With_Something_B'
                        
                        godact_trigger = translation_map.get(raw_trigger, raw_trigger)

                        if godact_trigger in self.godact_rule_names: # Controlla se è un trigger gestito
                            rules.append(GodActRule({
                                "name": r.get("name", "ExtractedRule"),
                                "trigger": godact_trigger, 
                                "confidence": 1.0, # Assumi alta fiducia nella regola fisica estratta
                                "priority": 5.0,   # Alto PER Priority per gli eventi fisici
                                "reward_modifier": 0.0 # Non deve modificare il reward, solo la priorità
                            }))
        
        # AGGIUNGI UNA REGOLA EURISTICA PER IL REWARD SHAPING (Azione Predittiva)
        # Questa regola non è estratta dalla fisica, ma migliora il training.
        rules.append(GodActRule({
            "name": "Predictive_Paddle_Move",
            "trigger": "paddle_approaching_ball",
            "confidence": 0.8,
            "priority": 1.0, # Priorità PER standard, è un reward shaping
            "reward_modifier": 0.3 # Piccolo bonus per il movimento corretto
        }))
        
        return rules