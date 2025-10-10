"""
Convertitore: Individual.pkl (Core Knowledge) → DQN Rules JSON
Legge gli atti di Dio dal file pickle e li trasforma in regole per la DQN

Input: best_individual.pkl (tuo formato)
Output: god_acts_rules.json (formato DQN)
"""

import pickle
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path


class IndividualToRulesConverter:
    """
    Converte Individual objects (Core Knowledge) in regole per DQN
    """
    
    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        self.individual = None
        self.rules = []
        self.god_acts_stats = defaultdict(int)
    
    def load_individual(self):
        """Carica l'Individual dal file pickle"""
        with open(self.pkl_path, 'rb') as f:
            self.individual = pickle.load(f)
        print(f"✓ Caricato Individual da {self.pkl_path}")
        return self.individual
    
    def analyze_unexplained_events(self):
        """Analizza tutti gli unexplained events per identificare pattern"""
        print("\n=== Analisi Atti di Dio ===")
        
        for obj_id, obj in self.individual.object_dict.items():
            obj_type = self._get_object_type(obj)
            
            # Analizza unexplained per ogni frame
            for frame_id in obj.frames_id:
                if frame_id in obj.unexplained and obj.unexplained[frame_id]:
                    unexplained_list = obj.unexplained[frame_id]
                    
                    for unexplained in unexplained_list:
                        god_act = self._classify_god_act(
                            obj_id, obj_type, frame_id, unexplained, obj
                        )
                        if god_act:
                            self.god_acts_stats[god_act['type']] += 1
                            print(f"  Frame {frame_id}, {obj_type}: {god_act['type']}")
        
        self._print_statistics()
    
    def _get_object_type(self, obj) -> str:
        """Estrae il tipo di oggetto dalla descrizione del primo patch"""
        if obj.sequence and len(obj.sequence) > 0:
            return obj.sequence[0].description
        return "unknown"
    
    def _classify_god_act(self, obj_id: int, obj_type: str, frame_id: int, 
                         unexplained, obj) -> Dict:
        """Classifica un unexplained event in un tipo di God Act"""
        
        # Estrai property change
        property_name = unexplained.property_class.__name__ if hasattr(unexplained.property_class, '__name__') else str(unexplained.property_class)
        prev_val = unexplained.previous_value
        final_val = unexplained.final_value
        
        # Ball velocity changes (collisioni)
        if obj_type == "ball":
            if "Speed" in property_name or "vx" in property_name.lower() or "vy" in property_name.lower():
                # Inversione velocità = collisione
                if self._is_velocity_inversion(prev_val, final_val):
                    collision_type = self._infer_collision_type(obj, frame_id)
                    return {
                        'type': collision_type,
                        'object_type': obj_type,
                        'frame_id': frame_id,
                        'property': property_name,
                        'magnitude': abs(final_val - prev_val)
                    }
            
            # Position jump (teleport - potrebbe essere bounce estremo)
            if "Pos" in property_name:
                distance = abs(final_val - prev_val)
                if distance > 5:  # Soglia per "salto" sospetto
                    return {
                        'type': 'ball_teleport',
                        'object_type': obj_type,
                        'frame_id': frame_id,
                        'property': property_name,
                        'magnitude': distance
                    }
        
        # Paddle position changes (input utente o AI)
        if obj_type == "paddle_center":
            if "Pos" in property_name:
                return {
                    'type': 'paddle_movement',
                    'object_type': obj_type,
                    'frame_id': frame_id,
                    'property': property_name,
                    'magnitude': abs(final_val - prev_val)
                }
        
        # Brick disappearance
        if "brick" in obj_type.lower():
            # Se un brick scompare (non più presente nei frame successivi)
            if frame_id + 1 in obj.frames_id:
                next_frame_present = (frame_id + 1) in obj.sequence_dict
                if not next_frame_present:
                    return {
                        'type': 'brick_destroyed',
                        'object_type': obj_type,
                        'frame_id': frame_id,
                        'property': 'existence',
                        'magnitude': 1.0
                    }
        
        return None
    
    def _is_velocity_inversion(self, prev_val, final_val) -> bool:
        """Controlla se c'è inversione di velocità (segno opposto)"""
        if prev_val == 0 or final_val == 0:
            return False
        return (prev_val * final_val) < 0
    
    def _infer_collision_type(self, obj, frame_id: int) -> str:
        """Inferisce il tipo di collisione dal contesto"""
        # Ottieni posizione della palla al frame
        if frame_id in obj.sequence_dict:
            patch = obj.sequence_dict[frame_id]
            ball_y = patch.properties.get('Pos_y', patch.properties.get('pos_y', 50))
            
            # Zone del campo
            if ball_y > 60:  # Zona paddle (bottom)
                return 'ball_paddle_collision'
            elif ball_y < 20:  # Zona bricks (top)
                return 'ball_brick_collision'
            elif ball_y < 5 or ball_y > 65:  # Zona wall
                return 'ball_wall_collision'
        
        return 'ball_collision_unknown'
    
    def _print_statistics(self):
        """Stampa statistiche sugli atti di Dio trovati"""
        print("\n--- Statistiche God Acts ---")
        total = sum(self.god_acts_stats.values())
        for god_act_type, count in sorted(self.god_acts_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {god_act_type:30s}: {count:3d} ({percentage:5.1f}%)")
        print(f"  {'TOTALE':30s}: {total:3d}")
    
    def generate_rules(self) -> List[Dict]:
        """Genera regole DQN dagli atti di Dio identificati"""
        rules = []
        
        # Priorità basate su frequenza e importanza
        rule_priorities = {
            'ball_paddle_collision': 3.0,
            'ball_brick_collision': 2.5,
            'ball_wall_collision': 1.5,
            'brick_destroyed': 2.0,
            'paddle_movement': 1.0,
            'ball_teleport': 2.5,
        }
        
        # Reward modifiers
        reward_modifiers = {
            'ball_paddle_collision': 2.0,  # Molto importante non perdere
            'ball_brick_collision': 3.0,   # Obiettivo del gioco
            'ball_wall_collision': 1.0,
            'brick_destroyed': 3.0,
            'paddle_movement': 1.2,
            'ball_teleport': 1.5,
        }
        
        # Confidence basata su frequenza osservata
        total_events = sum(self.god_acts_stats.values())
        
        rule_id = 1
        for god_act_type, count in self.god_acts_stats.items():
            if count > 0:
                confidence = min(0.95, count / total_events * 3.0)  # Max 0.95
                
                rule = {
                    'rule_id': f'R{rule_id}',
                    'trigger': god_act_type,
                    'effect': self._get_effect_description(god_act_type),
                    'confidence': round(confidence, 2),
                    'priority': rule_priorities.get(god_act_type, 1.0),
                    'reward_modifier': reward_modifiers.get(god_act_type, 1.0),
                    'observed_count': count,
                    'conditions': self._get_conditions(god_act_type)
                }
                
                rules.append(rule)
                rule_id += 1
        
        self.rules = rules
        return rules
    
    def _get_effect_description(self, god_act_type: str) -> str:
        """Descrizione dell'effetto di ogni tipo di God Act"""
        effects = {
            'ball_paddle_collision': 'velocity_invert_y_and_paddle_influence',
            'ball_brick_collision': 'brick_destroy_and_velocity_invert',
            'ball_wall_collision': 'velocity_invert_x',
            'brick_destroyed': 'object_disappear_and_score_increase',
            'paddle_movement': 'position_change_by_input',
            'ball_teleport': 'position_discontinuity',
        }
        return effects.get(god_act_type, 'unknown_effect')
    
    def _get_conditions(self, god_act_type: str) -> Dict:
        """Condizioni specifiche per riconoscere ogni God Act"""
        conditions = {
            'ball_paddle_collision': {
                'ball_y_range': [0.8, 1.0],
                'velocity_inversion': 'vy',
            },
            'ball_brick_collision': {
                'ball_y_range': [0.05, 0.3],
                'velocity_inversion': 'vy',
                'reward_positive': True,
            },
            'ball_wall_collision': {
                'ball_x_range': [[0.0, 0.05], [0.95, 1.0]],
                'velocity_inversion': 'vx',
            },
            'brick_destroyed': {
                'object_disappear': True,
                'reward_positive': True,
            },
            'paddle_movement': {
                'object_type': 'paddle',
                'position_change': True,
            },
        }
        return conditions.get(god_act_type, {})
    
    def save_rules_json(self, output_path: str = 'god_acts_rules.json'):
        """Salva le regole in formato JSON per la DQN"""
        output_data = {
            'metadata': {
                'source_file': self.pkl_path,
                'total_god_acts': sum(self.god_acts_stats.values()),
                'god_acts_breakdown': dict(self.god_acts_stats),
                'num_rules': len(self.rules),
            },
            'rules': self.rules
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Regole salvate in: {output_path}")
        return output_path
    
    def visualize_rules(self):
        """Visualizza le regole generate in modo leggibile"""
        print("\n=== Regole Generate per DQN ===\n")
        
        for rule in self.rules:
            print(f"[{rule['rule_id']}] {rule['trigger']}")
            print(f"  → Effetto: {rule['effect']}")
            print(f"  → Confidence: {rule['confidence']:.2f} | Priority: {rule['priority']:.1f} | Reward Mod: {rule['reward_modifier']:.1f}")
            print(f"  → Osservato: {rule['observed_count']} volte")
            if rule['conditions']:
                print(f"  → Condizioni: {rule['conditions']}")
            print()


# ============================================================================
# INTEGRAZIONE CON DQN ESISTENTE
# ============================================================================

def integrate_with_existing_dqn(rules_json_path: str, env, dqn_agent=None):
    """
    Integra le regole estratte con una DQN esistente.
    Usa il modulo godacts-dqn-integration.
    """
    try:
        from godacts_dqn_integration import GodActDQNIntegrator
        
        print("\n=== Integrazione con DQN ===")
        integrator = GodActDQNIntegrator(rules_json_path)
        
        # Wrappa environment
        enhanced_env = integrator.wrap_environment(env)
        
        # Crea replay buffer prioritizzato
        replay_buffer = integrator.create_replay_buffer(capacity=50000)
        
        print("✓ Environment wrappato con God Acts knowledge")
        print("✓ Replay buffer prioritizzato creato")
        
        return enhanced_env, replay_buffer, integrator
    
    except ImportError:
        print("⚠ Modulo 'godacts_dqn_integration' non trovato.")
        print("  Le regole sono state generate, ma serve il modulo di integrazione.")
        return env, None, None


# ============================================================================
# MAIN: CONVERSIONE COMPLETA
# ============================================================================

def convert_pkl_to_dqn_rules(pkl_path: str, output_json: str = 'god_acts_rules.json'):
    """Pipeline completa: PKL → Analisi → Regole → JSON"""
    
    print("=" * 70)
    print("CONVERSIONE: Core Knowledge Individual → DQN Rules")
    print("=" * 70)
    
    # 1. Carica Individual
    converter = IndividualToRulesConverter(pkl_path)
    converter.load_individual()
    
    # 2. Analizza atti di Dio
    converter.analyze_unexplained_events()
    
    # 3. Genera regole
    rules = converter.generate_rules()
    
    # 4. Visualizza regole
    converter.visualize_rules()
    
    # 5. Salva JSON
    output_path = converter.save_rules_json(output_json)
    
    print("\n" + "=" * 70)
    print(f"✓ CONVERSIONE COMPLETATA")
    print(f"  Regole pronte per l'uso in: {output_path}")
    print("=" * 70)
    
    return output_path, converter


# ============================================================================
# ESEMPIO DI UTILIZZO COMPLETO
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Path al tuo file pickle
    pkl_file = '1760088637384_best_individual.pkl'
    
    if not Path(pkl_file).exists():
        print(f"⚠ File non trovato: {pkl_file}")
        print("Uso: python pkl_to_dqn_converter.py [path_to_pkl_file]")
        if len(sys.argv) > 1:
            pkl_file = sys.argv[1]
        else:
            sys.exit(1)
    
    # Converti PKL → JSON
    rules_json, converter = convert_pkl_to_dqn_rules(pkl_file)
    
    print("\n" + "=" * 70)
    print("PROSSIMI PASSI:")
    print("=" * 70)
    print("""
1. Hai generato il file: god_acts_rules.json

2. Ora puoi integrare con la tua DQN:

   from godacts_dqn_integration import GodActDQNIntegrator
   from dqn_arkanoid_pygame import ArkanoidEnv
   
   # Carica regole
   integrator = GodActDQNIntegrator('god_acts_rules.json')
   
   # Wrappa environment
   env = ArkanoidEnv()
   env = integrator.wrap_environment(env)
   
   # Usa replay buffer prioritizzato
   replay_buffer = integrator.create_replay_buffer(50000)
   
   # Training normale, ma accelerato dalle regole!
   # ... il tuo codice DQN ...

3. Sperimenta con i parametri delle regole in god_acts_rules.json
   per ottimizzare il training!
    """)
    
    # Test: verifica che il JSON sia valido
    print("\n--- Verifica JSON ---")
    with open(rules_json, 'r') as f:
        data = json.load(f)
        print(f"✓ JSON valido: {len(data['rules'])} regole caricate")
        print(f"✓ Metadata: {data['metadata']}")