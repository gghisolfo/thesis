import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
from collections import deque, defaultdict
import re

from arkanoid_game import Game, grid_width, grid_height

BEST_POPULATION_PATH = "best_population.pkl"
SAVE_DIR = "./dqn/dqn_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# DYNAMIC SYMBOLIC RULE ANALYZER
# Analizza QUALSIASI set di regole simboliche senza assumere struttura fissa
# ============================================================================

class DynamicSymbolicAnalyzer:
    """
    Analizza dinamicamente le regole simboliche disponibili e estrae
    pattern utili SENZA assumere che certe regole esistano sempre.
    """
    
    def __init__(self, symbolic_rules_or_population):
        self.raw_rules = symbolic_rules_or_population
        self.analysis = self._analyze_available_rules()
    
    def _analyze_available_rules(self):
        """Analizza TUTTE le regole disponibili e categorizza per utilità strategica"""
        
        analysis = {
            'has_physics': False,
            'has_contacts': False,
            'has_movement': False,
            'velocity_changes': [],
            'position_changes': [],
            'contact_types': [],
            'predictability_score': 0.0,  # Quanto è prevedibile il sistema
            'rule_count': 0
        }
        
        if not self.raw_rules:
            print("⚠️  Nessuna regola simbolica disponibile - uso solo Q-learning")
            return analysis
        
        print(f"🔍 Analisi di {len(self.raw_rules)} regole simboliche...")
        
        for rule in self.raw_rules:
            rule_str = str(rule)
            analysis['rule_count'] += 1
            
            # Cerca pattern di velocità (vx, vy)
            if 'vx' in rule_str or 'vy' in rule_str:
                analysis['has_physics'] = True
                
                # Pattern: vy(i+1) = a * vy(i) + b
                velocity_pattern = re.findall(r'(vx|vy)\(i\+1\)\s*=\s*([-\d]+)\s*\*\s*(vx|vy)\(i\)\s*\+\s*([-\d]+)', rule_str)
                if velocity_pattern:
                    for match in velocity_pattern:
                        var, coef_a, _, coef_b = match
                        analysis['velocity_changes'].append({
                            'variable': var,
                            'coefficient_a': int(coef_a),
                            'coefficient_b': int(coef_b),
                            'is_bounce': int(coef_a) == -1
                        })
            
            # Cerca pattern di posizione (pos_x, pos_y)
            if 'pos_x' in rule_str or 'pos_y' in rule_str:
                analysis['has_movement'] = True
                
                position_pattern = re.findall(r'(pos_x|pos_y)\(i\+1\)\s*=\s*([-\d]+)\s*\*\s*(pos_x|pos_y)\(i\)\s*\+\s*([-\d]+)', rule_str)
                if position_pattern:
                    for match in position_pattern:
                        var, coef_a, _, coef_b = match
                        analysis['position_changes'].append({
                            'variable': var,
                            'coefficient_a': int(coef_a),
                            'coefficient_b': int(coef_b)
                        })
            
            # Cerca contatti
            if 'Contact' in rule_str:
                analysis['has_contacts'] = True
                
                # Estrai tipi di contatto
                contact_types = re.findall(r'Contact_With_Something_([A-Z])', rule_str)
                analysis['contact_types'].extend(contact_types)
        
        # Calcola score di predicibilità
        if analysis['has_physics'] and analysis['has_contacts']:
            analysis['predictability_score'] = 0.8
        elif analysis['has_physics']:
            analysis['predictability_score'] = 0.5
        elif len(self.raw_rules) > 5:
            analysis['predictability_score'] = 0.3
        else:
            analysis['predictability_score'] = 0.1
        
        # Rimuovi duplicati
        analysis['contact_types'] = list(set(analysis['contact_types']))
        
        print(f"✅ Analisi completata:")
        print(f"   - Fisica scoperta: {analysis['has_physics']}")
        print(f"   - Contatti trovati: {len(analysis['contact_types'])}")
        print(f"   - Cambi velocità: {len(analysis['velocity_changes'])}")
        print(f"   - Score predicibilità: {analysis['predictability_score']:.2f}")
        
        return analysis
    
    def can_predict_physics(self):
        """Possiamo fare predizioni fisiche?"""
        return self.analysis['predictability_score'] > 0.4
    
    def get_bounce_rules(self):
        """Estrae regole di rimbalzo se disponibili"""
        return [v for v in self.analysis['velocity_changes'] if v['is_bounce']]
    
    def get_movement_rules(self):
        """Estrae regole di movimento se disponibili"""
        return self.analysis['position_changes']


# ============================================================================
# ADAPTIVE STRATEGIC RULE GENERATOR
# Genera SOLO le regole possibili dato il set simbolico disponibile
# ============================================================================

class AdaptiveStrategicRuleGenerator:
    """
    Genera regole strategiche ADATTIVE basate su QUALSIASI
    set di regole simboliche disponibili.
    """
    
    def __init__(self, analyzer, grid_width, grid_height):
        self.analyzer = analyzer
        self.grid_width = grid_width
        self.grid_height = grid_height
    
    def generate_rules(self):
        """Genera solo le regole possibili con le info disponibili"""
        rules = []
        
        analysis = self.analyzer.analysis
        
        # ========== REGOLA BASE: Track Ball (SEMPRE disponibile) ==========
        # Non richiede regole simboliche, usa solo osservazioni
        rules.append(self._create_basic_tracking_rule())
        
        # ========== REGOLA: Centered Defense (SEMPRE disponibile) ==========
        rules.append(self._create_defensive_rule())
        
        # ========== REGOLE CONDIZIONALI: Solo se abbiamo info fisica ==========
        if self.analyzer.can_predict_physics():
            print("🧠 Fisica predicibile → aggiungo regole avanzate")
            
            # Se abbiamo regole di rimbalzo, possiamo anticipare
            if self.analyzer.get_bounce_rules():
                rules.append(self._create_bounce_anticipation_rule())
            
            # Se abbiamo regole di movimento, possiamo intercettare meglio
            if self.analyzer.get_movement_rules():
                rules.append(self._create_interception_rule())
        
        else:
            print("⚠️  Fisica non predicibile → uso solo regole base")
        
        # ========== REGOLA: Emergency (SEMPRE disponibile) ==========
        rules.append(self._create_emergency_rule())
        
        return rules
    
    def _create_basic_tracking_rule(self):
        """Regola base: segui la palla (sempre possibile)"""
        def condition(state):
            ball_y = (state[1] + 1) / 2  # Denormalizza
            return ball_y > 0.4  # Palla nella metà bassa
        
        def action(state):
            ball_x = (state[0] + 1) / 2 * self.grid_width
            paddle_x = (state[4] + 1) / 2 * self.grid_width
            diff = ball_x - paddle_x
            
            if abs(diff) < 5:
                return 1  # Stay
            return 0 if diff < 0 else 2
        
        return StrategicRule(
            name="Basic Ball Tracking",
            condition_fn=condition,
            action_fn=action,
            confidence=0.6,
            priority=2,
            requires_physics=False
        )
    
    def _create_defensive_rule(self):
        """Posizione difensiva centrale quando palla lontana"""
        def condition(state):
            ball_y = (state[1] + 1) / 2
            return ball_y < 0.4  # Palla in alto
        
        def action(state):
            paddle_x = (state[4] + 1) / 2 * self.grid_width
            center = self.grid_width / 2
            diff = center - paddle_x
            
            if abs(diff) < 8:
                return 1
            return 0 if diff < 0 else 2
        
        return StrategicRule(
            name="Defensive Centering",
            condition_fn=condition,
            action_fn=action,
            confidence=0.5,
            priority=1,
            requires_physics=False
        )
    
    def _create_bounce_anticipation_rule(self):
        """Anticipa rimbalzi (solo se abbiamo regole di bounce)"""
        bounce_rules = self.analyzer.get_bounce_rules()
        
        def condition(state):
            ball_y = (state[1] + 1) / 2
            vy = state[3]
            return ball_y < 0.5 and vy < 0  # Va verso l'alto
        
        def action(state):
            # Simula rimbalzo usando regole scoperte
            ball_x = (state[0] + 1) / 2 * self.grid_width
            ball_y = (state[1] + 1) / 2 * self.grid_height
            vx = state[2] * 10.0
            vy = state[3] * 10.0
            paddle_x = (state[4] + 1) / 2 * self.grid_width
            
            # Simula fino al rimbalzo superiore
            for _ in range(50):
                ball_y += vy
                ball_x += vx
                
                # Rimbalzo superiore
                if ball_y <= 3:
                    vy = -vy  # Usa regola scoperta
                    break
                
                # Rimbalzi laterali
                if ball_x <= 3 or ball_x >= self.grid_width - 3:
                    vx = -vx
            
            # Ora simula discesa
            for _ in range(50):
                ball_y += vy
                ball_x += vx
                
                if ball_y >= 0.6 * self.grid_height:
                    # Qui sarà quando torna giù
                    diff = ball_x - paddle_x
                    if abs(diff) < 5:
                        return 1
                    return 0 if diff < 0 else 2
            
            return 1  # Default
        
        return StrategicRule(
            name="Physics-Based Anticipation",
            condition_fn=condition,
            action_fn=action,
            confidence=0.85,
            priority=3,
            requires_physics=True
        )
    
    def _create_interception_rule(self):
        """Intercettazione precisa (se abbiamo regole di movimento)"""
        def condition(state):
            ball_y = (state[1] + 1) / 2
            vy = state[3]
            return ball_y > 0.5 and vy > 0  # Scende verso paddle
        
        def action(state):
            ball_x = (state[0] + 1) / 2 * self.grid_width
            ball_y = (state[1] + 1) / 2 * self.grid_height
            vx = state[2] * 10.0
            vy = state[3] * 10.0
            paddle_x = (state[4] + 1) / 2 * self.grid_width
            paddle_y = 60
            
            # Predici intersezione
            if vy == 0:
                return 1
            
            frames_to_paddle = (paddle_y - ball_y) / vy
            if frames_to_paddle < 0:
                return 1
            
            intercept_x = ball_x + vx * frames_to_paddle
            
            # Considera rimbalzi laterali
            while intercept_x < 3:
                intercept_x = 6 - intercept_x
            while intercept_x > self.grid_width - 3:
                intercept_x = 2 * (self.grid_width - 3) - intercept_x
            
            diff = intercept_x - paddle_x
            if abs(diff) < 3:
                return 1
            return 0 if diff < 0 else 2
        
        return StrategicRule(
            name="Precise Interception",
            condition_fn=condition,
            action_fn=action,
            confidence=0.9,
            priority=4,
            requires_physics=True
        )
    
    def _create_emergency_rule(self):
        """Emergenza: palla vicinissima"""
        def condition(state):
            ball_y = (state[1] + 1) / 2
            vy = state[3]
            return ball_y > 0.8 and vy > 0
        
        def action(state):
            ball_x = (state[0] + 1) / 2 * self.grid_width
            paddle_x = (state[4] + 1) / 2 * self.grid_width
            diff = ball_x - paddle_x
            
            if abs(diff) < 2:
                return 1
            return 0 if diff < 0 else 2
        
        return StrategicRule(
            name="Emergency Response",
            condition_fn=condition,
            action_fn=action,
            confidence=1.0,
            priority=5,
            requires_physics=False
        )


# ============================================================================
# STRATEGIC RULE (con flag requires_physics)
# ============================================================================

class StrategicRule:
    def __init__(self, name, condition_fn, action_fn, confidence=1.0, priority=1, requires_physics=False):
        self.name = name
        self.condition_fn = condition_fn
        self.action_fn = action_fn
        self.confidence = confidence
        self.priority = priority
        self.requires_physics = requires_physics  # Indica se serve fisica simbolica
    
    def applies(self, state):
        return self.condition_fn(state)
    
    def get_action(self, state):
        return self.action_fn(state)
    
    def __repr__(self):
        physics_marker = "🔬" if self.requires_physics else "🎯"
        return f"{physics_marker} {self.name} (conf={self.confidence:.2f}, pri={self.priority})"


# ============================================================================
# ADAPTIVE ACTION SELECTOR
# ============================================================================

class AdaptiveRuleSelector:
    """Selettore che si adatta alle regole disponibili"""
    
    def __init__(self, rules, initial_rule_prob=0.6):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self.rule_prob = initial_rule_prob
        self.stats = defaultdict(int)
        
        # Conta quante regole richiedono fisica
        self.physics_rules = [r for r in rules if r.requires_physics]
        self.basic_rules = [r for r in rules if not r.requires_physics]
        
        print(f"\n📋 Regole strategiche caricate:")
        print(f"   - Regole base: {len(self.basic_rules)}")
        print(f"   - Regole fisiche: {len(self.physics_rules)}")
    
    def select_action(self, state, q_values, epsilon):
        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, len(q_values) - 1), "random"
        
        # Prova regole
        if random.random() < self.rule_prob:
            for rule in self.rules:
                if rule.applies(state):
                    action = rule.get_action(state)
                    self.stats[rule.name] += 1
                    return action, rule.name
        
        # Q-network
        return int(q_values.argmax()), "q_network"
    
    def decay_rule_prob(self, decay_rate=0.99):
        """Riduci probabilità regole man mano che Q-network migliora"""
        self.rule_prob = max(0.2, self.rule_prob * decay_rate)
    
    def print_stats(self):
        print("\n📊 Rule Usage Statistics:")
        total = sum(self.stats.values())
        if total == 0:
            print("   Nessuna regola usata")
            return
        
        for rule_name, count in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100)
            print(f"   {rule_name}: {count} ({pct:.1f}%)")


# ============================================================================
# PRIORITIZED BUFFER (invariato)
# ============================================================================

class AdaptiveGuidedBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, rule_used=None):
        priority = 1.0
        
        if rule_used and rule_used not in ["random", "q_network"]:
            priority += 1.5
        
        if reward > 5:
            priority += 1.0
        
        if abs(reward) > 40:
            priority += 2.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), actions, rewards, np.array(next_states), dones)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# ARKANOID ENV (invariato)
# ============================================================================

class ArkanoidEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.done = False
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y
    
    def reset(self):
        self.game = Game()
        self._prev_bricks_alive = self.game.bricks_alive
        self._prev_ball_y = self.game.ball_y
        self.done = False
        return self._get_obs()
    
    def step(self, action):
        prev_bricks = self.game.bricks_alive
        prev_ball_y = self.game.ball_y
        
        if action == 0:
            self.game.set_paddle_speed(-1)
        elif action == 2:
            self.game.set_paddle_speed(1)
        else:
            self.game.set_paddle_speed(0)
        
        self.game.update()
        reward = self._compute_reward(prev_bricks, prev_ball_y)
        
        if self.game.bricks_alive == 0:
            self.done = True
            reward += 100.0
            print("🎉 VITTORIA!")
        
        if self.game.ball_lost or self.game.ball_y + self.game.ball_radius >= grid_height - 3:
            self.done = True
            reward -= 50.0
        
        return self._get_obs(), reward, self.done, {}
    
    def _get_obs(self):
        ball_x = self.game.ball_x / grid_width
        ball_y = self.game.ball_y / grid_height
        vx = self.game.ball_speed_x / 10.0
        vy = self.game.ball_speed_y / 10.0
        paddle_x = self.game.paddle_x / grid_width
        return np.array([ball_x*2-1, ball_y*2-1, vx, vy, paddle_x*2-1], dtype=np.float32)
    
    def _compute_reward(self, prev_bricks, prev_ball_y):
        r = 0.0
        if self.game.bricks_alive < prev_bricks:
            destroyed = prev_bricks - self.game.bricks_alive
            r += 10.0 * destroyed
        if self._check_ball_hits_paddle(prev_ball_y):
            r += 2.0
        distance_to_paddle = abs(self.game.ball_x - self.game.paddle_x)
        if distance_to_paddle < 10:
            r += 0.1
        if self.game.ball_y > grid_height - 15:
            r -= 0.2
        return r
    
    def _check_ball_hits_paddle(self, prev_ball_y):
        ball = self.game.elements['ball']
        paddle = self.game.elements['paddle_center']
        if self.game.ball_speed_y < 0 and prev_ball_y > self.game.ball_y:
            if abs(self.game.ball_y - paddle['pos_y']) < 3:
                overlap_x = (
                    (ball['hitbox_br_x'] >= paddle['hitbox_tl_x']) and
                    (ball['hitbox_tl_x'] <= paddle['hitbox_br_x'])
                )
                if overlap_x:
                    return True
        return False


# ============================================================================
# Q-NETWORK (invariata)
# ============================================================================

class QNetwork(nn.Module):
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


# ============================================================================
# ADAPTIVE TRAINING
# ============================================================================

def train_dqn_adaptive(total_episodes=1000, max_steps=2000):
    """
    Training DQN ADATTIVO:
    - Se trova regole simboliche utili → le usa
    - Se non le trova → procede con Q-learning puro
    - Si adatta dinamicamente a QUALSIASI set di regole
    """
    
    print("="*70)
    print("🚀 TRAINING DQN ADATTIVO CON REGOLE SIMBOLICHE DINAMICHE")
    print("="*70)
    
    # 1. Carica popolazione (che potrebbe contenere regole simboliche)
    print("\n📚 Caricamento popolazione...")
    try:
        with open(BEST_POPULATION_PATH, "rb") as f:
            population = pickle.load(f)
        print(f"✅ Caricata popolazione con {len(population)} individui")
    except:
        print("⚠️  Popolazione non trovata, uso approccio base")
        population = []
    
    # 2. Analizza dinamicamente regole simboliche disponibili
    analyzer = DynamicSymbolicAnalyzer(population)
    
    # 3. Genera SOLO le regole possibili
    generator = AdaptiveStrategicRuleGenerator(analyzer, grid_width, grid_height)
    strategic_rules = generator.generate_rules()
    
    print(f"\n✅ Generate {len(strategic_rules)} regole strategiche:")
    for rule in strategic_rules:
        print(f"   {rule}")
    
    # 4. Setup training
    env = ArkanoidEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    buffer = AdaptiveGuidedBuffer(capacity=50000)
    selector = AdaptiveRuleSelector(rules=strategic_rules, initial_rule_prob=0.6)
    
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_target.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995
    
    rewards_history = []
    
    print("\n🎮 Inizio training...")
    print("="*70)
    
    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = q_net(s_t)[0]
            
            action, rule_name = selector.select_action(state, q_vals, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            buffer.push(state, action, reward, next_state, done, rule_used=rule_name)
            
            # Training
            if len(buffer) >= 64:
                s, a, r, ns, d = buffer.sample(64)
                
                s_t = torch.tensor(s, device=device)
                a_t = torch.tensor(a, device=device).unsqueeze(1)
                r_t = torch.tensor(r, device=device)
                ns_t = torch.tensor(ns, device=device)
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
        
        # Riduci gradualmente dipendenza da regole
        if ep % 50 == 0:
            selector.decay_rule_prob()
        
        rewards_history.append(total_reward)
        
        if ep % 10 == 0:
            q_target.load_state_dict(q_net.state_dict())
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Ep {ep:4d} | Reward: {total_reward:7.2f} | Avg(10): {avg_reward:7.2f} | "
                  f"ε: {epsilon:.3f} | RuleProb: {selector.rule_prob:.3f}")
    
    # Salva modello
    final_path = os.path.join(SAVE_DIR, "dqn_adaptive_symbolic.pth")
    torch.save(q_net.state_dict(), final_path)
    print(f"\n✅ Training completato! Modello salvato in {final_path}")
    
    selector.print_stats()
    
    return rewards_history, strategic_rules, analyzer


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    rewards, rules, analyzer = train_dqn_adaptive(total_episodes=1000)
    
    print("\n" + "="*70)
    print("📊 RISULTATI FINALI")
    print("="*70)
    print(f"Reward medio (ultimi 100 ep): {np.mean(rewards[-100:]):.2f}")
    print(f"Reward massimo: {np.max(rewards):.2f}")
    print(f"\n🎯 Predicibilità fisica: {analyzer.analysis['predictability_score']:.2%}")
    print(f"🎯 Regole strategiche usate: {len(rules)}")