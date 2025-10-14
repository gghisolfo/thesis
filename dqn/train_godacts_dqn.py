import sys
import os

# Otteniamo la cartella root "thesis" e la aggiungiamo al path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# Ora possiamo importare i moduli
from arkanoid_game import Game

from components.traduzione.godAct.god_act_core import GodActDQNIntegrator


RULES_PATH = "god_acts_rules.json"

# ---------------------------
# Arkanoid Gym Environment
# ---------------------------



# ---------------------------
# Q-Network (simple DQN)
# ---------------------------
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

# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train(seed=0, total_episodes=400, render_interval=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not os.path.exists(RULES_PATH):
        rules = {
            "rules": [
                {"rule_id": "R1", "trigger": "ball_paddle_collision", "effect": "bounce", "confidence": 0.9, "priority": 2.0},
                {"rule_id": "R2", "trigger": "ball_brick_collision", "effect": "destroy_brick", "confidence": 0.8, "priority": 1.5},
                {"rule_id": "R3", "trigger": "ball_lost", "effect": "lose_life", "confidence": 1.0, "priority": 2.5}
            ]
        }
        with open(RULES_PATH, 'w') as f:
            json.dump(rules, f, indent=2)

    integrator = GodActDQNIntegrator(RULES_PATH)
    env = ArkanoidEnv()
    env = integrator.wrap_environment(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = QNetwork(state_dim, action_dim).to(device)
    q_target = QNetwork(state_dim, action_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = integrator.create_replay_buffer(50000)

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.995
    total_steps = 0

    for ep in range(1, total_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            total_steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    qvals = q_net(s)
                    action = int(torch.argmax(qvals).item())

            next_state, reward, done, _ = env.step(action)
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(replay) > batch_size:
                s, a, r, ns, d = replay.sample(batch_size)
                s = torch.tensor(s).to(device)
                a = torch.tensor(a).unsqueeze(1).to(device)
                r = torch.tensor(r).to(device)
                ns = torch.tensor(ns).to(device)
                d = torch.tensor(d).float().to(device)

                with torch.no_grad():
                    target_q = q_target(ns).max(1)[0]
                    td_target = r + gamma * (1 - d) * target_q

                current_q = q_net(s).gather(1, a).squeeze(1)
                loss = nn.functional.mse_loss(current_q, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % 1000 == 0:
                q_target.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"[Ep {ep}] Reward={ep_reward:.2f} ε={epsilon:.3f}")

        if render_interval and ep % render_interval == 0:
            env.render()

    torch.save(q_net.state_dict(), "dqn_arkanoid_godacts_final.pth")
    print("✅ Training completo. Modello salvato.")
    env.close()

if __name__ == "__main__":
    train(seed=1, total_episodes=400, render_interval=25)
