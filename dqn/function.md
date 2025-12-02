
class GenericEventDetector:
    def __init__(self,
                 threshold_for_change: float = 0.3,
                 min_relative_change: float = 0.03) -> None

    def detect_events(self, prev_state: Dict, curr_state: Dict) -> List[Dict]___ -------> new name _detect_change
    def _detect_value_change(self, key: str, prev: Any, curr: Any) -> Dict | None x
    def _detect_sign_flip(self, key: str, prev: Any, curr: Any) -> Dict | None x
    def _detect_threshold_cross(self, key: str, prev: Any, curr: Any) -> Dict | None x
    def _detect_discrete_change(self, key: str, prev: Any, curr: Any) -> Dict | None x


class CausalEventChainTracker:
    def __init__(self,
                 causal_window: int = 3,
                 base_reward: float = 1.0,
                 chain_exponent: float = 1.2) -> None

    def reset(self) -> None
    def add_events(self, events: List[Dict]) -> None
    def _calculate_chain_reward(self, chain_length: int) -> float
    def get_total_reward(self) -> float
    def get_statistics(self) -> Dict


class GenericSymbolicEnv(gym.Env):
    def __init__(self,
                 sim_object,
                 action_map: Dict[int, Callable],
                 observation_extractor: Callable = None,
                 termination_check: Callable = None,
                 causal_window: int = 3,
                 chain_exponent: float = 1.5) -> None

    def reset(self) -> np.ndarray
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]
    def _default_obs_extractor(self, sim) -> np.ndarray


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim) -> None
    def forward(self, x) -> torch.Tensor


# funzioni di utilità / factory / training
def create_arkanoid_env() -> GenericSymbolicEnv

def train_generic_dqn(env_factory: Callable,
                      total_episodes: int = 10000,
                      max_steps: int = 3000) -> Tuple[List[float], List[Dict], List[float]]
