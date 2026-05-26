from dataclasses import dataclass
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


@dataclass
class IQLAgentConfig:
    alpha: float = 0.1          # tasa de aprendizaje
    gamma: float = 1.0          # factor de descuento (1.0 = sin descuento; episodios finitos)
    min_epsilon: float = 0.01   # piso de exploracion
    max_t: int = 1000           # horizonte de decaimiento de epsilon (en pasos)
    seed: int | None = None


class IQLAgent(Agent):
    """Independent Q-Learning: cada agente corre su propio Q-learning tabular,
    tratando a los demas agentes como parte del entorno (no los modela)."""

    def __init__(self, game: SimultaneousGame, agent: AgentID, config: IQLAgentConfig | None = None) -> None:
        super().__init__(game=game, agent=agent)
        self.config = config if config is not None else IQLAgentConfig()
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.min_epsilon = self.config.min_epsilon
        self.max_t = self.config.max_t
        self.rng = np.random.default_rng(self.config.seed)

        self.n_actions = self.game.num_actions(self.agent)
        # Q-table como diccionario: clave_estado -> vector de valores por accion.
        # Se puebla sola; no hace falta predefinir el tamaño del espacio de estados.
        self.Q: dict = {}

        self.learn = True   # en False: sin exploracion ni actualizacion (modo evaluacion)
        self.t = 0          # contador global de pasos de aprendizaje (para decaer epsilon)
        self.last_state = None
        self.last_action = None

    def _key(self, obs):
        # Clave hasheable del estado. Observacion None (juegos one-shot) -> clave fija.
        if obs is None:
            return None
        return tuple(np.asarray(obs).flatten().tolist())

    def _q(self, key) -> np.ndarray:
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        return self.Q[key]

    @property
    def epsilon(self) -> float:
        if not self.learn:
            return 0.0
        # Decaimiento lineal de 1.0 a min_epsilon a lo largo de max_t pasos.
        frac = max(0.0, (self.max_t - self.t) / self.max_t)
        return self.min_epsilon + (1.0 - self.min_epsilon) * frac

    def reset(self) -> None:
        # Inicio de episodio: olvidamos la transicion previa, pero NO la Q-table.
        self.last_state = None
        self.last_action = None

    def action(self):
        key = self._key(self.game.observe(self.agent))
        q = self._q(key)
        if self.learn and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(self.n_actions))   # explora
        else:
            a = int(np.argmax(q))                         # explota
        self.last_state = key
        self.last_action = a
        return a

    def update(self) -> None:
        if not self.learn or self.last_action is None:
            return
        reward = self.game.reward(self.agent)
        done = self.game.terminations[self.agent] or self.game.truncations[self.agent]
        # Target TD: recompensa + (si no es terminal) valor descontado del mejor en s'.
        target = reward
        if not done:
            next_key = self._key(self.game.observe(self.agent))
            target += self.gamma * np.max(self._q(next_key))
        q = self._q(self.last_state)
        q[self.last_action] += self.alpha * (target - q[self.last_action])
        self.t += 1

    def policy(self):
        # Politica greedy (one-hot) sobre el estado actualmente observado.
        q = self._q(self._key(self.game.observe(self.agent)))
        p = np.zeros(self.n_actions)
        p[int(np.argmax(q))] = 1.0
        return p
