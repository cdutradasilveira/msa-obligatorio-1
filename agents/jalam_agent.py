from dataclasses import dataclass
from functools import reduce
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


@dataclass
class JALAMAgentConfig:
    alpha: float = 0.1
    gamma: float = 1.0
    min_epsilon: float = 0.01
    max_t: int = 1000
    seed: int | None = None


class JALAMAgent(Agent):
    """Joint-Action Learning con Agent Modeling: aprende Q sobre la accion conjunta
    de todos y modela la politica de los demas para elegir su mejor accion."""

    def __init__(self, game: SimultaneousGame, agent: AgentID, config: JALAMAgentConfig | None = None) -> None:
        super().__init__(game=game, agent=agent)
        self.config = config if config is not None else JALAMAgentConfig()
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.min_epsilon = self.config.min_epsilon
        self.max_t = self.config.max_t
        self.rng = np.random.default_rng(self.config.seed)

        self.agents = list(self.game.agents)
        self.my_idx = self.agents.index(self.agent)
        self.na = [self.game.num_actions(a) for a in self.agents]  # acciones de cada agente
        self.n_actions = self.na[self.my_idx]

        self.Q: dict = {}       # estado -> tensor de valores, indexado por la accion conjunta
        self.models: dict = {}  # estado -> {idx_otro: contador de sus acciones}

        self.learn = True
        self.t = 0
        self.last_state = None
        self.last_action = None

    def _key(self, obs):
        # one-shot: la observacion es la jugada (o None), no un estado -> uso un estado unico
        if obs is None or isinstance(obs, dict):
            return None
        return tuple(np.asarray(obs).flatten().tolist())

    def _q(self, key):
        if key not in self.Q:
            self.Q[key] = np.zeros(tuple(self.na))
        return self.Q[key]

    def _model_policy(self, key, i):
        # lo que creo que va a jugar el agente i (frecuencia que vi; uniforme si no vi nada)
        counts = self.models.get(key, {}).get(i)
        if counts is None or counts.sum() == 0:
            return np.full(self.na[i], 1.0 / self.na[i])
        return counts / counts.sum()

    def _expected_values(self, key):
        # valor esperado de cada accion mia, pesando las conjuntas por lo que creo de los otros
        q = self._q(key)
        dists = []
        for i in range(len(self.agents)):
            dists.append(np.ones(self.na[i]) if i == self.my_idx else self._model_policy(key, i))
        weights = reduce(np.multiply.outer, dists)  # producto externo -> tensor de pesos
        other_axes = tuple(i for i in range(len(self.agents)) if i != self.my_idx)
        return (q * weights).sum(axis=other_axes) if other_axes else q * weights

    @property
    def epsilon(self):
        if not self.learn:
            return 0.0
        frac = max(0.0, (self.max_t - self.t) / self.max_t)
        return self.min_epsilon + (1.0 - self.min_epsilon) * frac

    def reset(self):
        self.last_state = None
        self.last_action = None

    def action(self):
        key = self._key(self.game.observe(self.agent))
        if self.learn and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(self.n_actions))
        else:
            ev = self._expected_values(key)
            best = np.flatnonzero(ev == ev.max())         # desempata al azar (asi no se queda siempre en NONE)
            a = int(self.rng.choice(best))
        self.last_state = key
        self.last_action = a
        return a

    def update(self):
        if not self.learn or self.last_action is None:
            return
        joint = self.game.observe_action(self.agent)  # que jugo cada uno
        if joint is None:
            return
        # actualizo mi modelo de los otros en el estado donde jugue
        model = self.models.setdefault(self.last_state, {})
        for i, ag in enumerate(self.agents):
            if i == self.my_idx:
                continue
            if i not in model:
                model[i] = np.zeros(self.na[i])
            model[i][joint[ag]] += 1
        # actualizo Q de la conjunta que se jugo
        joint_idx = tuple(joint[ag] for ag in self.agents)
        reward = self.game.reward(self.agent)
        done = self.game.terminations[self.agent] or self.game.truncations[self.agent]
        target = reward
        if not done:
            target += self.gamma * np.max(self._expected_values(self._key(self.game.observe(self.agent))))
        q = self._q(self.last_state)
        q[joint_idx] += self.alpha * (target - q[joint_idx])
        self.t += 1

    def policy(self):
        ev = self._expected_values(self._key(self.game.observe(self.agent)))
        p = np.zeros(self.n_actions)
        p[int(np.argmax(ev))] = 1.0
        return p
