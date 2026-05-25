import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        # Utilidad contrafactual: que habria cobrado con cada accion alternativa a',
        # manteniendo fijas las acciones reales de los demas.
        for a_prime in g.action_iter(self.agent):
            actions[self.agent] = a_prime
            g.reset()
            g.step(actions)
            u[a_prime] = g.reward(self.agent)
        # Regret de no haber jugado a' = utilidad de a' menos la que realmente obtuve (u[a]).
        r = u - u[a]
        return r
    
    def regret_matching(self):
        # Solo cuentan los arrepentimientos positivos (acciones que "ojala hubiera jugado mas").
        positive = np.maximum(self.cum_regrets, 0)
        total = positive.sum()
        if total > 0:
            self.curr_policy = positive / total
        else:
            # Sin regret positivo: no hay nada que preferir, jugamos uniforme.
            n = self.game.num_actions(self.agent)
            self.curr_policy = np.full(n, 1 / n)
        # Acumulamos para poder promediar la estrategia (lo que converge al NE).
        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))
    
    def policy(self):
        return self.learned_policy
