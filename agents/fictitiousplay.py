from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        # count[ag]: conteos de acciones observadas de cada agente (la evidencia de las creencias).
        # learned_policy[ag]: creencia = frecuencia empirica normalizada de cada agente.
        self.count: dict[AgentID, ndarray] = {}
        self.learned_policy: dict[AgentID, ndarray] = {}
        for ag in self.game.agents:
            n = self.game.num_actions(ag)
            if initial is None:
                # Creencia inicial aleatoria: pesos fraccionarios validos que suman 1.
                self.count[ag] = np.random.dirichlet(np.ones(n))
            else:
                self.count[ag] = initial[ag].copy()
            self.learned_policy[ag] = self.count[ag] / np.sum(self.count[ag])

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        # product(*agents_actions) genera TODAS las acciones conjuntas posibles.
        for joint in product(*agents_actions):
            actions = dict(zip(g.agents, joint))
            g.reset()
            g.step(actions)
            # Reward que recibe ESTE agente bajo esa accion conjunta.
            rewards[joint] = g.reward(self.agent)
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        my_idx = self.game.agents.index(self.agent)
        for joint, r in rewards.items():
            # Probabilidad de que los OTROS jueguen su parte de la accion conjunta,
            # segun la creencia actual. Mi propia accion no se pondera (la estoy evaluando).
            prob_others = 1.0
            for idx, ag in enumerate(self.game.agents):
                if ag == self.agent:
                    continue
                prob_others *= self.learned_policy[ag][joint[idx]]
            # Acumulo el reward esperado en la casilla de MI accion dentro de esa conjunta.
            utility[joint[my_idx]] += r * prob_others
        return utility
    
    def bestresponse(self):
        # Mejor respuesta = accion de mayor utilidad esperada contra la creencia actual.
        return int(np.argmax(self.get_utility()))
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    