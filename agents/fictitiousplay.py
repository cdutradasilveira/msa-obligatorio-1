from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random 
        #

        self.learned_policy: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar learned_policy usando de count
        # 

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #s
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente. 
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        return utility
    
    def bestresponse(self):
        a = None
        #
        # TODO: retornar la acción de mayor utilidad
        #
        return a
     
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
    