import numpy as np


def _has(obj, name):
    return callable(getattr(obj, name, None))


def _learns_externally(agent):
    # IQL/JAL-AM: el runner les llama reset()+update(). FP/RM se actualizan en action(), Random no aprende.
    return _has(agent, "reset") and _has(agent, "update")


def _episodic(game):
    # los juegos con estado (Foraging) tienen done(); los one-shot no
    return _has(game, "done")


def _set_learn(agents, learn):
    for a in agents.values():
        if hasattr(a, "learn"):
            a.learn = learn


def _reset_agents(agents):
    for a in agents.values():
        if _has(a, "reset"):
            a.reset()


def play_episode(game, agents, learn=True):
    # un episodio completo de un juego con estado (reset -> pasos hasta done)
    _set_learn(agents, learn)
    game.reset()
    _reset_agents(agents)
    cum = {ag: 0.0 for ag in game.agents}
    while not game.done():
        actions = {ag: agents[ag].action() for ag in game.agents}
        game.step(actions)
        for ag in game.agents:
            cum[ag] += game.reward(ag)
            if _learns_externally(agents[ag]):
                agents[ag].update()
    return cum


def run(game, agents, episodes, learn=True):
    # devuelve (episodes, n_agentes) con el reward por episodio/ronda, en el orden de game.agents
    _set_learn(agents, learn)
    order = list(game.agents)
    hist = np.zeros((episodes, len(order)))
    if _episodic(game):
        for e in range(episodes):
            cum = play_episode(game, agents, learn=learn)
            hist[e] = [cum[ag] for ag in order]
    else:
        # one-shot: reseteo una sola vez y juego las rondas de corrido (asi FP/RM ven la jugada previa)
        game.reset()
        _reset_agents(agents)
        for e in range(episodes):
            actions = {ag: agents[ag].action() for ag in order}
            game.step(actions)
            for j, ag in enumerate(order):
                hist[e, j] = game.reward(ag)
                if _learns_externally(agents[ag]):
                    agents[ag].update()
    return hist


def train(game, agents, iterations, episodes_per_iter, learn=True):
    # corre iterations*episodes y promedia por iteracion -> (iterations, n_agentes), util para graficar
    h = run(game, agents, iterations * episodes_per_iter, learn=learn)
    return h.reshape(iterations, episodes_per_iter, -1).mean(axis=1)
