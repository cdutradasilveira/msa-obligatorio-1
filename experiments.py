import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventana: guardamos las figuras a archivo
import matplotlib.pyplot as plt

import runner
from games.mp import MP
from games.rps import RPS
from games.bos import BoS
from games.cournot import Cournot
from games.threeplayers import ThreePlayers
from games.foraging import Foraging
from agents.fictitiousplay import FictitiousPlay
from agents.regretmatching import RegretMatching
from agents.iql_agent import IQLAgent, IQLAgentConfig
from agents.jalam_agent import JALAMAgent, JALAMAgentConfig
from agents.random_agent import RandomAgent

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

# estilo limpio para todas las figuras (look tipo libro)
for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
    try:
        plt.style.use(_style); break
    except OSError:
        continue
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "lines.linewidth": 1.8,
})


def _simplex_point(policy, agent_idx):
    # agente 0 en el triangulo inferior-izq; agente 1 espejado en el superior-der (estilo libro)
    rock, paper = float(policy[0]), float(policy[1])
    return (rock, paper) if agent_idx == 0 else (1.0 - rock, 1.0 - paper)


def _draw_simplex(ax, title):
    ax.plot([0, 1], [1, 0], "k--", lw=0.8, alpha=0.6)  # diagonal que separa los dos simplex
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
    ax.set_xlabel(r"$\pi$(Piedra)"); ax.set_ylabel(r"$\pi$(Papel)"); ax.set_title(title)


# --- builders de agentes (paramétricos en N) ---
def _fp(g):    return {a: FictitiousPlay(game=g, agent=a, seed=i + 1) for i, a in enumerate(g.agents)}
def _rm(g):    return {a: RegretMatching(game=g, agent=a, seed=i + 1) for i, a in enumerate(g.agents)}
def _iql(g, max_t=20000):   return {a: IQLAgent(g, a, IQLAgentConfig(max_t=max_t, seed=i + 1)) for i, a in enumerate(g.agents)}
def _jalam(g, max_t=20000): return {a: JALAMAgent(g, a, JALAMAgentConfig(max_t=max_t, seed=i + 1)) for i, a in enumerate(g.agents)}
def _random(g): return {a: RandomAgent(game=g, agent=a) for a in g.agents}


def _l1_to_uniform(policy):
    n = len(policy)
    return float(np.abs(np.asarray(policy) - 1.0 / n).sum())


# ============ A. Convergencia al equilibrio en suma cero (MP, RPS) ============
def exp_convergence_zero_sum(total=20000, snap=200):
    for Game, gname in [(MP, "MP"), (RPS, "RPS")]:
        plt.figure(figsize=(7, 4))
        matchups = {
            "FP vs FP":   lambda g: _fp(g),
            "RM vs RM":   lambda g: _rm(g),
            "FP vs RM":   lambda g: {g.agents[0]: FictitiousPlay(game=g, agent=g.agents[0], seed=1),
                                     g.agents[1]: RegretMatching(game=g, agent=g.agents[1], seed=2)},
            "IQL vs IQL": lambda g: _iql(g, max_t=total // 2),
        }
        for label, build in matchups.items():
            g = Game()
            agents = build(g)
            dists = []
            def cb(e, agents=agents, g=g, dists=dists):
                if (e + 1) % snap == 0:
                    dists.append(_l1_to_uniform(agents[g.agents[0]].policy()))
            runner.run(g, agents, total, callback=cb)
            plt.plot(range(snap, total + 1, snap), dists, label=label)
        plt.xlabel("rondas")
        plt.ylabel("distancia de la política promedio al equilibrio (L1)")
        plt.title(f"Convergencia al equilibrio de Nash - {gname}")
        plt.legend(); plt.tight_layout()
        path = os.path.join(FIG, f"A_convergencia_{gname}.png")
        plt.savefig(path, dpi=120); plt.close()
        print("guardado:", path)


# ============ B. Coordinación (BoS) ============
def exp_coordination_bos(total=20000, last=2000):
    algos = {"FP": _fp, "RM": _rm,
             "IQL": lambda g: _iql(g, max_t=total // 2),
             "JAL-AM": lambda g: _jalam(g, max_t=total // 2)}
    res = {}
    for label, build in algos.items():
        g = BoS(); agents = build(g)
        h = runner.run(g, agents, total)
        res[label] = h[-last:].mean(axis=0)
    labels = list(res.keys())
    a0 = [res[l][0] for l in labels]; a1 = [res[l][1] for l in labels]
    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, a0, w, label="agent_0")
    plt.bar(x + w / 2, a1, w, label="agent_1")
    plt.xticks(x, labels); plt.ylabel(f"reward promedio (últimas {last} rondas)")
    plt.title("Coordinación en BoS (self-play)"); plt.legend(); plt.tight_layout()
    path = os.path.join(FIG, "B_coordinacion_BoS.png")
    plt.savefig(path, dpi=120); plt.close()
    print("guardado:", path, "|", {l: np.round(res[l], 2).tolist() for l in labels})


# ============ C. Cournot (chequeo numérico) y ThreePlayers (N>2) ============
def exp_cournot(total=20000, snap=200):
    g = Cournot()
    q_star, p_star, profit_star = g.get_nash_equilibrium()
    quantities = g._quantities
    agents = _fp(g)
    avgq = []
    def cb(e):
        if (e + 1) % snap == 0:
            p = agents[g.agents[0]].policy()
            avgq.append(float((p * quantities).sum()))
    runner.run(g, agents, total, callback=cb)
    plt.figure(figsize=(7, 4))
    plt.plot(range(snap, total + 1, snap), avgq, label="cantidad promedio FP (agent_0)")
    plt.axhline(q_star, color="r", ls="--", label=f"Nash teórico q*={q_star:.2f}")
    plt.xlabel("rondas"); plt.ylabel("cantidad producida")
    plt.title("Cournot: FP converge cerca de la cantidad de Nash"); plt.legend(); plt.tight_layout()
    path = os.path.join(FIG, "C_cournot.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"guardado: {path} | Nash q*={q_star:.3f} | cantidad final FP={avgq[-1]:.3f}")


def exp_threeplayers(total=5000):
    print("ThreePlayers (config 1), N=3, politica aprendida por agente:")
    for label, build in {"FP": _fp, "RM": _rm, "IQL": lambda g: _iql(g, max_t=total // 2)}.items():
        g = ThreePlayers(config=1); agents = build(g)
        runner.run(g, agents, total)
        print(f"  {label}: " + " | ".join(f"{a}={np.round(agents[a].policy(), 2).tolist()}" for a in g.agents))


# ============ D. Foraging (con estado) ============
def exp_foraging_competition(iters=40, eps=100):
    for cfg, mt in [("Foraging-5x5-2p-1f-v3", 20000), ("Foraging-8x8-2p-1f-v3", 50000)]:
        plt.figure(figsize=(7, 4))
        for label, build in {"IQL": lambda g: _iql(g, mt), "JAL-AM": lambda g: _jalam(g, mt)}.items():
            g = Foraging(config=cfg, seed=1); agents = build(g)
            curve = runner.train(g, agents, iters, eps)
            for j, a in enumerate(g.agents):
                plt.plot(curve[:, j], label=f"{label} {a}")
        plt.xlabel("iteración"); plt.ylabel("reward promedio por episodio")
        plt.title(f"Foraging competitivo - {cfg}"); plt.legend(); plt.tight_layout()
        path = os.path.join(FIG, f"D_competencia_{cfg}.png")
        plt.savefig(path, dpi=120); plt.close()
        print("guardado:", path)


def exp_foraging_coop(iters=160, eps=50, mt=100000):
    cfg = "Foraging-5x5-2p-1f-coop-v3"
    plt.figure(figsize=(7, 4))
    for label, build in {"IQL": lambda g: _iql(g, mt), "JAL-AM": lambda g: _jalam(g, mt)}.items():
        g = Foraging(config=cfg, seed=1); agents = build(g)
        curve = runner.train(g, agents, iters, eps)
        for j, a in enumerate(g.agents):
            plt.plot(curve[:, j], label=f"{label} {a}")
    plt.xlabel("iteración"); plt.ylabel("reward promedio por episodio")
    plt.title(f"Foraging cooperativo - {cfg} (suben juntos)"); plt.legend(); plt.tight_layout()
    path = os.path.join(FIG, "D_cooperacion_5x5.png")
    plt.savefig(path, dpi=120); plt.close()
    print("guardado:", path)


def exp_foraging_wall(iters=120, eps=50):
    # gradiente: hasta que tamaño de tablero se aprende a cooperar (2 jugadores)
    cfgs = ["Foraging-5x5-2p-1f-coop-v3", "Foraging-6x6-2p-1f-coop-v3",
            "Foraging-7x7-2p-1f-coop-v3", "Foraging-8x8-2p-1f-coop-v3"]
    finals = []
    for cfg in cfgs:
        g = Foraging(config=cfg, seed=1); agents = _jalam(g, max_t=150000)
        curve = runner.train(g, agents, iters, eps)
        finals.append(curve[-1].sum())
        print(f"  {cfg}: reward total final = {finals[-1]:.3f}")
    plt.figure(figsize=(7, 4))
    plt.bar([c.split("-")[1] for c in cfgs], finals)
    plt.ylabel("reward total final por episodio"); plt.xlabel("tamaño del tablero (2 jugadores, coop)")
    plt.title("El muro del descubrimiento: cooperación aprendible solo en tableros chicos")
    plt.tight_layout()
    path = os.path.join(FIG, "D_muro_descubrimiento.png")
    plt.savefig(path, dpi=120); plt.close()
    print("guardado:", path)


# ============ E. Réplicas de figuras del libro (Albrecht et al. 2024) ============
def exp_rps_simplex(total=500, snap=1):
    # Fig 6.5: la distribucion empirica de FP en RPS converge al equilibrio
    g = RPS(); agents = _fp(g)
    emp = {a: [] for a in g.agents}
    def cb(e):
        if (e + 1) % snap == 0:
            for a in g.agents:
                emp[a].append(agents[a].policy().copy())
    runner.run(g, agents, total, callback=cb)
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_simplex(ax, "FP en RPS: la distribución empírica converge al equilibrio")
    for idx, (a, col, mk) in enumerate(zip(g.agents, ("tab:blue", "tab:orange"), ("o", "X"))):
        pts = np.array([_simplex_point(p, idx) for p in emp[a]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=col, alpha=0.5, lw=1)
        ax.scatter(pts[::15, 0], pts[::15, 1], s=14, color=col, marker=mk, label=f"Agente {idx + 1} (empírica)")
        nx, ny = _simplex_point(np.array([1 / 3, 1 / 3, 1 / 3]), idx)
        ax.scatter([nx], [ny], marker="*", s=180, color="black", zorder=5)
    ax.legend(loc="upper center")
    path = os.path.join(FIG, "E_simplex_RPS.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("guardado:", path)


def exp_rm_rps_simplex(total=10000, snap=10):
    # Fig 6.14: RM en RPS. (a) politica actual oscila; (b) distribucion empirica converge.
    g = RPS(); agents = _rm(g)
    cur = {a: [] for a in g.agents}; emp = {a: [] for a in g.agents}
    def cb(e):
        if (e + 1) % snap == 0:
            for a in g.agents:
                cur[a].append(agents[a].curr_policy.copy())
                emp[a].append(agents[a].policy().copy())
    runner.run(g, agents, total, callback=cb)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5.5))
    _draw_simplex(axa, "(a) política actual")
    _draw_simplex(axb, "(b) distribución empírica")
    for idx, (a, col) in enumerate(zip(g.agents, ("tab:blue", "tab:orange"))):
        pa = np.array([_simplex_point(p, idx) for p in cur[a]])
        pb = np.array([_simplex_point(p, idx) for p in emp[a]])
        axa.plot(pa[:, 0], pa[:, 1], "-", color=col, alpha=0.35, lw=0.6)
        axb.plot(pb[:, 0], pb[:, 1], "-", color=col, alpha=0.7, lw=1, label=f"Agente {idx + 1}")
        nx, ny = _simplex_point(np.array([1 / 3, 1 / 3, 1 / 3]), idx)
        for ax in (axa, axb):
            ax.scatter([nx], [ny], marker="*", s=160, color="black", zorder=5)
    axb.legend(loc="upper center")
    fig.suptitle("RM en RPS (réplica Fig 6.14)")
    path = os.path.join(FIG, "E_simplex_RM_RPS.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("guardado:", path)


def exp_rm_regrets(total=10000, snap=10):
    # Fig 6.15: el regret promedio por accion (cum_regrets/niter) tiende a 0 (no-regret)
    g = RPS(); agents = _rm(g); a0 = g.agents[0]
    reg = []
    def cb(e):
        if (e + 1) % snap == 0:
            ag = agents[a0]
            reg.append(ag.cum_regrets / ag.niter)
    runner.run(g, agents, total, callback=cb)
    reg = np.array(reg)
    xs = range(snap, total + 1, snap)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(("Piedra", "Papel", "Tijera")):
        ax.plot(xs, reg[:, k], label=name)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5); ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("episodios"); ax.set_ylabel("regret promedio no condicional")
    ax.set_title("RM en RPS: el regret promedio tiende a 0 (réplica Fig 6.15)")
    ax.legend()
    path = os.path.join(FIG, "E_regrets_RM_RPS.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("guardado:", path)


def _build_alg(label, g, mt, off):
    if label == "IQL":
        return {a: IQLAgent(g, a, IQLAgentConfig(max_t=mt, seed=off + i + 1)) for i, a in enumerate(g.agents)}
    if label == "JAL-AM":
        return {a: JALAMAgent(g, a, JALAMAgentConfig(max_t=mt, seed=off + i + 1)) for i, a in enumerate(g.agents)}
    return {a: RandomAgent(game=g, agent=a) for a in g.agents}


def exp_foraging_multirun(cfg="Foraging-5x5-2p-1f-coop-v3", algos=("IQL", "JAL-AM", "Random"),
                          n_seeds=4, iters=120, eps=50, mt=100000):
    # replica Fig 6.7: return promediado sobre varias corridas, con banda de desvio
    plt.figure(figsize=(7, 4))
    for label in algos:
        runs = []
        for s in range(n_seeds):
            g = Foraging(config=cfg, seed=s + 1)
            agents = _build_alg(label, g, mt, off=s * 10)
            curve = runner.train(g, agents, iters, eps)
            runs.append(curve.sum(axis=1))  # return total del equipo por iteracion
        runs = np.array(runs)
        mean = runs.mean(axis=0); std = runs.std(axis=0)
        xs = np.arange(iters)
        line, = plt.plot(xs, mean, label=label)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color=line.get_color())
    plt.xlabel("iteración"); plt.ylabel(f"return total del equipo (promedio de {n_seeds} corridas)")
    plt.title(f"Foraging cooperativo {cfg.split('-')[1]}: curvas con desvío")
    plt.legend(); plt.tight_layout()
    path = os.path.join(FIG, "E_curvas_multirun.png")
    plt.savefig(path, dpi=120); plt.close()
    print("guardado:", path)


if __name__ == "__main__":
    # experimentos rápidos (normal-form)
    exp_convergence_zero_sum()
    exp_coordination_bos()
    exp_cournot()
    exp_threeplayers()
    exp_rps_simplex()
    exp_rm_rps_simplex()
    exp_rm_regrets()
