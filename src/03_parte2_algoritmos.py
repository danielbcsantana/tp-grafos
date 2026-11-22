"""Parte II — Algoritmos da disciplina: correção, complexidade e tempo real.

Experimento A (grafo completo): cada algoritmo é executado R = 30 vezes no
grafo tratado; reporta-se média, desvio padrão amostral e IC 95%. Como
n = 30 >= 30, o IC usa a Normal: x̄ ± z_{0,025}·s/√n, z = 1,96.
(Floyd-Warshall é Θ(V^3): no grafo completo é executado uma única vez,
 opcional via --fw-completo; as 30 repetições são feitas em subgrafos.)

Experimento B (complexidade observada): cada algoritmo é executado em
subgrafos conexos de tamanho crescente (prefixos da ordem BFS a partir do
vértice de maior grau) e ajusta-se tempo ~ a·x^b em escala log-log, com
x = V+E (lineares), (V+E)·log V (Dijkstra/Prim/Kruskal), k·E (Bellman-Ford)
ou V (Floyd-Warshall).
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grafo import (carregar_nao_direcionado, adjacencia_direcionada, ler_edgelist,
                   DIRECIONADO, subgrafo_induzido, num_arestas, RESULTS, FIGS)
import algoritmos as alg

R = 30
Z = 1.959963984540054  # z_{α/2}, α = 0,05


def estatisticas(tempos):
    t = np.array(tempos) * 1000  # ms
    media = t.mean()
    dp = t.std(ddof=1)
    meia = Z * dp / math.sqrt(len(t))
    return {"n": len(t), "media_ms": media, "desvio_ms": dp,
            "ic95_inf_ms": media - meia, "ic95_sup_ms": media + meia,
            "distribuicao_ic": "Normal (Z)" if len(t) >= 30 else "t-Student"}


def cronometrar(func, reps=R):
    tempos = []
    saida = None
    for _ in range(reps):
        t0 = time.perf_counter()
        saida = func()
        tempos.append(time.perf_counter() - t0)
    return tempos, saida


def prefixo_bfs(adj, raiz, tamanho):
    ordem = [raiz]; visto = {raiz}; i = 0
    while i < len(ordem) and len(ordem) < tamanho:
        u = ordem[i]; i += 1
        for w in adj[u]:
            if w not in visto:
                visto.add(w); ordem.append(w)
                if len(ordem) >= tamanho:
                    break
    return subgrafo_induzido(adj, ordem)


def ajuste_loglog(x, y):
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    return float(b), float(math.exp(loga))


def main():
    os.makedirs(FIGS, exist_ok=True)
    fw_completo = "--fw-completo" in sys.argv
    adj = carregar_nao_direcionado()
    adj_dir = adjacencia_direcionada(ler_edgelist(DIRECIONADO))
    V, E = len(adj), num_arestas(adj)
    graus = {u: len(w) for u, w in adj.items()}
    hub = max(graus, key=graus.get)
    s = 0  # vértice de origem fixo para as buscas
    res = {"V": V, "E": E, "origem": s, "repeticoes": R, "z": Z, "algoritmos": {}}

    # ---------------- validação/corretude (uma execução)
    d_bfs = alg.bfs(adj, s)
    ordem_dfs, _ = alg.dfs(adj, s)
    euler = alg.verificar_euleriano(adj)
    d_dij = alg.dijkstra(adj, s)
    d_bf, rodadas_bf, ciclo_neg = alg.bellman_ford(adj, s)
    sccs = alg.tarjan_scc(adj_dir)
    t_kr, w_kr = alg.kruskal(adj)
    t_pr, w_pr = alg.prim(adj, s)
    tam_scc = sorted((len(c) for c in sccs), reverse=True)
    res["corretude"] = {
        "bfs_alcancados": len(d_bfs),
        "bfs_excentricidade_origem": max(d_bfs.values()),
        "dfs_visitados": len(ordem_dfs),
        "euleriano": euler,
        "dijkstra_igual_bfs": d_dij == d_bfs,
        "bellman_ford_igual_bfs": all(d_bf[u] == d_bfs[u] for u in adj),
        "bellman_ford_rodadas_ate_convergir": rodadas_bf,
        "bellman_ford_ciclo_negativo": ciclo_neg,
        "tarjan_num_cfc": len(sccs),
        "tarjan_maior_cfc": tam_scc[0],
        "tarjan_maior_cfc_frac": tam_scc[0] / V,
        "tarjan_cfc_unitarias": sum(1 for t in tam_scc if t == 1),
        "tarjan_top5": tam_scc[:5],
        "kruskal_arestas": len(t_kr), "kruskal_peso": w_kr,
        "prim_arestas": len(t_pr), "prim_peso": w_pr,
    }
    # validação do Tarjan com NetworkX e do FW vetorizado com o puro
    import networkx as nx
    Gd = nx.DiGraph([(u, w) for u in adj_dir for w in adj_dir[u]])
    res["corretude"]["tarjan_confere_networkx"] = (
        sorted(len(c) for c in nx.strongly_connected_components(Gd)) == sorted(tam_scc))
    sub = prefixo_bfs(adj, hub, 150)
    Dv, _ = alg.floyd_warshall(sub)
    Dp = alg.floyd_warshall_puro(sub)
    res["corretude"]["fw_vetorizado_igual_puro_150v"] = bool(np.array_equal(Dv, np.array(Dp, dtype=np.float32)))
    print(json.dumps(res["corretude"], indent=2))

    # ---------------- Experimento A: 30 repetições no grafo completo
    tarefas = {
        "BFS": (lambda: alg.bfs(adj, s), "O(V+E)"),
        "DFS": (lambda: alg.dfs(adj, s), "O(V+E)"),
        "Eulerianidade": (lambda: alg.verificar_euleriano(adj), "O(V+E)"),
        "Dijkstra (heap binário)": (lambda: alg.dijkstra(adj, s), "O((V+E) log V)"),
        "Bellman-Ford": (lambda: alg.bellman_ford(adj, s), "O(V·E) (pior caso)"),
        "Tarjan (CFC, dirigido)": (lambda: alg.tarjan_scc(adj_dir), "O(V+E)"),
        "Kruskal (union-find)": (lambda: alg.kruskal(adj), "O(E log E)"),
        "Prim (heap binário)": (lambda: alg.prim(adj, s), "O(E log V)"),
    }
    for nome, (f, cplx) in tarefas.items():
        tempos, _ = cronometrar(f)
        est = estatisticas(tempos)
        est["complexidade_teorica"] = cplx
        res["algoritmos"][nome] = est
        print(f"{nome:28s} {est['media_ms']:9.2f} ms ± {est['desvio_ms']:.2f}  "
              f"IC95 [{est['ic95_inf_ms']:.2f}, {est['ic95_sup_ms']:.2f}]")

    # Floyd-Warshall: 30 repetições em subgrafo de 1000 vértices
    sub1000 = prefixo_bfs(adj, hub, 1000)
    tempos, _ = cronometrar(lambda: alg.floyd_warshall(sub1000))
    est = estatisticas(tempos)
    est["complexidade_teorica"] = "Θ(V^3)"
    est["obs"] = f"subgrafo conexo com V={len(sub1000)}, E={num_arestas(sub1000)}"
    res["algoritmos"]["Floyd-Warshall (V=1000)"] = est
    print("Floyd-Warshall (1000)", est["media_ms"])

    if fw_completo:
        t0 = time.perf_counter()
        D, _ = alg.floyd_warshall(adj)
        tfw = time.perf_counter() - t0
        res["fw_grafo_completo"] = {
            "tempo_s": tfw, "diametro": float(D.max()),
            "caminho_medio": float(D.sum() / (V * (V - 1))),
        }
        del D
        print("FW completo:", res["fw_grafo_completo"])

    # ---------------- Experimento B: escalabilidade
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    escala = {k: [] for k in tarefas}
    for fr in fracs:
        n = int(round(fr * V))
        g = prefixo_bfs(adj, hub, n) if fr < 1 else adj
        nodes = set(g)
        gd = {u: [w for w in adj_dir[u] if w in nodes] for u in nodes}
        e = num_arestas(g)
        src = hub
        fs = {
            "BFS": lambda: alg.bfs(g, src),
            "DFS": lambda: alg.dfs(g, src),
            "Eulerianidade": lambda: alg.verificar_euleriano(g),
            "Dijkstra (heap binário)": lambda: alg.dijkstra(g, src),
            "Bellman-Ford": lambda: alg.bellman_ford(g, src),
            "Tarjan (CFC, dirigido)": lambda: alg.tarjan_scc(gd),
            "Kruskal (union-find)": lambda: alg.kruskal(g),
            "Prim (heap binário)": lambda: alg.prim(g, src),
        }
        for nome, f in fs.items():
            tempos, out = cronometrar(f, reps=10)
            extra = {}
            if nome == "Bellman-Ford":
                extra["rodadas"] = out[1]
            escala[nome].append({"V": len(g), "E": e, "media_ms": float(np.mean(tempos) * 1000), **extra})
        print("escala", fr, len(g), e)

    fw_ns = [125, 250, 500, 1000, 2000]
    escala_fw = []
    for n in fw_ns:
        g = prefixo_bfs(adj, hub, n)
        tempos, _ = cronometrar(lambda: alg.floyd_warshall(g), reps=10 if n <= 1000 else 3)
        escala_fw.append({"V": len(g), "E": num_arestas(g), "media_ms": float(np.mean(tempos) * 1000)})
        print("fw", n, escala_fw[-1]["media_ms"])

    ajustes = {}
    for nome, pts in escala.items():
        if nome == "Bellman-Ford":
            x = [p["rodadas"] * 2 * p["E"] for p in pts]
            rot = "k·E"
        elif nome in ("Dijkstra (heap binário)", "Prim (heap binário)", "Kruskal (union-find)"):
            x = [(p["V"] + p["E"]) * math.log2(p["V"]) for p in pts]
            rot = "(V+E)·log V"
        else:
            x = [p["V"] + p["E"] for p in pts]
            rot = "V+E"
        y = [p["media_ms"] for p in pts]
        b, a = ajuste_loglog(x, y)
        b_ve, _ = ajuste_loglog([p["V"] + p["E"] for p in pts], y)
        ajustes[nome] = {"x": rot, "expoente_b": b, "expoente_em_V+E": b_ve}
    b, a = ajuste_loglog([p["V"] for p in escala_fw], [p["media_ms"] for p in escala_fw])
    ajustes["Floyd-Warshall"] = {"x": "V", "expoente_b": b}
    res["escalabilidade"] = escala
    res["escalabilidade_fw"] = escala_fw
    res["ajustes_loglog"] = ajustes
    print(json.dumps(ajustes, indent=2))

    with open(os.path.join(RESULTS, "parte2.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ---------------- Figuras
    fig, ax = plt.subplots(figsize=(6.4, 4))
    marcadores = "osD^v<>p"
    for (nome, pts), mk in zip(escala.items(), marcadores):
        ax.plot([p["V"] + p["E"] for p in pts], [p["media_ms"] for p in pts],
                marker=mk, ms=4, lw=1, label=nome)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("V + E do subgrafo"); ax.set_ylabel("tempo médio (ms)")
    ax.set_title("Complexidade observada (log-log)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "escalabilidade_algoritmos.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs = np.array([p["V"] for p in escala_fw]); ys = np.array([p["media_ms"] for p in escala_fw])
    ax.plot(xs, ys, "o-", label=f"medido (ajuste: V^{ajustes['Floyd-Warshall']['expoente_b']:.2f})")
    ax.plot(xs, ys[-1] * (xs / xs[-1]) ** 3, "--", color="gray", label="referência ∝ V³")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("V"); ax.set_ylabel("tempo médio (ms)")
    ax.set_title("Floyd-Warshall: tempo × V")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "escalabilidade_floyd_warshall.png"), dpi=200)
    plt.close(fig)

    nomes = list(res["algoritmos"].keys())
    medias = [res["algoritmos"][n]["media_ms"] for n in nomes]
    erros = [res["algoritmos"][n]["ic95_sup_ms"] - res["algoritmos"][n]["media_ms"] for n in nomes]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(nomes, medias, xerr=erros, color="#3b6ea5", capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("tempo médio (ms), barras = IC 95%")
    ax.invert_yaxis()
    ax.set_title(f"Tempo real (R = {R} execuções)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "tempos_algoritmos.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
