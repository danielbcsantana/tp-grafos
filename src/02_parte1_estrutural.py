"""Parte I — Análise estrutural obrigatória (grafo não direcionado tratado).

Métricas calculadas com implementação própria (lista de adjacência). Para as
distâncias de todos os pares (diâmetro, raio, caminho médio), usa-se BFS a
partir de cada vértice; para acelerar, o BFS é executado pelo
scipy.sparse.csgraph (BFS em C, unweighted=True) em blocos de origens.
O resultado foi conferido com o NetworkX.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grafo import (carregar_nao_direcionado, componentes, num_arestas, para_csr,
                   subgrafo_induzido, RESULTS, FIGS)


def triangulos_e_clustering(adj):
    """Coeficiente de clusterização local e nº de triângulos (implementação própria)."""
    viz = {u: set(ws) for u, ws in adj.items()}
    tri_por_no = {}
    c_local = {}
    for u, ws in viz.items():
        k = len(ws)
        t = 0
        for v in ws:
            t += len(ws & viz[v])
        t //= 2  # cada triângulo em u é contado 2 vezes
        tri_por_no[u] = t
        c_local[u] = 0.0 if k < 2 else 2 * t / (k * (k - 1))
    total_tri = sum(tri_por_no.values()) // 3
    return total_tri, c_local, tri_por_no


def distancias_todos_pares(adj, bloco=1000):
    from scipy.sparse.csgraph import shortest_path
    m, nos = para_csr(adj)
    n = len(nos)
    ecc = np.zeros(n, dtype=np.int64)
    soma = 0
    pares = 0
    hist = Counter()
    for ini in range(0, n, bloco):
        idx = np.arange(ini, min(ini + bloco, n))
        d = shortest_path(m, method="D", unweighted=True, directed=False, indices=idx)
        d[np.isinf(d)] = -1
        d = d.astype(np.int64)
        ecc[idx] = d.max(axis=1)
        validos = d[d > 0]
        soma += int(validos.sum())
        pares += validos.size
        vals, cnts = np.unique(validos, return_counts=True)
        for v, c in zip(vals, cnts):
            hist[int(v)] += int(c)
    return {
        "diametro": int(ecc.max()),
        "raio": int(ecc.min()),
        "comprimento_medio_caminhos": soma / pares,
        "pares_ordenados_considerados": int(pares),
        "hist_distancias": {k: hist[k] // 2 for k in sorted(hist)},
        "ecc_hist": dict(Counter(int(e) for e in ecc)),
        "centro": [nos[i] for i in np.where(ecc == ecc.min())[0]][:20],
    }


def main():
    os.makedirs(FIGS, exist_ok=True)
    adj = carregar_nao_direcionado()
    n = len(adj)
    m = num_arestas(adj)
    graus = {u: len(ws) for u, ws in adj.items()}
    gv = np.array(list(graus.values()))
    dist_graus = dict(sorted(Counter(gv.tolist()).items()))
    comps = componentes(adj)
    densidade = 2 * m / (n * (n - 1))

    tri, c_local, _ = triangulos_e_clustering(adj)
    c_medio = float(np.mean(list(c_local.values())))
    # transitividade global (complementar)
    trios = sum(k * (k - 1) / 2 for k in gv)
    transitividade = 3 * tri / trios

    print("calculando distâncias de todos os pares...")
    dist = distancias_todos_pares(adj)

    res = {
        "vertices": n,
        "arestas": m,
        "grau_min": int(gv.min()),
        "grau_max": int(gv.max()),
        "grau_medio": float(gv.mean()),
        "grau_mediana": float(np.median(gv)),
        "grau_desvio": float(gv.std()),
        "densidade": densidade,
        "num_componentes": len(comps),
        "tamanhos_componentes": [len(c) for c in comps],
        "diametro": dist["diametro"],
        "raio": dist["raio"],
        "comprimento_medio_caminhos": dist["comprimento_medio_caminhos"],
        "hist_distancias": dist["hist_distancias"],
        "hist_excentricidade": dist["ecc_hist"],
        "coef_clusterizacao_medio": c_medio,
        "transitividade_global": transitividade,
        "triangulos": tri,
        "frac_vertices_clustering_zero": float(np.mean([c == 0 for c in c_local.values()])),
        "distribuicao_graus": dist_graus,
    }

    # Conferência com NetworkX (apenas validação)
    try:
        import networkx as nx
        G = nx.Graph()
        for u, ws in adj.items():
            for w in ws:
                G.add_edge(u, w)
        res["validacao_networkx"] = {
            "clustering_medio": nx.average_clustering(G),
            "triangulos": sum(nx.triangles(G).values()) // 3,
            "densidade": nx.density(G),
        }
    except ImportError:
        pass

    with open(os.path.join(RESULTS, "parte1.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(RESULTS, "distribuicao_graus.csv"), "w") as f:
        f.write("grau,frequencia,probabilidade\n")
        for k, c in dist_graus.items():
            f.write(f"{k},{c},{c / n:.6f}\n")

    # --- Figuras ---
    # distribuição de graus (linear)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = list(dist_graus.keys()); cs = list(dist_graus.values())
    ax.bar(ks, cs, width=0.9, color="#3b6ea5")
    ax.set_xlabel("grau k"); ax.set_ylabel("número de vértices")
    ax.set_title("Distribuição de graus — p2p-Gnutella06")
    ax.set_xlim(0, 60)
    ax.axvline(gv.mean(), color="#c0392b", ls="--", lw=1, label=f"média = {gv.mean():.2f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "distribuicao_graus.png"), dpi=200)
    plt.close(fig)

    # distribuição das distâncias
    fig, ax = plt.subplots(figsize=(6, 3.4))
    hd = dist["hist_distancias"]
    tot = sum(hd.values())
    ax.bar(list(hd.keys()), [v / tot for v in hd.values()], color="#3b6ea5")
    ax.set_xlabel("distância d (saltos)"); ax.set_ylabel("fração dos pares")
    ax.set_title("Distribuição das distâncias entre pares")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "distribuicao_distancias.png"), dpi=200)
    plt.close(fig)

    # visualização reduzida: bola BFS de raio 2 a partir do vértice de maior grau,
    # limitada a ~400 vértices
    import networkx as nx
    hub = max(graus, key=graus.get)
    ordem = [hub]; visto = {hub}; i = 0
    while i < len(ordem) and len(ordem) < 400:
        u = ordem[i]; i += 1
        for w in sorted(adj[u], key=lambda x: -graus[x]):
            if w not in visto and len(ordem) < 400:
                visto.add(w); ordem.append(w)
    sub = subgrafo_induzido(adj, ordem)
    H = nx.Graph()
    for u, ws in sub.items():
        H.add_node(u)
        for w in ws:
            H.add_edge(u, w)
    pos = nx.spring_layout(H, seed=42, k=0.15)
    fig, ax = plt.subplots(figsize=(6, 6))
    tam = [8 + 4 * graus[u] for u in H.nodes()]
    cor = [graus[u] for u in H.nodes()]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.4)
    sc = nx.draw_networkx_nodes(H, pos, ax=ax, node_size=tam, node_color=cor,
                                cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="grau no grafo completo")
    ax.set_title(f"Subgrafo de {H.number_of_nodes()} vértices (BFS a partir do vértice {hub})")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "visualizacao_subgrafo.png"), dpi=200)
    plt.close(fig)

    resumo = {k: v for k, v in res.items() if k not in ("distribuicao_graus",)}
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
