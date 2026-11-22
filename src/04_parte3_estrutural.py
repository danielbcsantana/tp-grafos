"""Parte III — Small-world, lei de potência e robustez.

1) Small-world (Apêndice A) — implementação própria, sem funções prontas:
   L(G): média das distâncias via BFS próprio a partir de TODOS os vértices;
   C(G): coeficiente de clusterização médio (contagem própria de triângulos);
   grafo aleatório equivalente: G(n, m) de Erdős–Rényi gerado com código próprio
   (mesmo n e mesmo m => mesmo grau médio), 10 instâncias.
   Critério: L ≈ L_rand e C ≫ C_rand; índice σ = (C/C_rand)/(L/L_rand).
2) Lei de potência (Apêndice B): histograma e CCDF em log-log, reta por mínimos
   quadrados e estimador de máxima verossimilhança (Clauset et al., 2009) com
   k_min escolhido pela menor distância KS.
3) Robustez (Apêndice C): r = ceil(0,05·n); remoção aleatória (T = 30) e
   remoção dos r vértices de maior grau (ranking fixo). Métricas A–D.
"""
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grafo import (carregar_nao_direcionado, componentes, num_arestas, para_csr,
                   subgrafo_induzido, RESULTS, FIGS)
from algoritmos import bfs

SEED = 2026
Z = 1.959963984540054


# ------------------------------------------------------------ small-world
def caminho_medio_proprio(adj):
    """L = média de d(u,v) sobre todos os pares alcançáveis (BFS próprio)."""
    soma = 0
    pares = 0
    for s in adj:
        d = bfs(adj, s)
        soma += sum(d.values())
        pares += len(d) - 1
    return soma / pares


def clustering_medio_proprio(adj):
    viz = {u: set(ws) for u, ws in adj.items()}
    tot = 0.0
    for u, ws in viz.items():
        k = len(ws)
        if k < 2:
            continue
        t = sum(len(ws & viz[v]) for v in ws) // 2
        tot += 2 * t / (k * (k - 1))
    return tot / len(adj)


def erdos_renyi_gnm(n, m, rng):
    """G(n, m): m arestas distintas sorteadas uniformemente, sem laços."""
    arestas = set()
    while len(arestas) < m:
        u = rng.randrange(n); v = rng.randrange(n)
        if u == v:
            continue
        arestas.add((min(u, v), max(u, v)))
    adj = {u: [] for u in range(n)}
    for u, v in arestas:
        adj[u].append(v); adj[v].append(u)
    return adj


def small_world(adj, instancias=10):
    n, m = len(adj), num_arestas(adj)
    L = caminho_medio_proprio(adj)
    C = clustering_medio_proprio(adj)
    rng = random.Random(SEED)
    Ls, Cs, lccs = [], [], []
    for i in range(instancias):
        g = erdos_renyi_gnm(n, m, rng)
        comps = componentes(g)
        lcc = subgrafo_induzido(g, comps[0])
        lccs.append(len(comps[0]) / n)
        Ls.append(caminho_medio_proprio(lcc))
        Cs.append(clustering_medio_proprio(g))
        print(f"  ER {i}: L={Ls[-1]:.3f} C={Cs[-1]:.5f}")
    Lr, Cr = float(np.mean(Ls)), float(np.mean(Cs))
    k = 2 * m / n
    return {
        "L": L, "C": C,
        "L_rand_media": Lr, "L_rand_desvio": float(np.std(Ls, ddof=1)),
        "C_rand_media": Cr, "C_rand_desvio": float(np.std(Cs, ddof=1)),
        "rand_lcc_frac_media": float(np.mean(lccs)),
        "L_rand_teorico_ln_n_ln_k": math.log(n) / math.log(k),
        "C_rand_teorico_k_n": k / n,
        "razao_L": L / Lr, "razao_C": C / Cr,
        "sigma": (C / Cr) / (L / Lr),
        "instancias": instancias,
        "L_log_n": math.log(n),
    }


# ------------------------------------------------------------ lei de potência
def mle_powerlaw_discreta(graus, kmin):
    x = np.array([k for k in graus if k >= kmin], dtype=float)
    gama = 1 + len(x) / np.sum(np.log(x / (kmin - 0.5)))
    return gama, len(x)


def ks_distancia(graus, kmin, gama):
    x = np.sort(np.array([k for k in graus if k >= kmin]))
    ks = np.unique(x)
    emp = np.array([np.mean(x >= k) for k in ks])  # CCDF empírica
    teo = (ks / kmin) ** (1 - gama)                 # CCDF contínua aproximada
    return float(np.max(np.abs(emp - teo)))


def lei_de_potencia(adj):
    graus = [len(w) for w in adj.values()]
    n = len(graus)
    cont = Counter(graus)
    ks = np.array(sorted(cont))
    pk = np.array([cont[k] / n for k in ks])
    ccdf = np.array([np.mean(np.array(graus) >= k) for k in ks])

    # reta por mínimos quadrados em log P(k) × log k
    b_all, a_all = np.polyfit(np.log10(ks), np.log10(pk), 1)
    r2_all = np.corrcoef(np.log10(ks), np.log10(pk))[0, 1] ** 2
    b_c, a_c = np.polyfit(np.log10(ks), np.log10(ccdf), 1)
    r2_c = np.corrcoef(np.log10(ks), np.log10(ccdf))[0, 1] ** 2

    # MLE com escolha de kmin
    melhor = None
    for kmin in range(2, 40):
        g, ntail = mle_powerlaw_discreta(graus, kmin)
        if ntail < 50:
            break
        d = ks_distancia(graus, kmin, g)
        if melhor is None or d < melhor[2]:
            melhor = (kmin, g, d, ntail)
    kmin, gama_mle, dks, ntail = melhor

    # concentração: fração dos vértices com grau baixo
    res = {
        "ls_pk_gama": -b_all, "ls_pk_r2": r2_all,
        "ls_ccdf_inclinacao": b_c, "ls_ccdf_gama_equiv": 1 - b_c, "ls_ccdf_r2": r2_c,
        "mle_kmin": kmin, "mle_gama": gama_mle, "mle_ks": dks, "mle_n_cauda": ntail,
        "frac_grau_le_10": float(np.mean(np.array(graus) <= 10)),
        "razao_kmax_kmedio": max(graus) / np.mean(graus),
        "coef_variacao_grau": float(np.std(graus) / np.mean(graus)),
    }

    media = np.mean(graus)
    poisson = np.array([math.exp(-media + k * math.log(media) - math.lgamma(k + 1)) for k in ks])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axs[0]
    ax.loglog(ks, pk, "o", ms=3, label="P(k) observado")
    ax.loglog(ks, 10 ** (a_all + b_all * np.log10(ks)), "-", color="#c0392b",
              label=f"reta MQ: γ={-b_all:.2f}, R²={r2_all:.2f}")
    ax.loglog(ks, poisson, ":", color="gray", label="Poisson (ER, mesmo ⟨k⟩)")
    ax.set_ylim(pk.min() / 3, 1)
    ax.set_xlabel("k"); ax.set_ylabel("P(k)"); ax.set_title("Distribuição de graus (log-log)")
    ax.legend(fontsize=7)
    ax = axs[1]
    ax.loglog(ks, ccdf, "o", ms=3, label="CCDF observada")
    kk = ks[ks >= kmin]
    c0 = np.mean(np.array(graus) >= kmin)
    ax.loglog(kk, c0 * (kk / kmin) ** (1 - gama_mle), "-", color="#c0392b",
              label=f"lei de potência MLE: γ={gama_mle:.2f}, k_min={kmin}")
    ax.set_xlabel("k"); ax.set_ylabel("P(K ≥ k)"); ax.set_title("CCDF (log-log)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "lei_de_potencia.png"), dpi=200)
    plt.close(fig)
    return res


# ------------------------------------------------------------ robustez
def metricas(adj_rest, n_original):
    from scipy.sparse.csgraph import shortest_path
    comps = componentes(adj_rest)
    lcc = comps[0]
    isolados = sum(1 for u in adj_rest if not adj_rest[u])
    # Métrica C: distância média e diâmetro dentro da maior componente
    sub = subgrafo_induzido(adj_rest, lcc)
    m, _ = para_csr(sub)
    soma = 0; pares = 0; diam = 0
    nL = len(lcc)
    for ini in range(0, nL, 1500):
        idx = np.arange(ini, min(ini + 1500, nL))
        d = shortest_path(m, method="D", unweighted=True, directed=False, indices=idx)
        soma += d.sum(); pares += d.size - len(idx); diam = max(diam, int(d.max()))
    return {
        "A_maior_componente": len(lcc),
        "A_frac_n_original": len(lcc) / n_original,
        "A_frac_restantes": len(lcc) / len(adj_rest),
        "B_num_componentes": len(comps),
        "C_dist_media_lcc": soma / pares,
        "C_diametro_lcc": diam,
        "D_frac_isolados": isolados / len(adj_rest),
    }


def remover(adj, rem):
    rem = set(rem)
    return {u: [w for w in ws if w not in rem] for u, ws in adj.items() if u not in rem}


def robustez(adj, T=30):
    n = len(adj)
    r = math.ceil(0.05 * n)
    base = metricas(adj, n)
    rng = random.Random(SEED)
    nos = sorted(adj)
    aleat = []
    for t in range(T):
        rem = rng.sample(nos, r)
        aleat.append(metricas(remover(adj, rem), n))
    graus = {u: len(w) for u, w in adj.items()}
    ranking = sorted(nos, key=lambda u: (-graus[u], u))
    top = ranking[:r]
    cent = metricas(remover(adj, top), n)
    arestas_removidas_top = sum(graus[u] for u in top) - sum(
        1 for u in top for w in adj[u] if w in set(top)) // 2
    resumo = {}
    for chave in base:
        v = np.array([a[chave] for a in aleat], dtype=float)
        dp = v.std(ddof=1)
        resumo[chave] = {
            "original": base[chave],
            "aleatoria_media": float(v.mean()), "aleatoria_desvio": float(dp),
            "aleatoria_ic95": [float(v.mean() - Z * dp / math.sqrt(T)),
                               float(v.mean() + Z * dp / math.sqrt(T))],
            "aleatoria_min": float(v.min()), "aleatoria_max": float(v.max()),
            "direcionada_grau": cent[chave],
        }
    out = {"n": n, "r": r, "T": T, "centralidade": "grau (ranking fixo)",
           "grau_minimo_top_r": graus[top[-1]],
           "frac_arestas_removidas_ataque": arestas_removidas_top / num_arestas(adj),
           "metricas": resumo, "aleatoria_bruto": aleat}

    # boxplots
    chaves = [("A_frac_n_original", "A: |maior comp.| / n"),
              ("B_num_componentes", "B: nº de componentes"),
              ("C_dist_media_lcc", "C: distância média (LCC)"),
              ("D_frac_isolados", "D: fração de isolados")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (k, rot) in zip(axs, chaves):
        v = [a[k] for a in aleat]
        ax.boxplot(v, widths=0.5, showmeans=True)
        mu, sd = np.mean(v), np.std(v, ddof=1)
        ax.errorbar([1.35], [mu], yerr=[sd], fmt="s", color="#3b6ea5", ms=4, capsize=3,
                    label="média ± dp (aleatória)")
        ax.axhline(cent[k], color="#c0392b", ls="--", label="ataque (top 5% grau)")
        ax.axhline(base[k], color="gray", ls=":", label="grafo original")
        ax.set_title(rot, fontsize=9); ax.set_xticks([])
    axs[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Robustez: remoção de r = {r} vértices (5%) — aleatória (T={T}) × direcionada", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "robustez_boxplot.png"), dpi=200)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    adj = carregar_nao_direcionado()
    comps = componentes(adj)
    adj = subgrafo_induzido(adj, comps[0])  # política: maior componente (aqui = grafo todo)
    print("small-world...")
    sw = small_world(adj)
    print(json.dumps(sw, indent=2))
    print("lei de potência...")
    pl = lei_de_potencia(adj)
    print(json.dumps(pl, indent=2))
    print("robustez...")
    rb = robustez(adj)
    print(json.dumps(rb["metricas"], indent=2))
    with open(os.path.join(RESULTS, "parte3.json"), "w") as f:
        json.dump({"small_world": sw, "lei_de_potencia": pl, "robustez": rb}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
