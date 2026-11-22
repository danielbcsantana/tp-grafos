"""Utilitários de leitura e representação do grafo (lista de adjacência).

Todas as funções aqui são implementação própria (sem NetworkX), para que os
algoritmos da disciplina trabalhem sobre uma estrutura simples:
    adj: dict[int, list[int]]   (grafo não direcionado, simples)
"""
import gzip
import os
from collections import defaultdict

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(RAIZ, "data", "raw", "p2p-Gnutella06.txt.gz")
PROC_DIR = os.path.join(RAIZ, "data", "processed")
UNDIR = os.path.join(PROC_DIR, "gnutella06_nao_direcionado.edgelist")
LCC = os.path.join(PROC_DIR, "gnutella06_lcc.edgelist")
DIRECIONADO = os.path.join(PROC_DIR, "gnutella06_direcionado.edgelist")
RESULTS = os.path.join(RAIZ, "results")
FIGS = os.path.join(RESULTS, "figures")


def ler_arestas_raw(caminho=RAW):
    """Lê o arquivo SNAP (gz), ignorando linhas de comentário '#'."""
    arestas = []
    abrir = gzip.open if caminho.endswith(".gz") else open
    with abrir(caminho, "rt") as f:
        for linha in f:
            if linha.startswith("#") or not linha.strip():
                continue
            u, v = linha.split()
            arestas.append((int(u), int(v)))
    return arestas


def ler_edgelist(caminho):
    arestas = []
    with open(caminho) as f:
        for linha in f:
            if linha.startswith("#") or not linha.strip():
                continue
            u, v = linha.split()
            arestas.append((int(u), int(v)))
    return arestas


def salvar_edgelist(arestas, caminho, cabecalho=""):
    with open(caminho, "w") as f:
        if cabecalho:
            for l in cabecalho.strip().splitlines():
                f.write("# " + l + "\n")
        for u, v in arestas:
            f.write(f"{u}\t{v}\n")


def adjacencia(arestas, nos=None):
    """Lista de adjacência não direcionada a partir de arestas (u, v)."""
    adj = defaultdict(list)
    if nos is not None:
        for n in nos:
            adj[n]
    for u, v in arestas:
        adj[u].append(v)
        adj[v].append(u)
    return dict(adj)


def adjacencia_direcionada(arestas):
    adj = defaultdict(list)
    for u, v in arestas:
        adj[u].append(v)
        adj[v]
    return dict(adj)


def carregar_lcc():
    """Grafo tratado usado em todas as análises (maior componente conexa)."""
    return adjacencia(ler_edgelist(LCC))


def carregar_nao_direcionado():
    arestas = ler_edgelist(UNDIR)
    nos = set()
    for u, v in arestas:
        nos.add(u); nos.add(v)
    return adjacencia(arestas)


def num_arestas(adj):
    return sum(len(v) for v in adj.values()) // 2


def componentes(adj):
    """Componentes conexas via BFS (implementação própria)."""
    visto = set()
    comps = []
    for s in adj:
        if s in visto:
            continue
        comp = [s]
        visto.add(s)
        i = 0
        while i < len(comp):
            u = comp[i]; i += 1
            for w in adj[u]:
                if w not in visto:
                    visto.add(w); comp.append(w)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def subgrafo_induzido(adj, nos):
    nos = set(nos)
    return {u: [w for w in adj[u] if w in nos] for u in nos}


def para_csr(adj):
    """Converte para matriz esparsa (scipy) — usada só para acelerar
    as distâncias de todos os pares (BFS em C)."""
    import numpy as np
    from scipy.sparse import csr_matrix
    nos = sorted(adj)
    idx = {n: i for i, n in enumerate(nos)}
    linhas, cols = [], []
    for u in nos:
        for w in adj[u]:
            linhas.append(idx[u]); cols.append(idx[w])
    n = len(nos)
    m = csr_matrix((np.ones(len(linhas), dtype=np.int8), (linhas, cols)), shape=(n, n))
    return m, nos
