"""Algoritmos da disciplina — implementação própria (sem bibliotecas de grafos).

Representação: lista de adjacência dict[int, list[int]].
O p2p-Gnutella06 não tem pesos; para os algoritmos com pesos (Dijkstra,
Bellman-Ford, Floyd-Warshall, Prim, Kruskal) usa-se peso unitário w(e) = 1.
"""
import heapq
from collections import deque

import numpy as np

INF = float("inf")


# ---------------------------------------------------------------- BFS  O(V+E)
def bfs(adj, s):
    dist = {s: 0}
    fila = deque([s])
    while fila:
        u = fila.popleft()
        du = dist[u] + 1
        for w in adj[u]:
            if w not in dist:
                dist[w] = du
                fila.append(w)
    return dist


# ---------------------------------------------------------------- DFS  O(V+E)
def dfs(adj, s):
    """DFS iterativa (pilha explícita, evita estouro de recursão).
    Retorna a ordem de descoberta e o pai de cada vértice na árvore DFS."""
    pai = {s: None}
    ordem = []
    visitado = set()
    pilha = [s]
    while pilha:
        u = pilha.pop()
        if u in visitado:
            continue
        visitado.add(u)
        ordem.append(u)
        for w in reversed(adj[u]):
            if w not in visitado:
                pai.setdefault(w, u)
                pilha.append(w)
    return ordem, pai


# ------------------------------------------------- Eulerianidade  O(V+E)
def verificar_euleriano(adj):
    """Teorema de Euler: um grafo conexo (desconsiderando vértices isolados)
    tem circuito euleriano sse todos os graus são pares, e caminho euleriano
    sse exatamente 0 ou 2 vértices têm grau ímpar."""
    nao_isolados = [u for u in adj if adj[u]]
    conexo = True
    if nao_isolados:
        alc = bfs(adj, nao_isolados[0])
        conexo = all(u in alc for u in nao_isolados)
    impares = sum(1 for u in adj if len(adj[u]) % 2 == 1)
    return {
        "conexo": conexo,
        "vertices_grau_impar": impares,
        "circuito_euleriano": conexo and impares == 0,
        "caminho_euleriano": conexo and impares in (0, 2),
    }


# ------------------------------------------------- Dijkstra  O((V+E) log V)
def dijkstra(adj, s, peso=lambda u, v: 1):
    dist = {s: 0}
    heap = [(0, s)]
    fechado = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in fechado:
            continue
        fechado.add(u)
        for w in adj[u]:
            nd = d + peso(u, w)
            if nd < dist.get(w, INF):
                dist[w] = nd
                heapq.heappush(heap, (nd, w))
    return dist


# ------------------------------------------------- Bellman-Ford  O(V·E)
def bellman_ford(adj, s, peso=lambda u, v: 1):
    """Versão clássica: até V-1 rodadas relaxando todos os arcos; para quando
    uma rodada não altera nenhuma distância (parada antecipada). Uma rodada
    extra detecta ciclo negativo."""
    arcos = [(u, w) for u in adj for w in adj[u]]  # cada aresta nos 2 sentidos
    dist = {u: INF for u in adj}
    dist[s] = 0
    rodadas = 0
    for _ in range(len(adj) - 1):
        rodadas += 1
        mudou = False
        for u, w in arcos:
            du = dist[u]
            if du != INF:
                nd = du + peso(u, w)
                if nd < dist[w]:
                    dist[w] = nd
                    mudou = True
        if not mudou:
            break
    ciclo_negativo = any(dist[u] + peso(u, w) < dist[w] for u, w in arcos if dist[u] != INF)
    return dist, rodadas, ciclo_negativo


# ------------------------------------------------- Floyd-Warshall  O(V^3)
def floyd_warshall(adj, nos=None):
    """Matriz de distâncias de todos os pares. O laço em k é o do algoritmo;
    a atualização de cada iteração k (todos os pares i,j) é vetorizada com
    NumPy, o que não muda a complexidade Θ(V^3)."""
    if nos is None:
        nos = sorted(adj)
    idx = {u: i for i, u in enumerate(nos)}
    n = len(nos)
    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0)
    for u in nos:
        for w in adj[u]:
            if w in idx:
                D[idx[u], idx[w]] = 1
    for k in range(n):
        np.minimum(D, D[:, k, None] + D[None, k, :], out=D)
    return D, nos


def floyd_warshall_puro(adj, nos=None):
    """Versão em Python puro (3 laços) — usada só para validar a vetorizada
    em grafos pequenos."""
    if nos is None:
        nos = sorted(adj)
    idx = {u: i for i, u in enumerate(nos)}
    n = len(nos)
    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0
    for u in nos:
        for w in adj[u]:
            if w in idx:
                D[idx[u]][idx[w]] = 1
    for k in range(n):
        Dk = D[k]
        for i in range(n):
            dik = D[i][k]
            if dik == INF:
                continue
            Di = D[i]
            for j in range(n):
                if dik + Dk[j] < Di[j]:
                    Di[j] = dik + Dk[j]
    return D


# ------------------------------------------------- Tarjan (CFC)  O(V+E)
def tarjan_scc(adj_dir):
    """Componentes fortemente conexas (Tarjan, 1972), versão iterativa."""
    indice = {}
    low = {}
    na_pilha = set()
    pilha = []
    sccs = []
    contador = 0
    for raiz in adj_dir:
        if raiz in indice:
            continue
        trabalho = [(raiz, 0)]
        while trabalho:
            v, i = trabalho.pop()
            if i == 0:
                indice[v] = low[v] = contador
                contador += 1
                pilha.append(v)
                na_pilha.add(v)
            recursou = False
            viz = adj_dir[v]
            while i < len(viz):
                w = viz[i]
                i += 1
                if w not in indice:
                    trabalho.append((v, i))
                    trabalho.append((w, 0))
                    recursou = True
                    break
                elif w in na_pilha:
                    low[v] = min(low[v], indice[w])
            if recursou:
                continue
            if low[v] == indice[v]:
                comp = []
                while True:
                    w = pilha.pop()
                    na_pilha.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
            if trabalho:
                pai = trabalho[-1][0]
                low[pai] = min(low[pai], low[v])
    return sccs


# ------------------------------------------------- Kruskal  O(E log E)
def kruskal(adj, peso=lambda u, v: 1):
    pai = {u: u for u in adj}
    rank = {u: 0 for u in adj}

    def find(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    arestas = sorted((peso(u, w), u, w) for u in adj for w in adj[u] if u < w)
    arvore = []
    total = 0
    for p, u, w in arestas:
        ru, rw = find(u), find(w)
        if ru != rw:
            if rank[ru] < rank[rw]:
                ru, rw = rw, ru
            pai[rw] = ru
            if rank[ru] == rank[rw]:
                rank[ru] += 1
            arvore.append((u, w)); total += p
            if len(arvore) == len(adj) - 1:
                break
    return arvore, total


# ------------------------------------------------- Prim  O(E log V)
def prim(adj, s, peso=lambda u, v: 1):
    na_arvore = {s}
    heap = [(peso(s, w), s, w) for w in adj[s]]
    heapq.heapify(heap)
    arvore = []
    total = 0
    while heap and len(na_arvore) < len(adj):
        p, u, w = heapq.heappop(heap)
        if w in na_arvore:
            continue
        na_arvore.add(w)
        arvore.append((u, w)); total += p
        for x in adj[w]:
            if x not in na_arvore:
                heapq.heappush(heap, (peso(w, x), w, x))
    return arvore, total
