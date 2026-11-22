"""Floyd-Warshall (Θ(V^3)) no grafo completo, executado uma única vez
(30 repetições seriam inviáveis: ~20 min cada). Serve para validar o
diâmetro e o comprimento médio obtidos na Parte I."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from grafo import carregar_nao_direcionado, RESULTS
import algoritmos as alg

adj = carregar_nao_direcionado()
V = len(adj)
t0 = time.perf_counter()
D, _ = alg.floyd_warshall(adj)
t = time.perf_counter() - t0
out = {"V": V, "tempo_s": t, "diametro": float(D.max()),
       "raio": float(D.max(axis=1).min()),
       "comprimento_medio": float(D.astype(np.float64).sum() / (V * (V - 1)))}
print(out)
with open(os.path.join(RESULTS, "floyd_warshall_completo.json"), "w") as f:
    json.dump(out, f, indent=2)
