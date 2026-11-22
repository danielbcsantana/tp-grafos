"""Parte III, pergunta 4 — a descoberta mais interessante.

Separa os vértices pelo grau de SAÍDA no grafo dirigido original:
  - grau de saída 0: peers apenas citados (descobertos, mas não rastreados);
  - grau de saída > 0: peers efetivamente rastreados pelo crawler.
Mostra que a distribuição bimodal de graus é a mistura dessas duas populações.
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
from grafo import ler_edgelist, DIRECIONADO, RESULTS, FIGS

e = ler_edgelist(DIRECIONADO)
out = Counter(u for u, v in e)
inn = Counter(v for u, v in e)
nos = set(out) | set(inn)
grau = {u: out[u] + inn[u] for u in nos}
nao_rastreados = [u for u in nos if out[u] == 0]
rastreados = [u for u in nos if out[u] > 0]
dist_out = Counter(out[u] for u in rastreados)

res = {
    "V": len(nos),
    "nao_rastreados": len(nao_rastreados),
    "frac_nao_rastreados": len(nao_rastreados) / len(nos),
    "rastreados": len(rastreados),
    "grau_medio_nao_rastreados": float(np.mean([grau[u] for u in nao_rastreados])),
    "grau_mediano_nao_rastreados": float(np.median([grau[u] for u in nao_rastreados])),
    "grau_medio_rastreados": float(np.mean([grau[u] for u in rastreados])),
    "grau_mediano_rastreados": float(np.median([grau[u] for u in rastreados])),
    "arestas_entre_nao_rastreados": sum(1 for u, v in e if out[u] == 0 and out[v] == 0),
    "rastreados_com_saida_10": dist_out[10],
    "frac_rastreados_com_saida_10": dist_out[10] / len(rastreados),
    "grau_saida_max": max(out.values()),
    "vertices_grau_ge_30": sum(1 for u in nos if grau[u] >= 30),
    "media_entrada_grau_ge_30": float(np.mean([inn[u] for u in nos if grau[u] >= 30])),
}
with open(os.path.join(RESULTS, "descoberta.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
ks = np.arange(0, 41)
a = Counter(grau[u] for u in nao_rastreados)
b = Counter(grau[u] for u in rastreados)
axs[0].bar(ks, [a[k] for k in ks], color="#e08a2c", label=f"grau de saída = 0 (n={len(nao_rastreados)})")
axs[0].bar(ks, [b[k] for k in ks], bottom=[a[k] for k in ks], color="#3b6ea5",
           label=f"grau de saída > 0 (n={len(rastreados)})")
axs[0].set_xlabel("grau total k"); axs[0].set_ylabel("número de vértices")
axs[0].set_title("Distribuição de graus por tipo de vértice")
axs[0].legend(fontsize=7)
ko = np.arange(1, 31)
axs[1].bar(ko, [dist_out[k] for k in ko], color="#3b6ea5")
axs[1].set_xlabel("grau de saída"); axs[1].set_ylabel("número de vértices")
axs[1].set_title("Grau de saída dos vértices com saída > 0")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "descoberta_crawler.png"), dpi=200)
plt.close(fig)
