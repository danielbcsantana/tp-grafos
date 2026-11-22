"""Etapa 1 — Tratamento dos dados do p2p-Gnutella06.

Tratamentos:
  1. Remoção das linhas de comentário do arquivo SNAP.
  2. Remoção de auto-laços (u == v), se existirem.
  3. Grafo dirigido -> não direcionado (u->v e v->u viram uma única aresta {u,v}).
  4. Multigrafo -> grafo simples (arestas repetidas são removidas).
  5. Extração da maior componente conexa (LCC), usada nas análises de
     distância, small-world e robustez.
O grafo dirigido simples também é salvo, pois o algoritmo de Tarjan (componentes
fortemente conexas) só faz sentido na versão dirigida.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from grafo import (RAW, UNDIR, LCC, DIRECIONADO, RESULTS, PROC_DIR, ler_arestas_raw,
                   salvar_edgelist, adjacencia, componentes)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)
    bruto = ler_arestas_raw(RAW)
    nos_brutos = {u for e in bruto for u in e}
    laços = sum(1 for u, v in bruto if u == v)
    dirigidas = {(u, v) for u, v in bruto if u != v}
    duplicadas_dir = len(bruto) - laços - len(dirigidas)
    recipro = sum(1 for (u, v) in dirigidas if (v, u) in dirigidas) // 2

    nao_dir = sorted({(min(u, v), max(u, v)) for u, v in dirigidas})
    adj = adjacencia(nao_dir, nos_brutos)
    comps = componentes(adj)
    lcc = set(comps[0])
    arestas_lcc = [(u, v) for u, v in nao_dir if u in lcc]

    salvar_edgelist(sorted(dirigidas), DIRECIONADO,
                    "p2p-Gnutella06 dirigido simples (sem lacos/duplicatas)\nFromNodeId\tToNodeId")
    salvar_edgelist(nao_dir, UNDIR,
                    "p2p-Gnutella06 nao direcionado simples\nu\tv")
    salvar_edgelist(arestas_lcc, LCC,
                    "p2p-Gnutella06 nao direcionado simples - maior componente conexa\nu\tv")

    resumo = {
        "bruto_vertices": len(nos_brutos),
        "bruto_arestas_dirigidas": len(bruto),
        "auto_lacos_removidos": laços,
        "arestas_dirigidas_duplicadas": duplicadas_dir,
        "pares_reciprocos_u_v_e_v_u": recipro,
        "nao_dir_vertices": len(adj),
        "nao_dir_arestas": len(nao_dir),
        "nao_dir_componentes": len(comps),
        "lcc_vertices": len(lcc),
        "lcc_arestas": len(arestas_lcc),
        "lcc_fracao_vertices": len(lcc) / len(adj),
    }
    with open(os.path.join(RESULTS, "tratamento.json"), "w") as f:
        json.dump(resumo, f, indent=2)
    print(json.dumps(resumo, indent=2))


if __name__ == "__main__":
    main()
