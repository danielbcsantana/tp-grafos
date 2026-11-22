# TP — Analisando Grafos do SNAP (Teoria dos Grafos, UFBA)

**Aluno:** Daniel Santana — matrícula 224216274
**Professor:** Bruno Pereira dos Santos
**Grafo atribuído:** G57 — [`p2p-Gnutella06`](https://snap.stanford.edu/data/p2p-Gnutella06.html)
(rede peer-to-peer Gnutella de 6 de agosto de 2002; 8.717 vértices, 31.525 arestas dirigidas)

### Determinação do grafo

```
M = 224216274
soma dos dígitos = 2+2+4+2+1+6+2+7+4 = 30
últimos 2 dígitos = 74
f(M) = (30 × (74 + 1)) mod 129 = 2250 mod 129 = 57  →  G57 = p2p-Gnutella06
```

## Estrutura

```
data/
  raw/p2p-Gnutella06.txt.gz                     arquivo original do SNAP
  processed/gnutella06_direcionado.edgelist     dirigido simples (usado no Tarjan)
  processed/gnutella06_nao_direcionado.edgelist não direcionado simples (todas as análises)
  processed/gnutella06_lcc.edgelist             maior componente conexa (= grafo todo)
src/
  grafo.py                         leitura, lista de adjacência, componentes
  algoritmos.py                    BFS, DFS, Euler, Dijkstra, Bellman-Ford,
                                   Floyd-Warshall, Tarjan, Kruskal, Prim (implementação própria)
  01_tratamento.py                 tratamento dos dados
  02_parte1_estrutural.py          Parte I  — métricas estruturais + figuras
  03_parte2_algoritmos.py          Parte II — corretude, tempos (30 execuções, IC 95%) e escalabilidade
  03b_floyd_warshall_completo.py   Floyd-Warshall no grafo inteiro (execução única, ~20-30 min)
  04_parte3_estrutural.py          Parte III — small-world, lei de potência e robustez
  05_descoberta.py                 Parte III (4) — descoberta: vértices rastreados × não rastreados
results/
  *.json, distribuicao_graus.csv   resultados numéricos
  figures/*.png                    gráficos usados no relatório
relatorio/
  relatorio.tex, refs.bib, relatorio.pdf  relatório (ACM sigconf / WebMedia)
```

## Como executar

Requer Python ≥ 3.10.

```bash
pip install -r requirements.txt
bash run_all.sh                 # pipeline completo (~15-25 min)
bash run_all.sh --fw-completo   # + Floyd-Warshall no grafo inteiro (~20-30 min a mais)
```

Ou execute cada etapa em ordem: `python3 src/01_tratamento.py`, depois `02_…`, `03_…`, `04_…`.
Cada script grava seus resultados em `results/` e as figuras em `results/figures/`.
As sementes aleatórias são fixas (`SEED = 2026`), então os resultados são reproduzíveis.

Para compilar o relatório: `cd relatorio && pdflatex relatorio && bibtex relatorio && pdflatex relatorio && pdflatex relatorio`.

## Tratamento realizado

1. Remoção das linhas de comentário (`#`) do arquivo SNAP.
2. Verificação de auto-laços e arestas repetidas (não havia nenhum).
3. Dirigido → não direcionado: cada arco u→v vira a aresta {u, v}. O arquivo não tem pares recíprocos, então nenhuma aresta foi fundida (31.525 arestas).
4. Extração da maior componente conexa. O grafo não direcionado já é conexo, então a LCC é o grafo inteiro.
5. A versão dirigida simples foi mantida apenas para o algoritmo de Tarjan (componentes fortemente conexas).

## Observações de implementação

- Os algoritmos da Parte II e o cálculo de small-world (L, C e o grafo aleatório G(n, m)) são **implementação própria** em Python puro.
  O NetworkX é usado só para conferência (clusterização, triângulos, CFCs) e para desenhar a visualização.
- O grafo não tem pesos. Dijkstra, Bellman-Ford, Floyd-Warshall, Prim e Kruskal usam peso unitário.
- O Floyd-Warshall vetoriza com NumPy a atualização de cada iteração `k`; a complexidade continua Θ(V³).
  A versão vetorizada é validada contra uma versão com três laços em Python puro.
- Para as distâncias de todos os pares da Parte I e da robustez, o BFS é executado via `scipy.sparse.csgraph`
  (BFS em C). O mesmo valor de L é obtido pelo BFS próprio em `04_parte3_estrutural.py`.
