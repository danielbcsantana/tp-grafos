#!/usr/bin/env bash
# Executa todo o pipeline do trabalho na ordem.
set -e
cd "$(dirname "$0")"
python3 src/01_tratamento.py
python3 src/02_parte1_estrutural.py
python3 src/03_parte2_algoritmos.py
python3 src/04_parte3_estrutural.py
python3 src/05_descoberta.py
# Opcional (~20-30 min): Floyd-Warshall no grafo completo, execução única
if [ "$1" == "--fw-completo" ]; then
  python3 src/03b_floyd_warshall_completo.py
fi
