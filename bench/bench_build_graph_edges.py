"""Micro-benchmark e teste de equivalencia para build_graph_edges.

Tudo roda fora do repositorio microgene-bioai. Nada aqui importa, escreve ou
altera o repositorio: a implementacao de referencia abaixo foi COPIADA
verbatim de stages/s02_data.py:648-667 (revisao
2026-09-25-expanded-domain-direct-supervision), com a unica mudanca de
receber `dilations` como argumento obrigatorio em vez do default CFG.

Uso:
    .venv/Scripts/python.exe bench/bench_build_graph_edges.py
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
import torch

DILATIONS = (1, 2, 4, 8, 16)  # CFG.graph_dilations, stages/s00_bootstrap.py:176


# ---------------------------------------------------------------------------
# REFERENCIA — copia verbatim de stages/s02_data.py:648
# ---------------------------------------------------------------------------
def build_graph_edges(x: torch.Tensor, node_mask: torch.Tensor, potential_valid: torch.Tensor,
                      dilations: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    B, N = x.shape; pairs = []; attrs = []
    for b in range(B):
        n = int(node_mask[b].sum()); length = float(x[b, n-1]-x[b, 0]); local = x[b, 1:n]-x[b, :n-1]
        for dilation in (0, *dilations):
            if dilation == 0:
                iterator = ((i, i) for i in range(n))
            else:
                iterator = ((i, i+dilation) for i in range(max(0, n-dilation)))
            for i, j in iterator:
                directions = ((i, j),) if i == j else ((i, j), (j, i))
                for src, dst in directions:
                    dx = float(x[b, dst]-x[b, src]); adx = abs(dx); hs = float(local[min(src, n-2)]) if n > 1 else length
                    hd = float(local[min(dst, n-2)]) if n > 1 else length
                    pairs.append((b*N+src, b*N+dst))
                    attrs.append((dx/max(length, 1e-12), adx/max(length, 1e-12), math.log(adx/max(length, 1e-12)+1e-8),
                                  0.0 if dx == 0 else math.copysign(1.0, dx), dilation/max(max(dilations), 1),
                                  math.log((hs+1e-12)/(hd+1e-12)), float(potential_valid[b, src] and potential_valid[b, dst])))
    return torch.tensor(pairs, dtype=torch.long).T.contiguous(), torch.tensor(attrs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# PROPOSTA — topologia em cache por (n, dilations); atributos vetorizados
# ---------------------------------------------------------------------------
_TOPOLOGY_CACHE: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _topology(n: int, dilations: tuple[int, ...]):
    """Indices de aresta e coluna de dilatacao. Dependem so de n e das dilatacoes."""
    key = (n, dilations)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached
    scale = max(max(dilations), 1)
    srcs, dsts, dilcols = [], [], []
    for dilation in (0, *dilations):
        if dilation == 0:
            index = np.arange(n, dtype=np.int64)
            srcs.append(index); dsts.append(index)
            dilcols.append(np.full(n, dilation/scale, dtype=np.float64))
        else:
            count = max(0, n-dilation)
            i = np.arange(count, dtype=np.int64); j = i+dilation
            # Ordem do laco original: (i,j) e (j,i) intercalados, par a par.
            srcs.append(np.stack([i, j], axis=1).ravel())
            dsts.append(np.stack([j, i], axis=1).ravel())
            dilcols.append(np.full(2*count, dilation/scale, dtype=np.float64))
    built = (np.concatenate(srcs), np.concatenate(dsts), np.concatenate(dilcols))
    _TOPOLOGY_CACHE[key] = built
    return built


def build_graph_edges_fast(x: torch.Tensor, node_mask: torch.Tensor, potential_valid: torch.Tensor,
                           dilations: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    B, N = x.shape
    dilations = tuple(dilations)
    pair_blocks, attr_blocks = [], []
    for b in range(B):
        n = int(node_mask[b].sum())
        row = x[b]
        # length e local sao subtracoes em float32 ANTES do upcast, como no original.
        length = float(row[n-1]-row[0])
        local64 = (row[1:n]-row[:n-1]).to(torch.float64).numpy() if n > 1 else None
        src, dst, dilcol = _topology(n, dilations)
        src_t = torch.from_numpy(src); dst_t = torch.from_numpy(dst)
        dx = (row[dst_t]-row[src_t]).to(torch.float64).numpy()
        adx = np.abs(dx)
        denom = max(length, 1e-12)
        if n > 1:
            hs = local64[np.minimum(src, n-2)]; hd = local64[np.minimum(dst, n-2)]
        else:
            hs = np.full(src.shape, length, dtype=np.float64); hd = hs
        valid = potential_valid[b].numpy()
        attr_blocks.append(np.stack([
            dx/denom,
            adx/denom,
            np.log(adx/denom + 1e-8),
            np.where(dx == 0, 0.0, np.copysign(1.0, dx)),
            dilcol,
            np.log((hs+1e-12)/(hd+1e-12)),
            (valid[src] & valid[dst]).astype(np.float64),
        ], axis=1))
        pair_blocks.append(np.stack([b*N+src, b*N+dst], axis=1))
    pairs = np.concatenate(pair_blocks, axis=0)
    attrs = np.concatenate(attr_blocks, axis=0)
    return torch.from_numpy(pairs).T.contiguous(), torch.tensor(attrs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Geracao de lotes sinteticos parecidos com os reais
# ---------------------------------------------------------------------------
def make_batch(lengths: Sequence[int], seed: int, geometry: str = "nonuniform"):
    """Monta (x, node_mask, potential_valid) como collate_records faria."""
    rng = np.random.default_rng(seed)
    B = len(lengths); N = max(lengths)
    x = torch.zeros(B, N); node_mask = torch.zeros(B, N, dtype=torch.bool)
    potential_valid = torch.zeros(B, N, dtype=torch.bool)
    for b, n in enumerate(lengths):
        if geometry == "uniform":
            grid = np.linspace(-12.0, 12.0, n)
        elif geometry == "radial":           # origem em zero, malha adensada perto dela
            grid = np.linspace(0.0, 1.0, n)**1.6 * 20.0
        else:                                 # espacamento irregular, como a malha FEM
            steps = rng.uniform(0.6, 1.4, size=n-1)
            grid = np.concatenate([[0.0], np.cumsum(steps)]) - 8.0
        x[b, :n] = torch.as_tensor(grid, dtype=torch.float32)
        node_mask[b, :n] = True
        valid = rng.random(n) > 0.05          # alguns pontos invalidos, como em singularidades
        valid[0] = True
        potential_valid[b, :n] = torch.as_tensor(valid)
    return x, node_mask, potential_valid


# ---------------------------------------------------------------------------
# Teste de equivalencia
# ---------------------------------------------------------------------------
def equivalence() -> bool:
    print("=" * 78)
    print("TESTE DE EQUIVALENCIA")
    print("=" * 78)
    cases = []
    for geometry in ("nonuniform", "uniform", "radial"):
        for lengths in ([2], [3], [17], [65], [129], [129, 129], [129, 65],
                        [257, 257], [257, 200, 65], [513, 513]):
            cases.append((geometry, tuple(lengths)))
    cases.append(("nonuniform", (1,)))        # caso degenerado n=1
    cases.append(("nonuniform", (1, 129)))

    all_ok = True
    worst_attr = 0.0
    for seed, (geometry, lengths) in enumerate(cases):
        x, node_mask, potential_valid = make_batch(lengths, seed=seed, geometry=geometry)
        ref_index, ref_attr = build_graph_edges(x, node_mask, potential_valid, DILATIONS)
        new_index, new_attr = build_graph_edges_fast(x, node_mask, potential_valid, DILATIONS)
        index_ok = torch.equal(ref_index, new_index)
        attr_ok = torch.equal(ref_attr, new_attr)
        diff = float((ref_attr-new_attr).abs().max()) if ref_attr.numel() else 0.0
        worst_attr = max(worst_attr, diff)
        status = "OK " if (index_ok and attr_ok) else "FALHA"
        if not (index_ok and attr_ok):
            all_ok = False
            mismatches = int((ref_attr != new_attr).any(dim=1).sum())
            print(f"  {status} {geometry:>10} n={list(lengths)} arestas={ref_index.shape[1]} "
                  f"index_igual={index_ok} attr_igual={attr_ok} linhas_diferentes={mismatches} maxdiff={diff:.3e}")
        else:
            print(f"  {status} {geometry:>10} n={str(list(lengths)):<18} arestas={ref_index.shape[1]:>6} "
                  f"maxdiff={diff:.1e}")
    print(f"\n  Casos: {len(cases)} | Equivalencia exata: {all_ok} | Maior diferenca de atributo: {worst_attr:.3e}")
    return all_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def timeit(fn, *args, repeats: int, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter()-start)
    return best


def benchmark() -> None:
    print()
    print("=" * 78)
    print("BENCHMARK  (melhor de N repeticoes, 1 nucleo, CPU local)")
    print("=" * 78)
    print(f"{'modo':<14}{'n':>6}{'lote':>6}{'arestas':>9}{'ref (ms)':>11}{'novo (ms)':>11}{'ganho':>9}")
    scenarios = [
        ("QUICK/DEV", 129, 2), ("QUICK/DEV", 129, 8),
        ("STANDARD", 257, 2), ("STANDARD", 257, 8),
        ("FULL", 513, 2), ("FULL", 513, 8),
    ]
    rows = []
    for name, n, batch in scenarios:
        x, node_mask, potential_valid = make_batch([n]*batch, seed=99)
        repeats = 20 if n <= 257 else 10
        ref_ms = timeit(build_graph_edges, x, node_mask, potential_valid, DILATIONS, repeats=repeats)*1e3
        new_ms = timeit(build_graph_edges_fast, x, node_mask, potential_valid, DILATIONS, repeats=repeats)*1e3
        edges = build_graph_edges(x, node_mask, potential_valid, DILATIONS)[0].shape[1]
        print(f"{name:<14}{n:>6}{batch:>6}{edges:>9}{ref_ms:>11.2f}{new_ms:>11.3f}{ref_ms/new_ms:>8.0f}x")
        rows.append((name, n, batch, ref_ms, new_ms))

    print()
    print("=" * 78)
    print("EXTRAPOLACAO POR EPOCA  (so a construcao de arestas)")
    print("=" * 78)
    # Splits reais (outputs/*/run_card.json -> split_sizes) e tempo de epoca medido
    # na GPU deles (outputs/*/training_history.csv -> epoch_seconds, mediana).
    splits = {
        "DEVELOPMENT (n=129)": (129, 594, 135, 263.3),
        "STANDARD    (n=257)": (257, 2430, 513, 982.9),
    }
    print(f"{'execucao':<22}{'ms/amostra':>12}{'treino (s)':>12}{'+valid (s)':>12}"
          f"{'epoca real':>12}{'novo (s)':>10}")
    for label, (n, train, val, measured) in splits.items():
        per_sample = {}
        for key, fn in (("ref", build_graph_edges), ("new", build_graph_edges_fast)):
            x, node_mask, potential_valid = make_batch([n]*2, seed=7)
            per_sample[key] = timeit(fn, x, node_mask, potential_valid, DILATIONS, repeats=10)/2
        # treino: 1x por epoca. validacao: 2x (global + por familia, s05_training.py:259)
        train_s = per_sample["ref"]*train
        full_s = per_sample["ref"]*(train+2*val)
        new_s = per_sample["new"]*(train+2*val)
        print(f"{label:<22}{per_sample['ref']*1e3:>12.1f}{train_s:>12.1f}{full_s:>12.1f}"
              f"{measured:>12.1f}{new_s:>10.2f}")
    print()
    print("Como ler a tabela:")
    print("  'epoca real' e o tempo medido de uma epoca inteira na maquina deles,")
    print("  incluindo o calculo na GPU. As colunas 'treino' e '+valid' sao o custo")
    print("  SO da construcao de arestas, estimado nesta CPU local. Quando elas se")
    print("  aproximam da epoca real, a construcao de arestas domina a epoca.")
    print()
    print("Ressalvas:")
    print("  - Mede SO build_graph_edges. Rotulos de topologia, augmentacoes e copias")
    print("    profundas sao custos de preparacao adicionais, nao medidos aqui.")
    print("  - CPUs diferentes: o tempo absoluto nao transfere, o ganho relativo sim.")
    print("    Se '+valid' passar da 'epoca real', a CPU da nuvem e mais rapida que")
    print("    esta, ou epoch_seconds nao inclui a validacao. A conclusao nao muda:")
    print("    a ordem de grandeza e a mesma da epoca inteira.")
    print("  - A validacao conta 2x porque s05_training.py:259 refaz a passada por")
    print("    familia depois da global.")
    print("  - A epoca real citada vem da revisao 2026-07-20 (STANDARD); a arvore de")
    print("    trabalho esta em 2026-09-25. O custo atual precisa ser remedido.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    ok = equivalence()
    benchmark()
    print()
    print("RESULTADO:", "equivalencia exata confirmada" if ok else "DIVERGENCIA — nao aplicar")
    raise SystemExit(0 if ok else 1)
