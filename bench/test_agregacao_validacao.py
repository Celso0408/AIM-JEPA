"""Equivalencia da agregacao de validacao: duas passadas (atual) contra uma (proposta).

Nao carrega o modelo nem o repositorio. Reproduz SO a aritmetica de agregacao de
stages/s05_training.py:240-269, porque e ela que muda. Cada termo de perda e uma
media sobre o lote; a acumulacao pesa cada lote pelo seu tamanho.

Uso:
    .venv/Scripts/python.exe bench/test_agregacao_validacao.py
"""

from __future__ import annotations

import math
import random

EVALUATION_BATCH_SIZE = 2  # s00_bootstrap.py:382 -> min(CFG.batch_size, 2)


def validation_metric_score(metrics: dict[str, float]) -> float:
    """Copia de s05_training.py:235."""
    return ((1-float(metrics["fidelity"]))+.10*float(metrics["h1"])+.05*float(metrics["energy"]) +
            .05*float(metrics["residual"])+.02*float(metrics["orthogonality"]))


KEYS = ("fidelity", "h1", "energy", "residual", "orthogonality")


def batch_terms(rows, indices):
    """O que compute_losses devolveria: uma media por termo sobre o lote."""
    return {key: sum(rows[i][key] for i in indices)/len(indices) for key in KEYS}


def accumulate(rows, batches):
    """Acumulacao ponderada identica a dos dois caminhos."""
    totals, count = {}, 0
    for indices in batches:
        weight = len(indices)
        terms = batch_terms(rows, indices)
        count += weight
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0)+weight*value
    return {key: value/max(count, 1) for key, value in totals.items()}


def chunk(sequence, size):
    size = min(size, max(1, len(sequence)))
    return [sequence[start:start+size] for start in range(0, len(sequence), size)]


def caminho_atual(rows):
    """Passada global em ordem de dataset, mais uma passada por familia."""
    order = list(range(len(rows)))
    global_metrics = accumulate(rows, chunk(order, EVALUATION_BATCH_SIZE))
    families = sorted({r["family"] for r in rows})
    per_family = {}
    for family in families:
        indices = [i for i in order if rows[i]["family"] == family]
        per_family[family] = accumulate(rows, chunk(indices, EVALUATION_BATCH_SIZE))
    return global_metrics, per_family


def caminho_proposto(rows):
    """Uma passada so, com lotes que nunca misturam familias."""
    grouped = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["family"], []).append(i)
    batches = []
    for family in sorted(grouped):
        for part in chunk(grouped[family], EVALUATION_BATCH_SIZE):
            batches.append((family, part))

    totals, count = {}, 0
    group_totals, group_counts = {}, {}
    for family, indices in batches:
        weight = len(indices)
        terms = batch_terms(rows, indices)
        count += weight
        bucket = group_totals.setdefault(family, {})
        group_counts[family] = group_counts.get(family, 0)+weight
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0)+weight*value
            bucket[key] = bucket.get(key, 0.0)+weight*value
    global_metrics = {key: value/max(count, 1) for key, value in totals.items()}
    per_family = {family: {key: value/max(group_counts[family], 1) for key, value in bucket.items()}
                  for family, bucket in group_totals.items()}
    return global_metrics, per_family


def robustez(global_metrics, per_family):
    """Copia de s05_training.py:261-268."""
    scores = [validation_metric_score(per_family[f]) for f in sorted(per_family)]
    ordered = sorted(scores, reverse=True)
    hard = max(1, math.ceil(.25*len(ordered)))
    glob = validation_metric_score(global_metrics)
    return {
        "global": glob,
        "hard": sum(ordered[:hard])/hard,
        "worst": ordered[0],
        "count": float(len(ordered)),
        "robust": .55*glob + .30*(sum(ordered[:hard])/hard) + .15*ordered[0],
    }


def cenario(n_rows, n_families, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n_rows):
        rows.append({
            "family": f"familia_{i % n_families:02d}",
            "fidelity": rng.uniform(.80, .999),
            "h1": rng.uniform(.01, .30),
            "energy": rng.uniform(1e-4, .05),
            "residual": rng.uniform(1e-3, .20),
            "orthogonality": rng.uniform(1e-6, .01),
        })
    return rows


def main() -> int:
    print("=" * 76)
    print("EQUIVALENCIA DA AGREGACAO DE VALIDACAO")
    print("=" * 76)
    print(f"{'cenario':<26}{'familias exatas':>17}{'erro global':>14}{'erro robusto':>14}")
    cenarios = [
        ("STANDARD  513x27", 513, 27, 1),
        ("DEVELOPMENT 135x27", 135, 27, 2),
        ("familia com 1 linha", 28, 27, 3),
        ("2 familias", 64, 2, 4),
        ("1 familia", 40, 1, 5),
        ("impar, lote parcial", 41, 7, 6),
    ]
    ok = True
    for label, n_rows, n_families, seed in cenarios:
        rows = cenario(n_rows, n_families, seed)
        g_old, f_old = caminho_atual(rows)
        g_new, f_new = caminho_proposto(rows)

        exatas = all(f_old[f][k] == f_new[f][k] for f in f_old for k in KEYS)
        erro_global = max(abs(g_old[k]-g_new[k]) for k in KEYS)
        r_old, r_new = robustez(g_old, f_old), robustez(g_new, f_new)
        erro_robusto = max(abs(r_old[k]-r_new[k]) for k in r_old)
        if not exatas or erro_global > 1e-12 or erro_robusto > 1e-12:
            ok = False
        print(f"{label:<26}{str(exatas):>17}{erro_global:>14.2e}{erro_robusto:>14.2e}")

    print()
    print("Leitura:")
    print("  'familias exatas' = as metricas por familia sao bit-identicas, porque os lotes")
    print("  por familia sao exatamente os mesmos de hoje (mesmas linhas, mesma ordem,")
    print("  mesmo tamanho de lote).")
    print("  'erro global' = a metrica global muda so por ordem de soma, porque os lotes")
    print("  passam a nao misturar familias. A diferenca e de arredondamento.")
    print()
    print("Ressalva: este teste cobre a ARITMETICA de agregacao. O valor de cada termo")
    print("  vem do modelo e tem o seu proprio arredondamento em float32, que depende da")
    print("  composicao do lote. Por isso a metrica global nao e bit-exata na pratica,")
    print("  e a verificacao final e um DEVELOPMENT pareado, antes e depois.")
    print()
    print("RESULTADO:", "equivalencia confirmada" if ok else "DIVERGENCIA")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
