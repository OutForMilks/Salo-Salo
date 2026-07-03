from collections import Counter

def temperature_sampling_weights(lang_file, temperature=5.0):
    with open(lang_file, encoding="utf-8") as f:
        langs = [line.strip() for line in f if line.strip()]

    counts = Counter(langs)
    n_total = len(langs)

    # p_l = natural probability, q_l = temperature-scaled probability
    p = {l: c / n_total for l, c in counts.items()}
    q_unnorm = {l: p[l] ** (1.0 / temperature) for l in p}
    z = sum(q_unnorm.values())
    q = {l: v / z for l, v in q_unnorm.items()}

    lang_weight = {l: q[l] / counts[l] for l in counts}
    weights = [lang_weight[l] for l in langs]

    print("Language distribution (natural -> temperature-scaled):")
    for l in counts:
        print(f"  {l:12s} n={counts[l]:6d}  p={p[l]:.4f}  q={q[l]:.4f}")

    return weights