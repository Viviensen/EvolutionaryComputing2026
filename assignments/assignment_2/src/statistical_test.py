import numpy as np
from scipy import stats

# Final best distance results from the experiments (mean, std, n) for each group
groups = {
    "sigma_0.05": {"mean": 0.8977, "std": 0.4122, "n": 10},
    "sigma_0.20": {"mean": 0.9828, "std": 0.1761, "n": 10},
    "sigma_0.50": {"mean": 1.4031, "std": 0.2213, "n": 10},
    "Random Search": {"mean": 1.4110, "std": 0.1064, "n": 10},
}

# One-way ANOVA
means = [v["mean"] for v in groups.values()]
stds = [v["std"] for v in groups.values()]
ns = [v["n"] for v in groups.values()]
k = len(groups)
total_n = sum(ns)

grand_mean = sum(n * m for n, m in zip(ns, means)) / total_n
ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(ns, means))
ss_within = sum((n - 1) * (s**2) for n, s in zip(ns, stds))

df_between = k - 1
df_within = total_n - k
ms_between = ss_between / df_between
ms_within = ss_within / df_within

f_stat = ms_between / ms_within
p_anova = stats.f.sf(f_stat, df_between, df_within)

print(f"One-way ANOVA:({df_between}, {df_within}) = {f_stat:.2f}, p = {p_anova:.4e}")


# Welch's two-sample t-tests (with Bonferroni correction)
comparisons = [
    ("sigma_0.05", "Random Search"),
    ("sigma_0.20", "Random Search"),
    ("sigma_0.50", "Random Search"),
    ("sigma_0.05", "sigma_0.50"),
    ("sigma_0.20", "sigma_0.50"),
    ("sigma_0.05", "sigma_0.20"),
]

num_comparisons = len(comparisons)

print(f"Pairwise Welch's T-TESTS (with Bonferroni correction, m={num_comparisons}):")
print(f"{'Comparison':<30} | {'t-value':<8} | {'df':<6} | {'p (raw)':<10} | {'p (Bonferroni)':<12}")

for g1, g2 in comparisons:
    m1, s1, n1 = groups[g1]["mean"], groups[g1]["std"], groups[g1]["n"]
    m2, s2, n2 = groups[g2]["mean"], groups[g2]["std"], groups[g2]["n"]

    diff = m1 - m2
    se_diff = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    t_val = diff / se_diff

    # Welch-Satterthwaite degrees of freedom
    df_welch = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
    )

    # two-tailed p-value
    p_raw = 2 * stats.t.sf(np.abs(t_val), df=df_welch)
    p_bonf = min(p_raw * num_comparisons, 1.0)

    print(
        f"{g1 + ' vs ' + g2:<30} | {t_val:>8.2f} | {df_welch:>6.1f} | {p_raw:>10.4f} | {p_bonf:>12.4f}"
    )
