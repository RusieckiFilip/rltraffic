"""Gate G6 (BRIEF_42 Amendment E), the coordinator's INDEPENDENT route to A26(d): A_d(K), s_d, T1-T3 (one-sided Wilcoxon,
ties averaged, continuity-corrected, normal approximation), Holm, the partition, S / G1 / G2 with 95% CIs -- from the raw
chunks ONLY; the artifact is not read. Shares no code with offline/. Writes g6_independent.json beside itself.
Run from the main tree: .venv/bin/python -P docs/notes/p5_3c_g6/g6_statistic.py"""
import json, math, statistics
from collections import Counter
from pathlib import Path
C = Path('/home/filip/rltraffic/output/p5_3c/cells'); HERE = Path(__file__).resolve().parent
SEEDS = (101, 202, 303, 404, 505); DRAWS = list(range(1000, 1100)); KS = (1, 2, 5, 10, 20); COEF = (-2, -1, 0, 1, 2)
DELTA = 0.6263; ALPHA = 0.05
def val(arm, s, d, defn):
    return float(json.loads((C / f'cell_{arm}_seed{s}_draw{d}.json').read_bytes())['episode'][defn])
def A(arm, defn):
    out = {}
    for d in DRAWS:
        acc = 0.0
        for s in SEEDS: acc = acc + val(arm, s, d, defn)          # ascending seeds, sequential float64 sum, then / 5
        out[d] = acc / 5.0
    return out
def phi(z): return 0.5 * math.erfc(-z / math.sqrt(2.0))
def wilcoxon(diffs, alt):
    d = [x for x in diffs if x != 0.0]; n = len(d); nz = len(diffs) - n
    if n == 0: return dict(n=0, n_zero=nz, w_plus=0.0, E=0.0, var=0.0, z=0.0, p=1.0)
    mags = [abs(x) for x in d]; order = sorted(range(n), key=lambda i: mags[i]); ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and mags[order[j + 1]] == mags[order[i]]: j += 1
        for k in range(i, j + 1): ranks[order[k]] = (i + j + 2) / 2.0
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0); E = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in Counter(mags).values()) / 48.0
    if var <= 0: return dict(n=n, n_zero=nz, w_plus=w_plus, E=E, var=var, z=0.0, p=1.0)
    z = (w_plus - E + (0.5 if alt == 'less' else -0.5)) / math.sqrt(var)
    return dict(n=n, n_zero=nz, w_plus=w_plus, E=E, var=var, z=z, p=phi(z) if alt == 'less' else phi(-z))
def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: (ps[i], i)); rej = [False] * m
    for step, i in enumerate(order):
        if ps[i] <= ALPHA / (m - step): rej[i] = True
        else: break
    return rej
def ci(xs):
    m = statistics.fmean(xs); h = 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)); return m, m - h, m + h
def family(subject, defn, label):
    lv = {k: A(f'{subject}_k{k}_b64', defn) for k in KS}
    s_d = []
    for d in DRAWS:
        tot = 0.0
        for c, k in zip(COEF, KS): tot = tot + c * lv[k][d]
        s_d.append(tot)
    t1 = wilcoxon(s_d, 'less')
    t2 = wilcoxon([(lv[1][d] - lv[20][d]) - DELTA for d in DRAWS], 'greater')
    t3 = wilcoxon([(lv[2][d] - lv[20][d]) - DELTA for d in DRAWS], 'greater')
    ps = [t1['p'], t2['p'], t3['p']]; rej = holm(ps)
    outcome = 'iii' if not rej[0] else ('i' if rej[1] and rej[2] else 'ii')
    S = ci(s_d); G1 = ci([lv[1][d] - lv[20][d] for d in DRAWS]); G2 = ci([lv[2][d] - lv[20][d] for d in DRAWS])
    print(f'===== {label}: {subject} on {defn} =====')
    print('  per-arm mean of A_d:', {k: round(statistics.fmean(lv[k].values()), 4) for k in KS})
    print(f'  S = {S[0]:.6f} [{S[1]:.6f}, {S[2]:.6f}]   (mean s_d; n_zero {t1["n_zero"]})')
    print(f'  G1 = {G1[0]:.6f} [{G1[1]:.6f}, {G1[2]:.6f}]   G2 = {G2[0]:.6f} [{G2[1]:.6f}, {G2[2]:.6f}]')
    for name, t in (('T1 (s<0)', t1), ('T2 (K=1 gap-delta>0)', t2), ('T3 (K=2 gap-delta>0)', t3)):
        print(f'  {name}: W+ {t["w_plus"]:.1f} E {t["E"]:.1f} var {t["var"]:.1f} z {t["z"]:.6f} p {t["p"]:.6e} n {t["n"]} zeros {t["n_zero"]}')
    print(f'  Holm rejected: T1 {rej[0]}, T2 {rej[1]}, T3 {rej[2]}  ->  OUTCOME ({outcome})')
    return dict(S=S, G1=G1, G2=G2, p=ps, rej=rej, outcome=outcome, levels=lv)
res = {'conf': family('mappo1000', 'att_engine', 'CONFIRMATORY'), 'ours': family('mappo1000', 'att_ours', 'co-reported'),
       'mix': family('mix50', 'att_engine', 'EXPLORATORY'), 'mix_ours': family('mix50', 'att_ours', 'exploratory, att_ours')}
for arm, base in (('mappo1000_k1_b1280', 'mappo1000_k1_b64'), ('mappo1000_k2_b640', 'mappo1000_k2_b64')):
    a = A(arm, 'att_engine'); b = A(base, 'att_engine'); k20 = res['conf']['levels'][20]
    d1 = ci([a[d] - b[d] for d in DRAWS]); d2 = ci([a[d] - k20[d] for d in DRAWS])
    print(f'{arm}: mean A_d {statistics.fmean(a.values()):.4f} | minus {base}: {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | minus K=20: {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]')
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'levels'} for k, v in res.items()}, open(HERE / 'g6_independent.json', 'w'), indent=1, default=list)
print('saved:', HERE / 'g6_independent.json')
