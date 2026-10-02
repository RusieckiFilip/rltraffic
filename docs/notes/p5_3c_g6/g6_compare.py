"""Gate G6 (BRIEF_42 Amendment E): the artifact's confirmatory, att_ours and mix50 families against the coordinator's
independent numbers (g6_statistic.py's g6_independent.json, computed FIRST), under == where the reduction order is
registered and to 1e-12 on the normal CDF. Run from the main tree: .venv/bin/python -P docs/notes/p5_3c_g6/g6_compare.py"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
A = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/filip/rltraffic/output/p5_3c/artifacts/p5_3c_context_sweep.json')
raw = A.read_bytes(); print('artifact', A, '| sha256', hashlib.sha256(raw).hexdigest(), '| bytes', len(raw))
art = json.loads(raw); mine = json.load(open(HERE / 'g6_independent.json'))
c = art['confirmatory']
print('OUTCOME', c['outcome'], '| tests_not_rejected', c['tests_not_rejected'], '| arms', c.get('arms'), '| k20_arm', c.get('k20_arm'))
print('SENTENCE:', c['sentence'])
def cmp(label, a, b, tol=0.0):
    ok = (a == b) if tol == 0 else abs(a - b) <= tol
    print(f"  {label}: artifact {a!r} | mine {b!r} | {'EQUAL' if ok else 'DIFFER'}"); return ok
decisions_ok = True; ulp = 0
for key, name in (('conf', 'confirmatory'), ('ours', 'att_ours'), ('mix', 'exploratory_mix50')):
    fam = art[name]['family']; m = mine[key]; print(f'--- {name} ---')
    for label, a, b in (('S.mean', fam['S']['mean'], m['S'][0]), ('S.ci_low', fam['S']['ci95_low'], m['S'][1]), ('S.ci_high', fam['S']['ci95_high'], m['S'][2]),
                        ('G1.mean', fam['G']['1']['mean'], m['G1'][0]), ('G1.ci_low', fam['G']['1']['ci95_low'], m['G1'][1]),
                        ('G2.mean', fam['G']['2']['mean'], m['G2'][0]), ('G2.ci_high', fam['G']['2']['ci95_high'], m['G2'][2])):
        if not cmp(label, a, b): ulp += int(abs(a - b) <= 1e-12)
    for i, t in enumerate(('T1', 'T2', 'T3')):
        decisions_ok &= cmp(f'{t}.p (tol 1e-12)', fam['tests'][t]['p_value'], m['p'][i], tol=1e-12)
        decisions_ok &= cmp(f'{t}.rejected', fam['holm']['rejected'][t], m['rej'][i])
    decisions_ok &= cmp('outcome', fam['outcome'], m['outcome'])
print('EVERY p-value (to 1e-12), DECISION AND OUTCOME EQUAL:', decisions_ok, '| descriptive values differing within 1e-12 (1 ulp):', ulp)
