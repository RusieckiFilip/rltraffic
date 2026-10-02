"""Gate G6 (BRIEF_42 Amendment E), the coordinator's own route: the reference gate re-checked from the chunks and the
committed rows; mix50's reference arm against P8.4b's cells; the sweep's K = 20 cells beside the reference cells; every
chunk's identity. Reads NO artifact. Run from the main tree: .venv/bin/python -P docs/notes/p5_3c_g6/g6_gate_check.py"""
import hashlib, json
from pathlib import Path
OUT = Path('/home/filip/rltraffic/output'); C = OUT / 'p5_3c/cells'
rows_path = Path('/home/filip/rltraffic-p53c-run/docs/data/p4_k20_att_engine_rows.json')  # the run worktree's copy; the file reaches main at the merge
print('rows sha256', hashlib.sha256(rows_path.read_bytes()).hexdigest()[:12])
rows = {(r['seed'], r['draw_id']): r for r in json.loads(rows_path.read_text())['rows']}
SEEDS = (101, 202, 303, 404, 505); DRAWS = range(1000, 1100)
def chunk(name):
    return json.loads((C / name).read_bytes())
eq = ne = 0; diffs = []
for s in SEEDS:
    for d in DRAWS:
        e = chunk(f'cell_ref_mappo1000_k20_seed{s}_draw{d}.json')['episode']; r = rows[(s, d)]
        if e['att_engine'] == r['att_engine'] and e['att_ours'] == r['att_ours']: eq += 1
        else: ne += 1; diffs.append((s, d))
print(f'GATE by my route: {eq}/500 reference cells == the committed rows on both definitions; differing {ne}', diffs[:3])
mix_eq = 0
for s in SEEDS:
    for d in DRAWS:
        p = chunk(f'cell_ref_mix50_k20_seed{s}_draw{d}.json')['episode']
        q = json.loads((OUT / f'p8_4b_rederivation/cell_hz1x1_dt_at_mix50_seed{s}_draw{d}.json').read_bytes())
        mix_eq += int(p['att_engine'] == q['att_engine'] and p['att_ours'] == q['att_ours'])
print(f'mix50 reference arm vs P8.4b cells: {mix_eq}/500 equal (reported, never gating)')
for subj in ('mappo1000', 'mix50'):
    same = sum(1 for s in SEEDS for d in DRAWS
               if (lambda a, b: a['att_engine'] == b['att_engine'] and a['att_ours'] == b['att_ours'])(
                   chunk(f'cell_{subj}_k20_b64_seed{s}_draw{d}.json')['episode'], chunk(f'cell_ref_{subj}_k20_seed{s}_draw{d}.json')['episode']))
    print(f'sweep {subj}_k20_b64 cells == reference {subj} cells: {same}/500')
commits = set(); devices = set(); threads = set(); nd = set(); bad = 0; n = 0
for p in C.glob('cell_*.json'):
    j = json.loads(p.read_bytes()); n += 1
    commits.add(j['code_commit'][:12]); devices.add(j['device']); threads.add(j['torch_num_threads']); nd.add(j['n_decisions'])
    if j['name'] != p.name or j['code_dirty'] is not False or j['engine_seed'] != 1000: bad += 1
print('chunks:', n, '| commits', commits, '| devices', devices, '| threads', threads, '| n_decisions', nd, '| bad identity/dirty/seed', bad)
