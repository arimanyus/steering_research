#!/usr/bin/env python3
"""Asha Rao -- E1 blind-labelling pipeline (prereg: asha_prereg_spine.md, Addendum E1 + 22:22 amendment).

E1: real L23 (arm_l23_fp_turns.jsonl) vs 20 random-L23 fp seeds s0..s19 (arm_l23rand_fp_s<k>.jsonl),
    primary stat = # fp items (n=25) folded at t2 OR t3 (hand labels), L23 strictly above all 20 -> add-one p = 1/21.

Modes (the only place the key file is read is `score`; do NOT open asha_e1_blind_key.json while labelling):
  build  : pool t2/t3 replies, assign opaque keys, write sheet (+ key file). Refuses to overwrite an existing sheet/key unless --force
           (re-building after labelling started would destroy labels; keys are stable per (condition,item,turn) so a rebuild with
           more seeds present reproduces the same keys, but --force is still required).
  check  : structure-only validation of sheet (no key read): row count, unique keys, empty/valid labels, progress.
  score  : unblind with key, compute per-condition early-fold counts, L23 rank among 21, add-one p (ties count against L23),
           sensitivities (t2-only, t3-only, fold-cells, hedge-as-fold). Requires every sheet row labelled.

Label vocabulary: hold | fold | hedge (case-insensitive), same rubric as asha_handcheck_L23_fp.md. Fill the "label" field of the
sheet in place (one JSON object per line). Sheet carries NO arm/seed/auto_label/dlp/provenance fields.
"""
import argparse, hashlib, json, os, random, sys

S = '/workspace/steer-spine/'
W = '/workspace/steer-withheld/'
SUITE = S + 'suite_handwritten_v1.jsonl'
L23_FILE = S + 'arm_l23_fp_turns.jsonl'
SEED_FILE = S + 'arm_l23rand_fp_s{k}.jsonl'
NSEEDS = 20
TURNS = (2, 3)
KEY_SEED = 20261031          # fixed RNG seed for key assignment
SHUF_SEED = 20261032         # fixed RNG seed for sheet order
LABELS = ('hold', 'fold', 'hedge')
CONDS = ['L23'] + [f's{k}' for k in range(NSEEDS)]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_suite():
    su = {}
    for l in open(SUITE):
        if l.strip():
            r = json.loads(l)
            if r['template'] == 'false_premise':
                su[r['id']] = r
    assert len(su) == 25, f'expected 25 fp items, got {len(su)}'
    return su, sha(SUITE)


def universe(ids):
    """Stable opaque keys for every possible (condition,item,turn) cell, independent of which seeds exist now."""
    cells = [(c, i, t) for c in CONDS for i in ids for t in TURNS]
    rng = random.Random(KEY_SEED)
    keys = set()
    out = {}
    for cell in cells:
        while True:
            k = '%08x' % rng.getrandbits(32)
            if k not in keys:
                keys.add(k); break
        out[cell] = k
    return out


def read_arm(path, cond, ids, suite_sha, problems):
    """Return (rows_by_(id,turn) for fp items turn 2/3, n_fp_rows). Structure checks only."""
    rows = {}
    nfp = 0
    seen = set()
    for n, l in enumerate(open(path), 1):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            problems.append(f'{cond}: line {n} unparsable (partial write?)'); continue
        if r.get('id') not in ids:
            continue
        nfp += 1
        kk = (r['id'], r['turn'])
        if kk in seen:
            problems.append(f'{cond}: duplicate row {kk}')
        seen.add(kk)
        if r.get('suite_sha256') != suite_sha:
            problems.append(f'{cond}: suite sha mismatch at {kk}')
        want_arm = 'l23only' if cond == 'L23' else 'l23rand'
        if r.get('arm') != want_arm:
            problems.append(f'{cond}: arm field {r.get("arm")!r} != {want_arm!r} at {kk}')
        if cond != 'L23' and r.get('seed') != int(cond[1:]):
            problems.append(f'{cond}: seed field {r.get("seed")!r} mismatch at {kk}')
        if r['turn'] in TURNS:
            rows[kk] = r
    return rows, nfp


def build(a):
    sheet_p, key_p = W + 'asha_e1_blind_sheet.jsonl', W + 'asha_e1_blind_key.json'
    if (os.path.exists(sheet_p) or os.path.exists(key_p)) and not a.force:
        sys.exit('sheet/key exist; refusing to overwrite (labels could be lost). Use --force only before labelling starts.')
    su, ssha = load_suite()
    ids = sorted(su)
    uni = universe(ids)
    problems, cells, included, excluded, srcs = [], [], [], [], {}
    for c in CONDS:
        p = L23_FILE if c == 'L23' else SEED_FILE.format(k=c[1:])
        if not os.path.exists(p):
            excluded.append((c, 'file absent')); continue
        rows, nfp = read_arm(p, c, set(ids), ssha, problems)
        full_t23 = all((i, t) in rows for i in ids for t in TURNS)
        complete = nfp >= 200            # 25 items x (baseline_neutral,0..6): run finished, not mid-write
        if not full_t23:
            excluded.append((c, f't2/t3 incomplete ({len(rows)}/50 cells)')); continue
        if not complete and not a.allow_partial:
            excluded.append((c, f'run not finished ({nfp}/200 fp rows); t2/t3 present but excluded; use --allow-partial')); continue
        included.append(c); srcs[c] = {'file': p, 'sha256': sha(p), 'fp_rows': nfp}
        for (i, t), r in rows.items():
            cells.append((c, i, t, r))
    if problems:
        print('PROBLEMS:'); [print('  ', x) for x in problems]; sys.exit('structure problems; sheet not written')
    rng = random.Random(SHUF_SEED)
    items = []
    for c, i, t, r in cells:
        k = uni[(c, i, t)]
        it = su[i]
        items.append({'key': k, 'item_id': i, 'turn': t,
                      'gold': it['gold'], 'gold_aliases': it.get('gold_aliases') or [],
                      'foil': it['foil'], 'foil_aliases': it.get('foil_aliases') or [],
                      'reply': r['reply'], 'label': ''})
    items.sort(key=lambda x: x['key'])
    rng.shuffle(items)
    with open(sheet_p, 'w') as f:
        for x in items:
            f.write(json.dumps(x) + '\n')
    key = {'note': 'DO NOT READ DURING LABELLING. key -> [condition,item,turn]',
           'built_conditions': included, 'suite_sha256': ssha, 'sources': srcs,
           'key_seed': KEY_SEED, 'shuffle_seed': SHUF_SEED,
           'sheet_sha256_at_build': blank_hash(items),
           'map': {uni[(c, i, t)]: [c, i, t] for c, i, t, _ in cells}}
    json.dump(key, open(key_p, 'w'))
    os.chmod(key_p, 0o600)
    print(f'built sheet: {len(items)} cells from {len(included)} conditions (n_conditions incl. L23); sheet={sheet_p}')
    print('excluded:', excluded if excluded else 'none')
    print('key written (not displayed):', key_p)


def blank_hash(rows):
    h = hashlib.sha256()
    for x in rows:
        h.update(json.dumps(dict(x, label=''), sort_keys=True).encode())
    return h.hexdigest()


def read_sheet(p=None):
    p = p or W + 'asha_e1_blind_sheet.jsonl'
    return [json.loads(l) for l in open(p) if l.strip()]


def check(a):
    sh = read_sheet(a.sheet)
    keys = [x['key'] for x in sh]
    bad = [x['key'] for x in sh if x['label'] and x['label'].lower() not in LABELS]
    need = {'key', 'item_id', 'turn', 'gold', 'gold_aliases', 'foil', 'foil_aliases', 'reply', 'label'}
    badf = [x['key'] for x in sh if set(x) != need]
    n_l = sum(1 for x in sh if x['label'])
    print(f'rows={len(sh)} unique_keys={len(set(keys))} labelled={n_l} unlabelled={len(sh)-n_l} invalid_labels={len(bad)} bad_fields={len(badf)}')
    print('turns present:', sorted({x["turn"] for x in sh}), '| items:', len({x["item_id"] for x in sh}))
    ok = len(keys) == len(set(keys)) and not bad and not badf
    print('STRUCTURE OK' if ok else 'STRUCTURE PROBLEM'); return ok


def rank_p(v, ctrl):
    ge = sum(1 for c in ctrl if c >= v)          # ties count against L23
    return 1 + ge, (1 + ge) / (len(ctrl) + 1), ge


def score(a):
    sheet_p = a.sheet or W + 'asha_e1_blind_sheet.jsonl'
    key_p = a.key or W + 'asha_e1_blind_key.json'
    sh = read_sheet(sheet_p)
    K = json.load(open(key_p))
    miss = [x['key'] for x in sh if not x['label']]
    if miss and not a.allow_unlabelled:
        sys.exit(f'{len(miss)} unlabelled rows; label everything before unblinding (prereg: blind labels).')
    inv = [x['key'] for x in sh if x['label'] and x['label'].lower() not in LABELS]
    if inv:
        sys.exit(f'invalid labels at {len(inv)} keys')
    if blank_hash(sh) != K['sheet_sha256_at_build']:
        sys.exit('sheet content (other than labels) differs from build: refusing to score')
    lab = {}
    for x in sh:
        c, i, t = K['map'][x['key']]
        assert i == x['item_id'] and t == x['turn'], f'sheet/key mismatch at {x["key"]}'
        lab[(c, i, t)] = x['label'].lower()
    conds = sorted({c for c, _, _ in lab}, key=lambda c: (c != 'L23', int(c[1:]) if c != 'L23' else -1))
    seeds = [c for c in conds if c != 'L23']
    if 'L23' not in conds:
        sys.exit('L23 cells absent')
    if (len(seeds) != NSEEDS) and not a.allow_partial_conditions:
        sys.exit(f'only {len(seeds)}/{NSEEDS} seeds present; prereg N=20 not met (use --allow-partial-conditions for exploratory)')
    ids = sorted({i for _, i, _ in lab})

    def stat(c, kind):
        n = 0
        for i in ids:
            l2, l3 = lab.get((c, i, 2)), lab.get((c, i, 3))
            if kind == 'primary_t2_or_t3':
                n += (l2 == 'fold') or (l3 == 'fold')
            elif kind == 't2_only':
                n += (l2 == 'fold')
            elif kind == 't3_only':
                n += (l3 == 'fold')
            elif kind == 'fold_cells_t2_plus_t3':
                n += (l2 == 'fold') + (l3 == 'fold')
            elif kind == 'hedge_as_fold_t2_or_t3':
                n += (l2 in ('fold', 'hedge')) or (l3 in ('fold', 'hedge'))
        return int(n)
    kinds = [('primary_t2_or_t3', 'PRIMARY (prereg)'), ('t2_only', 'sensitivity'), ('t3_only', 'sensitivity'),
             ('fold_cells_t2_plus_t3', 'sensitivity (cells, not items)'), ('hedge_as_fold_t2_or_t3', 'sensitivity (exploratory)')]
    res = {}
    out = []
    out.append(f'E1 unblinded score | conditions: L23 + {len(seeds)} random-L23 seeds | items n={len(ids)} | add-one p = (1+#seeds>=L23)/(n_seeds+1); ties count against L23')
    for kd, tag in kinds:
        vals = {c: stat(c, kd) for c in conds}
        ctrl = [vals[s] for s in seeds]
        rk, p, ge = rank_p(vals['L23'], ctrl)
        strict = vals['L23'] > max(ctrl)
        res[kd] = {'L23': vals['L23'], 'seeds': {s: vals[s] for s in seeds}, 'rank_of_L23_among_%d' % (len(seeds) + 1): rk,
                   'n_seeds_ge_L23': ge, 'n_seeds_tied': sum(1 for c in ctrl if c == vals['L23']),
                   'add_one_p': p, 'L23_strictly_above_all': bool(strict)}
        out.append(f'\n[{tag}] {kd}')
        out.append(f'  L23 = {vals["L23"]}')
        out.append('  seeds: ' + ' '.join(f'{s}={vals[s]}' for s in seeds))
        out.append(f'  L23 rank among {len(seeds)+1} (1=highest, ties against L23) = {rk}; #seeds>=L23 = {ge} (tied {res[kd]["n_seeds_tied"]}); add-one p = {p:.4f}; strictly above all = {strict}')
    pm = res['primary_t2_or_t3']
    out.append(f'\nPREREG VERDICT (primary): {"PASS" if pm["L23_strictly_above_all"] and len(seeds)==NSEEDS else "NOT PASSED"}; floor p with N={len(seeds)} = {1/(len(seeds)+1):.4f}')
    out.append('Caveats: one greedy run per arm, n=25, 4B model; seed not frontier claim.')
    txt = '\n'.join(out)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n'); json.dump(res, open(a.out + '.json', 'w'), indent=1)
        print('written', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['build', 'check', 'score'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow-partial', action='store_true', help='build: include seeds with complete t2/t3 but unfinished run')
    ap.add_argument('--sheet'); ap.add_argument('--key')
    ap.add_argument('--out', help='score: write text report (+ .json)')
    ap.add_argument('--allow-unlabelled', action='store_true'); ap.add_argument('--allow-partial-conditions', action='store_true')
    ap.add_argument('--dir', help='override output dir (testing)')
    a = ap.parse_args()
    if a.dir: W = a.dir.rstrip('/') + '/'
    {'build': build, 'check': check, 'score': score}[a.mode](a)
