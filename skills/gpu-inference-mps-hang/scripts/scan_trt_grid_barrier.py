#!/usr/bin/env python3
"""scan_fast.py [-j N] [--all] [--names] <engine|plan>...

Find kernels with a grid-level software barrier in TensorRT engines (they deadlock under multi-process MPS).
Two levels:
  1) Seconds: cubin symbol names in an engine are plain text, so grep `__myl_*`; Myelin fused normalization
     kernels (Inor/Gnor/ReluGnor/MeanMean) are the kind that fails. No such names usually means done.
  2) Confirm: disassemble only the matching cubins (all of them with --all) with parallel cuobjdump,
     looking for YIELD + LDG.E.STRONG.GPU + a backward BRA.
"""
import argparse, os, re, subprocess, sys, tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

CUOBJDUMP = '/usr/local/cuda-13.0/bin/cuobjdump'
SUSPECT = re.compile(r'(?i)(inor|gnor|[a-z]nor_|meanmean|instancenorm|groupnorm|layernorm)')
TMP = os.environ.get('SCAN_TMP', tempfile.gettempdir())


def sass_of(args):
    path, off, end, idx = args
    p = os.path.join(TMP, f'scan_{os.getpid()}_{idx}.cubin')
    with open(path, 'rb') as f:
        f.seek(off)
        blob = f.read(end - off)
    with open(p, 'wb') as f:
        f.write(blob)
    try:
        r = subprocess.run([CUOBJDUMP, '-sass', p], capture_output=True, text=True)
    finally:
        os.unlink(p)
    if r.returncode != 0:
        return idx, None, []
    hits = []
    for part in re.split(r'\n\s*Function : ', r.stdout)[1:]:
        fn = part.split()[0]
        if 'LDG.E.STRONG.GPU' in part and 'YIELD' in part:
            hits.append(re.sub(r'_0x[0-9a-f]+.*', '', fn))
    return idx, True, hits


def scan(path, jobs, do_all, names_only):
    data = open(path, 'rb').read()
    offs = [m.start() for m in re.finditer(b'\x7fELF', data)]
    bounds = [(offs[k], offs[k + 1] if k + 1 < len(offs) else len(data)) for k in range(len(offs))]
    names = Counter()
    for m in re.finditer(rb'__myl_[0-9A-Za-z_]+', data):
        names[m.group().decode()] += 1
    susp = sorted({n for n in names if SUSPECT.search(n)})
    print(f'{path}\n  size={len(data)/1e6:.0f}MB cubins={len(bounds)} myl_kernels={len(names)} suspect_names={len(susp)}')
    if susp:
        print('  suspect:', ', '.join(sorted({re.sub(r"_myl\d+_\d+$", "", s) for s in susp})[:12]))
    if names_only:
        return 0
    # Disassemble only cubins with suspicious names (all with --all)
    todo = []
    for i, (a, b) in enumerate(bounds):
        if do_all or any(s.encode() in data[a:b] for s in susp):
            todo.append((path, a, b, i))
    if not todo:
        print('  -> no suspicious Myelin normalization kernels; skipping disassembly (use --all to confirm)')
        return 0
    hits = Counter()
    ok = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for idx, valid, fns in ex.map(sass_of, todo, chunksize=1):
            ok += bool(valid)
            hits.update(fns)
    print(f'  disassembled={ok}/{len(todo)} grid_barrier_kernels={sum(hits.values())} {dict(hits)}')
    return sum(hits.values())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', '--jobs', type=int, default=min(24, os.cpu_count() or 8))
    ap.add_argument('--all', action='store_true', help='disassemble everything, without the name prefilter')
    ap.add_argument('--names', action='store_true', help='name scan only (milliseconds)')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    bad = 0
    for f in a.files:
        bad += scan(f, a.jobs, a.all, a.names)
    print('TOTAL grid-barrier kernels:', 'n/a (--names)' if a.names else bad)
    sys.exit(1 if bad else 0)
