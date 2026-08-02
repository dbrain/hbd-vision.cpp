import struct, sys, os, glob

def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

def skip_kv_val(f, t):
    prim = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    if t in prim: rd(f, prim[t]); return
    if t == 8: rstr(f); return
    if t == 9:
        (et,) = rd(f, "<I"); (n,) = rd(f, "<Q")
        for _ in range(n): skip_kv_val(f, et)
        return
    raise ValueError(f"kv type {t}")

hits, scanned, errs = [], 0, []
roots = sys.argv[1:]
files = []
for r in roots:
    files += glob.glob(os.path.join(r, "**", "*.gguf"), recursive=True)
for p in sorted(set(files)):
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF": continue
            (ver,) = rd(f, "<I")
            (n_tensors,) = rd(f, "<Q"); (n_kv,) = rd(f, "<Q")
            for _ in range(n_kv):
                rstr(f); (t,) = rd(f, "<I"); skip_kv_val(f, t)
            bad = []
            for _ in range(n_tensors):
                name = rstr(f); (nd,) = rd(f, "<I")
                for _ in range(nd): rd(f, "<Q")
                (tt,) = rd(f, "<I"); rd(f, "<Q")
                if tt == 42: bad.append(name)
            scanned += 1
            if bad: hits.append((p, len(bad), bad[:3]))
    except Exception as e:
        errs.append((p, str(e)[:70]))

print(f"scanned {scanned} gguf files")
if hits:
    print("\n!!! TYPE ID 42 FOUND (fork F8_E4M3 -> now reads as upstream Q2_0):")
    for p, n, names in hits:
        print(f"  {p}\n    {n} tensors, e.g. {names}")
else:
    print("no type-id-42 tensors found")
if errs:
    print(f"\nunparsed ({len(errs)}):")
    for p, e in errs[:10]: print(f"  {os.path.basename(p)}: {e}")
