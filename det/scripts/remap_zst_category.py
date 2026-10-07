#!/usr/bin/env python
"""Renumber the categoryId of a detection .zst in place, losslessly.

The viewer identifies a class by **both** `categoryId` (a small number) and `termId` (the platform's
stable id for the annotation term). Renumbering means moving the *number* while each class keeps its
own termId and `nt` flag -- so per feature only the categoryId value changes and the termId does not
move at all; what changes is which termId sits at a given number.

This edits the Mapbox-Vector-Tile protobuf directly, rewriting only the value index that the
`categoryId` tag of each feature points at. Geometry bytes, the extent, and every other property are
carried over untouched, which avoids the decode/re-encode y-axis trap entirely (`mvt.decode` returns
y-up, so a naive re-encode has to pick `y_coord_down=False` or the slide renders upside down).

  python det/scripts/remap_zst_category.py --swap 1 2 results/tils3-wsi/*.zst
  python det/scripts/remap_zst_category.py --map 1=2 --map 2=1 --dry-run <file.zst>
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys


def load_tile(path):
    import zstandard
    from mapbox_vector_tile.Mapbox import vector_tile_pb2 as pb
    raw = open(path, "rb").read()
    blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=8_000_000_000)
    tile = pb.tile()
    tile.ParseFromString(blob)
    return tile, len(raw)


def summarise(tile):
    """{(categoryId, termId, nt): count} straight from the protobuf tag tables."""
    out = collections.Counter()
    for layer in tile.layers:
        keys = list(layer.keys)
        vals = []
        for v in layer.values:
            for f in ("string_value", "bool_value", "int_value", "double_value", "float_value",
                      "uint_value", "sint_value"):
                if v.HasField(f):
                    vals.append(getattr(v, f)); break
            else:
                vals.append(None)
        for feat in layer.features:
            t = list(feat.tags)
            p = {keys[t[i]]: vals[t[i + 1]] for i in range(0, len(t), 2)}
            out[(p.get("categoryId"), p.get("termId"), p.get("nt"))] += 1
    return out


def remap(tile, mapping):
    """mapping: {old_categoryId: new_categoryId} as strings. Returns number of features changed."""
    changed = 0
    for layer in tile.layers:
        keys = list(layer.keys)
        if "categoryId" not in keys:
            continue
        cat_key = keys.index("categoryId")
        # every value index that holds a given categoryId string (duplicates are legal)
        by_str = collections.defaultdict(set)
        for i, v in enumerate(layer.values):
            if v.HasField("string_value"):
                by_str[v.string_value].add(i)
        # one canonical index per *target* string, appending the value if the file never used it
        target_idx = {}
        for new in set(mapping.values()):
            if by_str.get(new):
                target_idx[new] = min(by_str[new])
            else:
                layer.values.add().string_value = new
                target_idx[new] = len(layer.values) - 1
                by_str[new].add(target_idx[new])
        src_idx = {old: set(by_str.get(old, ())) for old in mapping}
        for feat in layer.features:
            t = list(feat.tags)
            hit = False
            for i in range(0, len(t), 2):
                if t[i] != cat_key:
                    continue
                for old, new in mapping.items():
                    if t[i + 1] in src_idx[old]:
                        t[i + 1] = target_idx[new]
                        hit = True
                        break
            if hit:
                del feat.tags[:]
                feat.tags.extend(t)
                changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--swap", nargs=2, metavar=("A", "B"), help="swap categoryId A and B")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="explicit renumbering, repeatable")
    ap.add_argument("--backup-dir", default=None, help="copy each original here before writing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--level", type=int, default=3, help="zstd compression level")
    args = ap.parse_args()

    mapping = {}
    if args.swap:
        a, b = args.swap
        mapping = {a: b, b: a}
    for m in args.map:
        old, new = m.split("=", 1)
        mapping[old] = new
    if not mapping:
        sys.exit("nothing to do: pass --swap or --map")
    # a renumbering must be a permutation, or two classes collapse onto one number
    if len(set(mapping.values())) != len(mapping):
        sys.exit(f"--map targets collide: {mapping}")
    print(f"categoryId remap: {mapping}")

    import zstandard
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    for path in args.files:
        tile, raw_size = load_tile(path)
        before = summarise(tile)
        n_feat = sum(before.values())
        changed = remap(tile, mapping)
        after = summarise(tile)

        # the renumbering must move numbers only: the (termId, nt) -> count histogram is invariant
        inv_b = collections.Counter({(t, nt): c for (_, t, nt), c in before.items()})
        inv_a = collections.Counter({(t, nt): c for (_, t, nt), c in after.items()})
        assert inv_b == inv_a, f"{path}: class identity changed\n  {inv_b}\n  {inv_a}"
        assert sum(after.values()) == n_feat, f"{path}: feature count changed"

        print(f"\n{os.path.basename(path)}  {n_feat:,} features, {changed:,} renumbered")
        for (c, t, nt), v in sorted(before.items(), key=lambda x: str(x[0][0])):
            new_c = mapping.get(c, c)
            print(f"   cat {c!r} -> {new_c!r}   nt={str(nt):5s} term={t}   x{v:,}")
        if args.dry_run:
            continue
        if args.backup_dir:
            shutil.copy2(path, os.path.join(args.backup_dir, os.path.basename(path)))
        blob = tile.SerializeToString()
        packed = zstandard.ZstdCompressor(level=args.level).compress(blob)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fp:
            fp.write(packed)
        os.replace(tmp, path)
        print(f"   written: {len(packed)/2**20:.1f} MB (was {raw_size/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
