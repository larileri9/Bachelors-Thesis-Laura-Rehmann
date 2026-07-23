#!/usr/bin/env python3


import argparse
import json
from collections import defaultdict
from pathlib import Path


# this does loco (can be length matched)
# real chromosomes are roman numerals  Everything else (chrR, Mito,  MT, NW_* / scaffold_*)  --> drop
#a_niger names like 'I_1' --> we test the part before the first '_' --> --group-by-prefix / collapse into one chrom to rotate
# hard code these to check will need to be updated for bigger orgs
ROMAN_1_20 = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
}


def is_roman_chrom(name):
    return name.split("_")[0] in ROMAN_1_20


#chrom_sizes.txt is two columns and we return them as chrom:length
def read_chrom_sizes(path):
    sizes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            sizes[parts[0]] = int(parts[1])
    return sizes



def group_units(sizes, group_by_prefix):
    # which real chrom names belong to which unit
    members = {}
    for name in sizes:
        if group_by_prefix:
            unit = name.split("_")[0]
        else:
            unit = name

        if unit not in members:
            members[unit] = []
        members[unit].append(name)

    # total bp per unit --> needed for mathcing 
    unit_size = {}
    for unit in members:
        total = 0
        for name in members[unit]:
            total = total + sizes[name]
        unit_size[unit] = total

    return members, unit_size




#greedy algorithm for length sroting
def partition_size_matched(unit_size, k):
    # biggest chrom first
    units_sorted = sorted(unit_size, key=unit_size.get, reverse=True)

    groups, totals = [],[]
    for i in range(k):
        groups.append([])
        totals.append(0)

    for unit in units_sorted:
        #find bin with least bp so far 
        target = min(range(k), key=lambda i: totals[i])

        groups[target].append(unit)
        totals[target] = totals[target] + unit_size[unit]

    return groups



# makes a list of lists of units 
def one_per_group(unit_size):
    return [[u] for u in sorted(unit_size, key=unit_size.get, reverse=True)]
# trun I back to I_1 and I_2
def expand(units, members):
    out = []
    for u in units:
        out.extend(members[u])
    return sorted(out)

# build folds with each group and rotate through how many we need
def build_folds(groups, members):
    g = len(groups)
    folds = []
    for i in range(g):
        valid_idx = (i + 1) % g
        train_units = []
        for j, grp in enumerate(groups):
            if j == i or j == valid_idx:
                continue
            train_units.extend(grp)
        folds.append({
            "test": expand(groups[i], members),
            "valid": expand(groups[valid_idx], members),
            "train": expand(train_units, members),
        })
    return folds

# sanity check --> assert if two partitions in a fold are not disjoint
def sanity_check(folds, all_chroms):
    full = set(all_chroms)
    for i, fold in enumerate(folds):
        t, v, tr = set(fold["test"]), set(fold["valid"]), set(fold["train"])
        assert t.isdisjoint(v), f"fold {i}: test and valid overlap"
        assert t.isdisjoint(tr), f"fold {i}: test and train overlap"
        assert v.isdisjoint(tr), f"fold {i}: valid and train overlap"
        assert len(t) >= 1 and len(tr) >= 1, f"fold {i}: empty test or train"


def generate_species_folds(chrom_sizes, species_short, output_dir, group_by_prefix=False, size_matched_k=None):
    all_sizes = read_chrom_sizes(chrom_sizes)
    sizes = {}
    # get anmes and sizes
    for name in all_sizes:
        if is_roman_chrom(name):
            sizes[name] = all_sizes[name]
    # group and make list
    members, unit_size = group_units(sizes, group_by_prefix)
    all_chroms = list(sizes.keys())
   
    if size_matched_k:
        # need at least 3 --> train test val need to rotate once then get unit size and reduce to size matched k for this fold if specified
        k = max(3, min(size_matched_k, len(unit_size)))
        groups = partition_size_matched(unit_size, k)
    else:
        groups = one_per_group(unit_size)

    folds = build_folds(groups, members)
    sanity_check(folds, all_chroms)

    out_dir = Path(output_dir) / species_short
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fold in enumerate(folds):
        path = out_dir / f"fold_{i}.json"
        path.write_text(json.dumps(fold, indent=2))
        paths.append(path)
    return paths, sizes


def parse_args():
    p = argparse.ArgumentParser(description="Generate loco folds for one species")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--species-short", required=True, help="e.g. s_cerevisiae")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--group-by-prefix", action="store_true",
                   help="collapse 'I_1','I_2' scaffolds into one chromosome (mostly for a_niger)")
    p.add_argument("--size-matched-k", type=int, default=None,
                   help="length match into this many folds (s_cerevisiae=8)")
    return p.parse_args()


def main():
    args = parse_args()
    paths, sizes = generate_species_folds(
        chrom_sizes=args.chrom_sizes,
        species_short=args.species_short,
        output_dir=args.output_dir,
        group_by_prefix=args.group_by_prefix,
        size_matched_k=args.size_matched_k,
    )
    for path in paths:
        fold = json.loads(path.read_text())
        print(f"wrote {path}")
    print(f"\ndone: {len(paths)} folds")


if __name__ == "__main__":
    main()