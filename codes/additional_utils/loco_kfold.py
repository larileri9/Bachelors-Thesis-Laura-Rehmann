


from pathlib import Path

import yaml

import train_benchmarking_utils as utils
from loco_kfold_splits import generate_species_folds

"""

Setup helpers for loco runs and everything lives under chrombpnet_data_kfold:

kfold_dir/loco_folds/{species_short}/fold_0.json 
kfold_dir/loco_configs/{species_short}/fold_0.yaml   

Exceptions per specie:

s_cerevisiae --> length matched into 8 folds
a_niger --> collapse I_1/I_2 scaffolds, then one chromosome per fold
everyone else --> one chromosome per fold

Only roman numeral chromosomes are used --> chrR, Mito and scaffolds in chrom_sizes.txt are dropped 
"""


# in case wed ever add sth have it fixed ehre or we conduct different fold tests like more or less folds 
size_matched_k_fixed = {"s_cerevisiae": 8}
group_by_prefix_fixed = {"a_niger"}


# path construction
def construct_loco_folds_dir(chrombpnet_data):
    return Path(chrombpnet_data) / "loco_folds"


def construct_loco_fold_path(chrombpnet_data, species, fold):
    short = utils.construct_species_short(species)
    return construct_loco_folds_dir(chrombpnet_data) / short / f"fold_{fold}.json"


def construct_loco_configs_dir(chrombpnet_data):
    return Path(chrombpnet_data) / "loco_configs"


def construct_loco_config_path(chrombpnet_data, species, fold):
    short = utils.construct_species_short(species)
    return construct_loco_configs_dir(chrombpnet_data) / short / f"fold_{fold}.yaml"


# here see if folds are there or not otherwise generate them
def ensure_folds(chrombpnet_data, species, chrom_sizes):
    short = utils.construct_species_short(species)
    species_dir = construct_loco_folds_dir(chrombpnet_data) / short

    existing = []
    if species_dir.is_dir():
        existing = sorted(species_dir.glob("fold_*.json"))

    #early exit here if exists
    if existing:
        print(f"[folds] {short}: found folds")
        return existing

    k = None
    if short in size_matched_k_fixed:
        k = size_matched_k_fixed[short]

    group_by_prefix = False
    if short in group_by_prefix_fixed:
        group_by_prefix = True

    if k is None:
        print(f"[folds] {short}: generating one chrom per fold")
    else:
        print(f"[folds] {short}: generating length matched into {k} folds")

    paths, sizes = generate_species_folds(
        chrom_sizes=str(chrom_sizes),
        species_short=short,
        output_dir=str(construct_loco_folds_dir(chrombpnet_data)),
        group_by_prefix=group_by_prefix,
        size_matched_k=k,
    )
    print(f"[folds] {short}: wrote {len(paths)} folds")
    return paths






# and here see if configs exist --> template yaml is the holdout singular yaml 
def ensure_config(chrombpnet_data, template_yaml, species, fold):
    target = construct_loco_config_path(chrombpnet_data, species, fold)
    if target.exists():
        return target

    # load the template with all six species in it
    cfg = yaml.safe_load(Path(template_yaml).read_text())

     # take out the block for this species only, as a copy so we dont edit the template
    block = cfg["species"][species].copy()   # species is the full name here

    # point it at this fold instead of whatever the template said
    block["folds"] = str(construct_loco_fold_path(chrombpnet_data, species, fold))
    
    # make the new config and write
    new_cfg = {"species": {species: block}}
    if "adapter_seq" in cfg:
        new_cfg["adapter_seq"] = cfg["adapter_seq"]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(new_cfg, sort_keys=False))
    return target


