
import os

folds_dir = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data_kfold/loco_folds/s_cerevisiae/"
folds = glob_wildcards(os.path.join(folds_dir, "fold_{fold}.json")).fold

folds = [0,1,2,3,4,5,6,7]


root_dir = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data_kfold/s_cerevisiae"
input_root = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data/s_cerevisiae" 
out_dir = "params_2"


BLACKLIST = "/s/project/multispecies/fungi_code/atac/debug/schep/data/sacCer_blacklist.bed"
CHROM_SIZES = "/s/project/multispecies/fungi_code/atac/debug/schep/data/saccCer.chrom.sizes"
FASTA = "/s/project/multispecies/fungi_code/atac/debug/schep/data/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"

#### CONFIGS

# Default
INPUT_LEN = 1216        # Default 2114
OUTPUT_LEN = 608        # Default 1000
MAX_JITTER = 50        # Default 250
DIL_LAYERS = 7          # Default 8 (depends on output len: 20 + sum(2**(i+1) for i=1..DIL_LAYERS) + 74 + OUTPUT_LEN)
N_FILTERS = 64         # Default 512
LEARNING_RATE = 0.001   # Default 0.001

# INPUT_LEN = 1052        # Default 2114
# OUTPUT_LEN = 500        # Default 1000
# MAX_JITTER = 250        # Default 250
# DIL_LAYERS = 6          # Default 8 (depends on output len: 20 + sum(2**(i+1) for i=1..DIL_LAYERS) + 74 + OUTPUT_LEN)
# N_FILTERS = 512         # Default 512
# LEARNING_RATE = 0.001   # Default 0.001


bias_name = f"test_no_shift_flag_il_{INPUT_LEN}_ol_{OUTPUT_LEN}"
model_name = f"debug_il_{INPUT_LEN}_ol_{OUTPUT_LEN}_nf_{N_FILTERS}_dl_{DIL_LAYERS}"

print()
print("--- root_dir", root_dir)
print("--- bias name:", bias_name)
print("--- model name:", model_name)
print()

if not os.path.exists(root_dir): 
    os.makedirs(root_dir, exist_ok=True)


PEAKS = os.path.join(
    input_root, 
    "results", 
    "06_peaks", 
    "all_merged_relaxed_peaks_no_blacklist.bed"
)
BAM = os.path.join(
    input_root, 
    "results", 
    "05_merged",
    "all_merged.sorted.shifted.bam"
)

print(PEAKS)
print(BAM)

rule all:
    input: 
        expand(
            os.path.join(
                root_dir, 
                out_dir,
                "models", 
                "data", 
                "fold_{fold}",
                "fold_negatives.bed" 
            ), 
            fold=folds
        ),
        expand(
            os.path.join(
                root_dir, 
                out_dir,
                "models", 
                "bias", 
                bias_name,
                "fold_{fold}",
                "evaluation",
                "overall_report.pdf"
            ), 
            fold=folds
        ), 
        expand(
            os.path.join(
                root_dir, 
                out_dir,
                "models", 
                "chrombpnet", 
                model_name,
                "fold_{fold}",
                "evaluation", 
                "overall_report.pdf"
            ), 
            fold=folds
        )

rule non_peak_regions:
    input: 
        fasta = FASTA, 
        chrom_sizes = CHROM_SIZES,
        blacklist = BLACKLIST,
        peaks = PEAKS,
        # folds_dir = folds_dir,
        fold_json = os.path.join(folds_dir, "fold_{fold}.json")

    output: 
        fold_json = os.path.join(
            root_dir, 
            out_dir, 
            "folds", 
            "fold_{fold}.json"
        ),
        
        path=os.path.join(
            root_dir, 
            out_dir,
            "models", 
            "data", 
            "fold_{fold}",
            "fold_negatives.bed" 
        ),
        save_dir = directory(
            os.path.join(
                root_dir, 
                out_dir,
                "models", 
                "data", 
                "fold_{fold}"
            )
        )
    params: 
        prefix = lambda wc: os.path.join(
            root_dir, 
            out_dir, 
            "models", 
            "data", 
            f"fold_{wc.fold}", 
            "fold"
        ), 
        input_len=INPUT_LEN, 
        stride=500 
    conda: "pjo_chrombpnet"
    threads: 16
    resources: 
        mem_mb=64000

    shell: 
        """
        rm -f -r {output.save_dir}; \
        echo "FOLDS: {root_dir}/{out_dir}/folds"; \
        echo "{input.fold_json} > {output.fold_json}"; \
        cp {input.fold_json} {output.fold_json}; \
        chrombpnet prep nonpeaks \
            -g {input.fasta} \
            -p {input.peaks} \
            -c {input.chrom_sizes} \
            -fl {output.fold_json} \
            -br {input.blacklist} \
            -o {params.prefix} \
            -st 250 \
            -il {params.input_len}
        """


rule train_bias:
    input:
        bam = BAM,
        fasta = FASTA, 
        chrom_sizes = CHROM_SIZES,
        peaks = PEAKS, 
        negatives = rules.non_peak_regions.output.path, 
        fold_json = os.path.join(folds_dir, "fold_{fold}.json")

    output:
        report=os.path.join(
            root_dir, 
            out_dir,
            "models", 
            "bias", 
            bias_name,
            "fold_{fold}",
            "evaluation", 
            "overall_report.pdf"
        ),
        model=os.path.join(
            root_dir, 
            out_dir,
            "models", 
            "bias", 
            bias_name,
            "fold_{fold}",
            "models", 
            "bias.h5"
        ),
        save_dir = directory(
            os.path.join(
                root_dir, 
                out_dir,
                "models",
                "bias", 
                bias_name,
                "fold_{fold}"
            )
        )
    conda: "pjo_chrombpnet"
    threads: 16
    resources:
        mem_mb=64000, 
        gpu=1
    params: 
        input_len=INPUT_LEN, 
        output_len=OUTPUT_LEN
    shell:
        """
        rm -f -r {output.save_dir}; \
        chrombpnet bias pipeline \
            -ibam {input.bam} \
            -d "ATAC" \
            -g {input.fasta} \
            -c {input.chrom_sizes} \
            -p {input.peaks} \
            -n {input.negatives} \
            -fl {input.fold_json} \
            -b 0.4 \
            -o {output.save_dir} \
            -il {params.input_len}  \
            -ol {params.output_len} \
            -ps 4 \
            -ms -5
        """


rule train_model:
    input:
        bam = BAM,
        fasta = FASTA, 
        chrom_sizes = CHROM_SIZES,
        peaks = PEAKS, 
        negatives = rules.non_peak_regions.output.path, 
        fold_json = os.path.join(folds_dir, "fold_{fold}.json"),
        bias_model = rules.train_bias.output.model
        
    output:
        report = os.path.join(
            root_dir, 
            out_dir,
            "models", 
            "chrombpnet", 
            model_name,
            "fold_{fold}",
            "evaluation", 
            "overall_report.pdf"
        ),

        model=os.path.join(
            root_dir, 
            out_dir,
            "models", 
            "chrombpnet", 
            model_name,
            "fold_{fold}",
            "models", 
            "chrombpnet_nobias.h5"
        ),
        save_dir = directory(
            os.path.join(
                root_dir,
                out_dir,
                "models",
                "chrombpnet", 
                model_name,
                "fold_{fold}"
            )
        )
    conda: "pjo_chrombpnet"
    resources: 
        mem_mb=64000,
        gpu=1
    threads: 16
    params: 
        libpath="/opt/modules/i12g/anaconda/envs/pjo_chrombpnet/lib::/usr/local/cuda/lib64:/usr/local/cuda/lib64",
        n_filters=N_FILTERS, # default 512
        dil_layers=DIL_LAYERS,  # default 8
        input_len=INPUT_LEN, # default 2114
        output_len=OUTPUT_LEN, # default 1000
        max_jitter=MAX_JITTER,  # default 500, 
        learning_rate=LEARNING_RATE # default 0.001

    shell:
        """
        rm -f -r {output.save_dir}; \
        LD_LIBRARY_PATH={params.libpath}; \
        chrombpnet pipeline \
            -ibam {input.bam} \
            -d "ATAC" \
            -g {input.fasta} \
            -c {input.chrom_sizes} \
            -p {input.peaks} \
            -n {input.negatives} \
            -fl {input.fold_json} \
            -b {input.bias_model} \
            -o {output.save_dir} \
            -fil {params.n_filters} \
            -dil {params.dil_layers} \
            -j {params.max_jitter} \
            -il {params.input_len} \
            -ol {params.output_len} \
            -ps 4 \
            -ms -5
        """
