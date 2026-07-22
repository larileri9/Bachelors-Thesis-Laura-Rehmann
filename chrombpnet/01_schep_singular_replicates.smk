import os

folds = [0,1,2,3,4,5,6,7]

#### CONFIGS
INPUT_LEN = 1216
OUTPUT_LEN = 608
MAX_JITTER = 50
DIL_LAYERS = 7
N_FILTERS = 64
LEARNING_RATE = 0.001

root_dir = "/s/project/multispecies/fungi_code/atac/data/chrom_bp_pipeline/GSE66386/control_pipeline_11/"

samples = [
    "SRR1822137",
    "GSM1621324",
    "GSM1621325",
    "GSM1621326",
    "GSM1621327",
    "GSM1621328",
    "GSM1621329",
    "GSM1621330",
    "GSM1621331",
    "GSM1621332"
]

folds_dir = "folds/"

BLACKLIST = "/s/project/multispecies/fungi_code/atac/debug/schep/data/sacCer_blacklist.bed"
CHROM_SIZES = "/s/project/multispecies/fungi_code/atac/debug/schep/data/saccCer.chrom.sizes"
FASTA = "/s/project/multispecies/fungi_code/atac/debug/schep/data/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa"

def sample_dir(sample):
    return os.path.join(root_dir, "singular_replicates", sample)

def peaks(sample):
    return os.path.join(sample_dir(sample), "results", "06_peaks", "all_merged_relaxed_peaks_no_blacklist.bed")

def bam(sample):
    return os.path.join(sample_dir(sample), "results", "05_merged", "all_merged.sorted.shifted.bam")

def out_dir(sample):
    return os.path.join(sample_dir(sample), "trained_model_new")

def bias_name(sample):
    return f"bias_il_{INPUT_LEN}_ol_{OUTPUT_LEN}"

def model_name(sample):
    return f"chrombpnet_il_{INPUT_LEN}_ol_{OUTPUT_LEN}_nf_{N_FILTERS}_dl_{DIL_LAYERS}"


rule all:
    input:
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "data", "fold_{fold}", "fold_negatives.bed"),
            sample=samples, fold=folds
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "bias", "fold_{fold}", "evaluation", "overall_report.pdf"),
            sample=samples, fold=folds
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "chrombpnet", "fold_{fold}", "evaluation", "overall_report.pdf"),
            sample=samples, fold=folds
        )


rule non_peak_regions:
    input:
        fasta       = FASTA,
        chrom_sizes = CHROM_SIZES,
        blacklist   = BLACKLIST,
        peaks       = lambda wc: peaks(wc.sample),
        fold_json   = os.path.join(folds_dir, "fold_{fold}.json")
    output:
        fold_json = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "folds", "fold_{fold}.json"),
        path      = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "data", "fold_{fold}", "fold_negatives.bed"),
        save_dir  = directory(os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "data", "fold_{fold}"))
    params:
        prefix    = lambda wc: os.path.join(root_dir, "singular_replicates", wc.sample, "trained_model_new", "models", "data", f"fold_{wc.fold}", "fold"),
        input_len = INPUT_LEN,
        stride    = 500
    conda: "pjo_chrombpnet"
    threads: 16
    resources:
        mem_mb=64000
    shell:
        """
        rm -f -r {output.save_dir}; \
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
        bam         = lambda wc: bam(wc.sample),
        fasta       = FASTA,
        chrom_sizes = CHROM_SIZES,
        peaks       = lambda wc: peaks(wc.sample),
        negatives   = rules.non_peak_regions.output.path,
        fold_json   = os.path.join(folds_dir, "fold_{fold}.json")
    output:
        report   = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "bias", "fold_{fold}", "evaluation", "overall_report.pdf"),
        model    = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "bias", "fold_{fold}", "models", "bias.h5"),
        save_dir = directory(os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "bias", "fold_{fold}"))
    conda: "pjo_chrombpnet"
    threads: 16
    resources:
        mem_mb=64000,
        gpu=1
    params:
        input_len  = INPUT_LEN,
        output_len = OUTPUT_LEN
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
            -il {params.input_len} \
            -ol {params.output_len} \
	        -ps 4 \
	        -ms -5
        """


rule train_model:
    input:
        bam         = lambda wc: bam(wc.sample),
        fasta       = FASTA,
        chrom_sizes = CHROM_SIZES,
        peaks       = lambda wc: peaks(wc.sample),
        negatives   = rules.non_peak_regions.output.path,
        fold_json   = os.path.join(folds_dir, "fold_{fold}.json"),
        bias_model  = rules.train_bias.output.model
    output:
        report   = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "chrombpnet", "fold_{fold}", "evaluation", "overall_report.pdf"),
        model    = os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "chrombpnet", "fold_{fold}", "models", "chrombpnet_nobias.h5"),
        save_dir = directory(os.path.join(root_dir, "singular_replicates", "{sample}", "trained_model_new", "models", "chrombpnet", "fold_{fold}"))
    conda: "pjo_chrombpnet"
    resources:
        mem_mb=64000,
        gpu=1
    threads: 16
    params:
        libpath       = "/opt/modules/i12g/anaconda/envs/pjo_chrombpnet/lib::/usr/local/cuda/lib64:/usr/local/cuda/lib64",
        n_filters     = N_FILTERS,
        dil_layers    = DIL_LAYERS,
        input_len     = INPUT_LEN,
        output_len    = OUTPUT_LEN,
        max_jitter    = MAX_JITTER,
        learning_rate = LEARNING_RATE
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
            -ol {params.output_len}
        """
