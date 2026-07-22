root_dir = "/s/project/multispecies/fungi_code/atac/data/chrom_bp_pipeline/GSE66386/control_pipeline_11/"
saccer_index = "/s/project/multispecies/fungi_code/atac/debug/schep/data/index/index"
BLACKLIST = "/s/project/multispecies/fungi_code/atac/debug/schep/data/sacCer_blacklist.bed"
CHROM_SIZES = "/s/project/multispecies/fungi_code/atac/debug/schep/data/saccCer.chrom.sizes"

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

rule all:
    input:
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "results", "04_shifted", "{sample}.shifted.bam"),
            sample=samples
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "results", "05_merged", "all_merged.sorted.shifted.bam"),
            sample=samples
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "results", "05_merged", "all_merged.sorted.shifted.bam.bai"),
            sample=samples
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks", "all_merged_relaxed_peaks.narrowPeak"),
            sample=samples
        ),
        expand(
            os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks", "all_merged_relaxed_peaks_no_blacklist.bed"),
            sample=samples
        )

print(samples)

rule trim:
    input:
        mate_1 = os.path.join(root_dir, "data", "fastq", "{sample}_1.fastq"),
        mate_2 = os.path.join(root_dir, "data", "fastq", "{sample}_2.fastq")
    output:
        trimmed_1 = os.path.join(root_dir, "data", "fastq", "trimmed", "{sample}_1.trimmed.fastq"),
        trimmed_2 = os.path.join(root_dir, "data", "fastq", "trimmed", "{sample}_2.trimmed.fastq")
    resources:
        mem_mb=32000
    threads: 8
    shell:
        """
        fastp \
            -i {input.mate_1} \
            -I {input.mate_2} \
            -o {output.trimmed_1} \
            -O {output.trimmed_2} \
            --detect_adapter_for_pe \
            -w 8
        """

rule align:
    input:
        mate_1 = rules.trim.output.trimmed_1,
        mate_2 = rules.trim.output.trimmed_2
    output:
        bam = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "00_alignment", "raw", "{sample}.bam")
    params:
        index = saccer_index
    threads: 16
    resources:
        mem_mb=32000
    shell:
        """
        bowtie2 \
            -x {params.index} \
            -1 {input.mate_1} \
            -2 {input.mate_2} \
            -t -q -N 1 -L 25 -X 2000 \
            -p {threads} \
            --no-mixed --no-discordant --no-unal \
        | samtools view -b -@{threads} -o {output.bam} -
        """

rule sort_and_index:
    input:
        bam = rules.align.output.bam
    output:
        sorted_bam = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "00_alignment", "sorted", "{sample}.sorted.bam"),
        flagstat   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "00_alignment", "00_alignment", "sorted", "{sample}.sorted.flagstat.txt")
    threads: 16
    resources:
        mem_mb=32000
    shell:
        """
        samtools sort -@{threads} -o {output.sorted_bam} {input.bam};
        samtools index {output.sorted_bam};
        samtools flagstat {output.sorted_bam} > {output.flagstat}
        """

rule remove_duplicates:
    input:
        bam = rules.sort_and_index.output.sorted_bam
    output:
        dedup_grp_tmp = temp(os.path.join(root_dir, "singular_replicates", "{sample}", "results", "02_duplicated_marked", "{sample}.grouped.bam")),
        dedup_bam     = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "02_duplicated_marked", "{sample}.dedup.bam"),
        dedup_index   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "02_duplicated_marked", "{sample}.dedup.bam.bai"),
        dedup_metrics = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "alignment", "02_duplicated_marked", "{sample}.dedup.metrics.txt")
    resources:
        mem_mb=32000
    threads: 16
    shell:
        """
        picard AddOrReplaceReadGroups \
            I={input.bam} O={output.dedup_grp_tmp} \
            RGID={wildcards.sample} RGLB=lib1 RGPL=ILLUMINA RGPU=unit1 RGSM={wildcards.sample};
        picard MarkDuplicates \
            I={output.dedup_grp_tmp} O={output.dedup_bam} \
            M={output.dedup_metrics} REMOVE_DUPLICATES=true;
        samtools index {output.dedup_bam}
        """

rule filter_reads:
    input:
        bam   = rules.remove_duplicates.output.dedup_bam,
        index = rules.remove_duplicates.output.dedup_index
    output:
        filtered       = temp(os.path.join(root_dir, "singular_replicates", "{sample}", "results", "03_filtered", "{sample}.filtered.bam")),
        filtered_index = temp(os.path.join(root_dir, "singular_replicates", "{sample}", "results", "03_filtered", "{sample}.filtered.bam.bai")),
        sorted_bam     = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "03_filtered", "{sample}.filtered.sorted.bam"),
        sorted_index   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "03_filtered", "{sample}.filtered.sorted.bam.bai")
    threads: 8
    resources:
        mem_mb=32000
    shell:
        """
        samtools view -b -@{threads} -f 2 -F 1804 -q 30 {input.bam} > {output.filtered};
        samtools index {output.filtered};
        samtools sort -@{threads} -o {output.sorted_bam} {output.filtered};
        samtools index {output.sorted_bam};
        """

rule shift_reads:
    input:
        bam   = rules.filter_reads.output.sorted_bam,
        index = rules.filter_reads.output.sorted_index
    output:
        shifted_bam   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "04_shifted", "{sample}.shifted.bam"),
        sorted_bam    = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "04_shifted", "{sample}.shifted.sorted.bam"),
        shifted_index = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "04_shifted", "{sample}.shifted.sorted.bam.bai")
    threads: 8
    resources:
        mem_mb=32000
    shell:
        """
        alignmentSieve --bam {input.bam} --ATACshift -o {output.shifted_bam};
        samtools sort -@{threads} -o {output.sorted_bam} {output.shifted_bam};
        samtools index {output.sorted_bam}
        """

rule merge_bams:
    input:
        bams = lambda wc: [os.path.join(root_dir, "singular_replicates", wc.sample, "results", "04_shifted", f"{wc.sample}.shifted.bam")]
    output:
        merged_bam          = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "05_merged", "all_merged.unsorted.shifted.bam"),
        merged_sorted_bam   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "05_merged", "all_merged.sorted.shifted.bam"),
        merged_sorted_index = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "05_merged", "all_merged.sorted.shifted.bam.bai")
    threads: 8
    resources:
        mem_mb=32000
    shell:
        """
        rm -f {output.merged_bam} {output.merged_sorted_bam} {output.merged_sorted_index} \
            && samtools merge {output.merged_bam} {input.bams} \
            && samtools sort -@{threads} {output.merged_bam} -o {output.merged_sorted_bam} \
            && samtools index {output.merged_sorted_bam}
        """

rule call_peaks:
    input:
        rules.merge_bams.output.merged_sorted_bam
    output:
        out_dir = directory(os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks")),
        peaks   = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks", "all_merged_relaxed_peaks.narrowPeak")
    resources:
        mem_mb=32000
    threads: 16
    shell:
        """
        macs2 callpeak \
            -t {input} -f BAM \
            -n all_merged_relaxed \
            --outdir {output.out_dir} \
            -g 1.2e7 -p 0.01 \
            --nomodel --shift -100 --extsize 200 \
            --keep-dup all --call-summits
        """

rule remove_blacklist_peaks:
    input:
        peaks       = rules.call_peaks.output.peaks,
        blacklist   = BLACKLIST,
        chrom_sizes = CHROM_SIZES
    output:
        tmp_bed      = temp(os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks", "all_merged_relaxed_peaks_blacklist_slopped.tmp.bed")),
        filtered_bed = os.path.join(root_dir, "singular_replicates", "{sample}", "results", "06_peaks", "all_merged_relaxed_peaks_no_blacklist.bed")
    shell:
        """
        bedtools slop -i {input.blacklist} -g {input.chrom_sizes} -b 1057 > {output.tmp_bed}; \
        bedtools intersect -v -a {input.peaks} -b {output.tmp_bed} > {output.filtered_bed}
        """
