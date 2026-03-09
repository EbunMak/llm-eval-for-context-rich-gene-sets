module load python/3.12.5

python compare_prf_plots.py \
  --csvs \
    out/genesets/evaluation/per_phenotype_prf.csv \
    out/genesets/deepseek-r1:8b/evaluation/per_phenotype_prf.csv \
    out/genesets/llama3.1:8b/evaluation/per_phenotype_prf.csv \
    out/genesets/qwen3:32b/evaluation/per_phenotype_prf.csv \
    out/direct-prompting/gmts/deepseek/evaluation/per_phenotype_prf.csv \
    out/direct-prompting/gmts/llama3/evaluation/per_phenotype_prf.csv \
    out/direct-prompting/gmts/qwen/evaluation/per_phenotype_prf.csv \
  --labels \
    consensus \
    deepseek-r1 \
    llama3.1 \
    qwen3 \
    deepseek-r1-prompt-only \
    llama3.1-prompt-only \
    qwen3-prompt-only \
  --out_dir out/prf_comparison_plots

