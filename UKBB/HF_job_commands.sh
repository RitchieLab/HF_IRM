# extract pgs catalog variants and create plink files- run the following code in terminal
bgen_input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/Bulk/Imputation/Imputation from genotype (TOPmed)"
variants_input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/input"
output_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/output/plink"

for chr in $(seq 1 22) X
do
dx run app-swiss-army-knife \
-y \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen.bgi" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.bgen" \
-iin="${bgen_input_dir}/ukb21007_c${chr}_b0_v1.sample" \
-iin="${variants_input_dir}/HF.PGS005097.var_list.txt" \
--brief \
--name "extact_pgs_vars.chr${chr}" \
-icmd="plink2 --bgen ukb21007_c${chr}_b0_v1.bgen ref-first \
                --sample ukb21007_c${chr}_b0_v1.sample \
                --extract range HF.PGS005097.var_list.txt \
                --set-all-var-ids @:#:\\\$r:\\\$a \
                --new-id-max-allele-len 1000 \
                --make-pgen \
                --out ukb21007_c${chr}_b0_v1.topmed_imputed.hf_pgs_vars" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# submit pgsc-calc job- run the following code in terminal
plink_input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/output/plink"
other_input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/input/"
output_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/output/pgsc_calc"

input_flags=()
for file in $(dx ls "${plink_input_dir}" --brief)
do
input_flags+=("-iinput=${plink_input_dir}/${file}")
done

dx run pgsc_calc \
-y \
"${input_flags[@]}" \
-iinput="${other_input_dir}/UKBB.HF.PGSC_CALC.samplesheet.csv" \
-iinput="${other_input_dir}/pgsc_HGDP+1kGP_v1.tar.zst" \
-iinput="${other_input_dir}/PGS005097_hmPOS_GRCh38.txt.gz" \
-icommand="nextflow run pgscatalog/pgsc_calc -profile conda \
            --input input/UKBB.HF.PGSC_CALC.samplesheet.csv \
            --scorefile input/PGS005097_hmPOS_GRCh38.txt.gz \
            --target_build GRCh38 \
            --outdir output/ \
            --max_cpus 64 \
            --max_memory 256.GB \
            --min_overlap 0.0 \
            --max_time 240.h \
            --run_ancestry input/pgsc_HGDP+1kGP_v1.tar.zst \
            --keep_multiallelic True \
            --hwe_ref 0 \
            --pca_maf_target 0.05" \
--brief \
--name "pgsc_calc" \
--destination ${output_dir} \
--instance-type mem2_ssd1_v2_x64

# feature selection python script- run the following code in terminal
input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/input/"
scripts_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/scripts/"
output_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/output/regressions/training/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/hf_training_script.py" \
-iin="${input_dir}/HF_Clinical_PGS.phenotype.no_missing.variable_transformation.csv" \
--brief \
--name "training_${i}" \
-icmd="python hf_training_script.py \
                --input HF_Clinical_PGS.phenotype.no_missing.variable_transformation.csv \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done

# module evaluation python script- run the following code in terminal
input_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/input/"
scripts_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/scripts/"
output_dir="project-GgYky4QJj5pkv6G9P8Zp6P26:/HF_Clinical_PGS/output/regressions/eval/"

for i in $(seq 1 1000)
do
dx run app-swiss-army-knife \
-y \
-iin="${scripts_dir}/hf_eval_script.py" \
-iin="${input_dir}/HF_Clinical_PGS.phenotype.no_missing.variable_transformation.csv" \
-iin="${input_dir}/UKBB.significant_vars_95.csv" \
-iin="${input_dir}/UKBB.important_vars_95.csv" \
-iin="${input_dir}/UKBB.LR_beta_all_iter.csv" \
--brief \
--name "eval_${i}" \
-icmd="python hf_eval_script.py \
                --input HF_Clinical_PGS.phenotype.no_missing.variable_transformation.csv \
                --sig UKBB.significant_vars_95.csv \
                --important UKBB.important_vars_95.csv \
                --beta UKBB.LR_beta_all_iter.csv \
                --iter ${i}" \
--destination ${output_dir} \
--instance-type mem1_ssd2_v2_x8
done