process FRAPOSA_PROJECT {
    // labels are defined in conf/modules.config
    label 'process_high_memory'
    label 'fraposa' // controls conda, docker, + singularity options

    tag "${target_geno.baseName.tokenize('_')[1]}"
    
    cachedir = params.genotypes_cache ? file(params.genotypes_cache) : workDir
    storeDir cachedir / "ancestry" / "fraposa" / "project"

    conda "${task.ext.conda}"

    container "${ workflow.containerEngine == 'singularity' &&
        !task.ext.singularity_pull_docker_container ?
        "${task.ext.singularity}${task.ext.singularity_version}" :
        "${task.ext.docker}${task.ext.docker_version}" }"

    input:
    tuple val(meta), path(ref_geno), path(ref_pheno), path(ref_variants),
        path(target_geno), path(target_pheno), path(target_variants), path(split_fam),
        path(pca)

    output:
    tuple val(oadp_meta), path("${output}.pcs"), emit: pca
    path "versions.yml", emit: versions

    script:
    target_id = target_geno.baseName.tokenize('_')[1]
    oadp_meta = ['target_id':target_id]
    output = "${target_geno.baseName}_${split_fam}"
    """
    wc -l ${target_geno.baseName}.fam
    head ${target_geno.baseName}.fam
    tail ${target_geno.baseName}.fam
    wc -l $split_fam
    tail $split_fam
    fraposa ${ref_geno.baseName} \
        --method $params.projection_method \
        --dim_ref 10 \
        --stu_filepref ${target_geno.baseName} \
        --stu_filt_iid $split_fam \
        --out ${target_geno.baseName}_${split_fam}

    cat <<-END_VERSIONS > versions.yml
    ${task.process.tokenize(':').last()}:
        fraposa_pgsc: \$(echo \$(python -c 'import fraposa_pgsc; print(fraposa_pgsc.__version__)'))
    END_VERSIONS
    """
}
