FROM nextflow/nextflow:24.10.5

WORKDIR /app

# set variables
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# install dependencies
RUN yum install -y wget bash tar pip gzip \
    # install and configure conda
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda config --prepend channels bioconda \
    && conda config --prepend channels conda-forge \
    # install conda packages
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    # download pgsc calc scripts
    && nextflow pull pgscatalog/pgsc_calc \
    # install plugins
    && nextflow plugin install nf-schema@2.0.0 \
    && nextflow plugin install nf-prov@1.2.2 
    # install pgs catalog packages
    # pip install pgscatalog-utils pgscatalog-core

# reconfigure scripts
COPY conda_env.yml /app/conda_env.yml
COPY report.yml /app/report.yml
COPY fraposa_env.yml /app/fraposa_env.yml

# create conda environments
RUN conda env create -f /app/conda_env.yml \
    && conda env create -f /app/report.yml \
    && conda env create -f /app/fraposa_env.yml
    # download nf core config profiles 
    #&& wget https://raw.githubusercontent.com/nf-core/configs/master/nfcore_custom.config \
    #&& cp nfcore_custom.config /.nextflow/assets/pgscatalog/pgsc_calc/conf/nfcore_custom.config

# reconfigure more scripts
COPY modules.config /.nextflow/assets/pgscatalog/pgsc_calc/conf/modules.config
COPY base.config /.nextflow/assets/pgscatalog/pgsc_calc/conf/base.config
COPY score_aggregate.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/score_aggregate.nf
COPY intersect_variants.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/ancestry/intersect_variants.nf
COPY combine_scorefiles.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/combine_scorefiles.nf 
COPY match_combine.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/match_combine.nf
COPY match_variants.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/match_variants.nf
#COPY fraposa_project.nf /.nextflow/assets/pgscatalog/pgsc_calc/modules/local/ancestry/oadp/fraposa_project.nf
# remove curl
RUN rm /usr/bin/curl

# NF offline
ENV NXF_OFFLINE='true'
ARG NXF_OFFLINE='true'

USER root
