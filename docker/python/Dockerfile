FROM continuumio/miniconda3
WORKDIR /app

# install common python packages & bash tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends coreutils grep gzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && conda install -y -n base -c conda-forge -c bioconda imbalanced-learn statsmodels scikit-learn scipy pandas seaborn matplotlib numpy python=3.11 \
    && conda clean --all --yes

USER root
