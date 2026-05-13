export APPTAINER_CACHEDIR=/local_scratch/gzappavi/.cache/apptainer

apptainer build --force containers/cuda_ubuntu.sif containers/cuda_ubuntu.def
