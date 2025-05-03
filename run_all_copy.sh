#!/bin/bash

# List of tasks
tasks=(
    # "albuterol_similarity"
    # "amlodipine_mpo"
    # "celecoxib_rediscovery"
    # "deco_hop"
    # "drd2"
    # "fexofenadine_mpo"
    # "gsk3b"
    "isomers_c7h8n2o2"
    "isomers_c9h10n2o2pf2cl"
    "jnk3"
    "median1"
    "median2"
    "mestranol_similarity"
    "osimertinib_mpo"
    "perindopril_mpo"
    "qed"
    "ranolazine_mpo"
    "scaffold_hop"
    "sitagliptin_mpo"
    "thiothixene_rediscovery"
    "troglitazon_rediscovery"
    "valsartan_smarts"
    "zaleplon_mpo"
)

# List of versions
# versions=("v1.py" "v2.py" "v3.py" "v4.py")
versions=("v4.py")

# Initialize api_num
api_num=1

# Your conda environment name
conda_env="hm_chromophore2"  # <<< Update this to your real environment!

# Loop over tasks and versions
for task in "${tasks[@]}"
do
  for version in "${versions[@]}"
  do
    session_name="${version}_${task}_api_${api_num}"
    echo "Starting tmux session: $session_name"

    # Correct expansion
    tmux new-session -d -s "$session_name" "
      source /compuworks/anaconda3/etc/profile.d/conda.sh;
      conda activate $conda_env;
      cd /home/khm/chemiloop;
      python $version --task $task --api-num $api_num;
      exec bash
    "

    # Increment api_num and wrap back to 1 after 48
    ((api_num++))
    if [ "$api_num" -gt 11 ]; then
      api_num=1
    fi
  done
done
