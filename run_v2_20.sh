#!/bin/bash
conda_env="hm_chromophore2"
property_name="albuterol_similarity"


for repeat in {1..10}; do
    echo "=== Repeat $repeat ==="
    
    # Launch 10 tmux sessions in parallel
    for tmux_num in {1..10}; do
        session_name="v2_${property_name}_${repeat}_${tmux_num}"

        echo "Starting tmux session: $session_name"

        tmux new-session -d -s "$session_name" bash -i -c '
            source /compuworks/anaconda3/etc/profile.d/conda.sh
            conda activate '"$conda_env"'
            cd /home/khm/chemiloop;
            python v2_sci_rev.py;
            sleep 3;
            tmux kill-session -t $session_name
        '
    done

    # Wait for all 10 tmux sessions to finish
    for tmux_num in {1..10}; do
        session_name="v2_${property_name}_${repeat}_${tmux_num}"

        echo "Waiting for tmux session: $session_name"
        while tmux has-session -t "$session_name" 2>/dev/null; do
            sleep 5
        done
        echo "Finished tmux session: $session_name"
    done

    echo "=== Finished repeat $repeat ==="
done

