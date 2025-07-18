#!/bin/bash

# VRAM usage threshold in MiB (10.5 GB * 1024 MiB/GB)
THRESHOLD_MIB=${THRESHOLD_MIB:-10752} # 10.5 GiB = 10.5 * 1024 MiB
# Threshold used when running in CPU mode (in MiB)
CPU_THRESHOLD_MIB=${CPU_THRESHOLD_MIB:-2048}
# Check interval in seconds
# Allow overriding the check interval for testing
INTERVAL=${INTERVAL:-60}
# Number of consecutive checks above threshold to trigger restart
CONSECUTIVE_CHECKS_LIMIT=3

# If set to 1 the script will use host memory metrics instead of nvidia-smi.
CPU_MODE=${CPU_MODE:-0}
# When >0 the loop exits after the given number of iterations. Useful for CI.
MAX_ITERATIONS=${MAX_ITERATIONS:-0}

# Counter for consecutive high VRAM usage checks
consecutive_high_usage_count=0

# Log file
LOG_FILE="${LOG_FILE:-/var/log/vram_watchdog.log}"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# Function to get llm-sidecar container ID
get_llm_container_id() {
    # Adjust the filter based on how Docker Compose names the container.
    # Usually it's <project_name>-<service_name>-<instance_number>.
    # Using a filter that checks for a label 'com.docker.compose.service=llm-sidecar' is robust.
    # If not using labels, a name filter like '*llm-sidecar*' can be used but might be less precise.
    # For now, assuming a simple name filter will work or that the container is named 'llm-sidecar'.
    docker ps -q --filter "name=llm-sidecar" | head -n 1
}

log_message "VRAM Watchdog started. Monitoring llm-sidecar."
log_message "VRAM Threshold: ${THRESHOLD_MIB}MiB. Check Interval: ${INTERVAL}s. Consecutive Limit: ${CONSECUTIVE_CHECKS_LIMIT}. CPU mode: ${CPU_MODE}."

iteration_count=0
while true; do
    if [ "$MAX_ITERATIONS" -gt 0 ] && [ "$iteration_count" -ge "$MAX_ITERATIONS" ]; then
        log_message "Reached max iterations ($MAX_ITERATIONS). Exiting."
        break
    fi

    LLM_CONTAINER_ID=$(get_llm_container_id)

    if [ -z "$LLM_CONTAINER_ID" ]; then
        log_message "Error: llm-sidecar container not found. Will retry."
        sleep $INTERVAL
        continue
    fi

    if [ "$CPU_MODE" -eq 1 ]; then
        usage_output=$(docker exec $LLM_CONTAINER_ID free -m | awk '/^Mem:/ {print $3}')
    else
        # Execute nvidia-smi inside the llm-sidecar container
        usage_output=$(docker exec $LLM_CONTAINER_ID nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    fi
    
    if [ -z "$usage_output" ]; then
        log_message "Error: 'docker exec $LLM_CONTAINER_ID nvidia-smi' command failed or returned empty. Ensure llm-sidecar has NVIDIA tools and is running."
        sleep $INTERVAL
        continue
    fi

    # nvidia-smi output is typically just the number of MiB used, e.g., "1500"
    used_mem_mib=$(echo $usage_output | xargs) # xargs to trim whitespace

    # Check if used_mem_mib is a valid number
    if ! [[ "$used_mem_mib" =~ ^[0-9]+$ ]]; then
        log_message "Error: Could not parse VRAM usage from nvidia-smi output of llm-sidecar. Output: '$usage_output'. Parsed used: '$used_mem_mib'."
        sleep $INTERVAL
        continue
    fi

    threshold_to_use=$THRESHOLD_MIB
    [ "$CPU_MODE" -eq 1 ] && threshold_to_use=$CPU_THRESHOLD_MIB

    log_message "Current memory usage: ${used_mem_mib}MiB (threshold ${threshold_to_use}MiB)."

    if [ "$used_mem_mib" -gt "$threshold_to_use" ]; then
        ((consecutive_high_usage_count++))
        log_message "Usage (${used_mem_mib}MiB) is above threshold (${threshold_to_use}MiB). Consecutive count: $consecutive_high_usage_count."
    else
        if [ "$consecutive_high_usage_count" -gt 0 ]; then
            log_message "Usage (${used_mem_mib}MiB) is below threshold. Resetting consecutive count."
        fi
        consecutive_high_usage_count=0
    fi

    if [ "$consecutive_high_usage_count" -ge "$CONSECUTIVE_CHECKS_LIMIT" ]; then
        log_message "VRAM usage has been above threshold for $consecutive_high_usage_count consecutive checks. Attempting to restart llm-sidecar..."
        
        # LLM_CONTAINER_ID is already fetched at the start of the loop
        if [ -n "$LLM_CONTAINER_ID" ]; then # Should always be true if we got this far
            log_message "Attempting to restart llm-sidecar (ID: $LLM_CONTAINER_ID)..."
            docker restart $LLM_CONTAINER_ID
            restart_status=$?
            if [ $restart_status -eq 0 ]; then
                log_message "llm-sidecar container (ID: $LLM_CONTAINER_ID) restarted successfully."
            else
                log_message "Error restarting llm-sidecar container (ID: $LLM_CONTAINER_ID). Exit code: $restart_status."
            fi
        else
            # This case should ideally not be reached if LLM_CONTAINER_ID was found earlier in the loop.
            log_message "Consistency error: LLM_CONTAINER_ID was lost or not found before restart attempt."
        fi
        
        # Reset count after attempting restart
        consecutive_high_usage_count=0
    fi

    iteration_count=$((iteration_count + 1))
    sleep $INTERVAL
done
