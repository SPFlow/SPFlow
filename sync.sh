#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_name="$(basename "$script_dir")"

# Destination layout on remote hosts.
remote_projects_root="${SYNC_REMOTE_PROJECTS_ROOT:-~}"
remote_results_root="${SYNC_REMOTE_RESULTS_ROOT:-~/results}"
ssh_bin="${SYNC_SSH_BIN:-/usr/bin/ssh}"
ssh_opts="${SYNC_SSH_OPTS:--T -o RequestTTY=no -o RemoteCommand=none}"
rsync_rsh="${ssh_bin} ${ssh_opts}"
sync_delete="${SYNC_DELETE:-0}"

# Source layout on local machine.
local_code_root="$script_dir"
local_results_root="${SYNC_LOCAL_RESULTS_ROOT:-$script_dir/results}"

# Default hosts if none are provided on the command line.
# Override with: SYNC_HOSTS="host1 host2" ./sync.sh
IFS=' ' read -r -a default_hosts <<< "${SYNC_HOSTS:-dgxb}"

sync_code() {
    local host="$1"
    local remote_project_dir="${remote_projects_root%/}/$project_name"
    local -a delete_flag=()

    if [ "$sync_delete" = "1" ]; then
        delete_flag+=(--delete)
    fi

    rsync -av "${delete_flag[@]}" -e "$rsync_rsh" \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.mypy_cache/' \
        --exclude='.pytest_cache/' \
        --exclude='.ruff_cache/' \
        --exclude='__pycache__/' \
        --exclude='.coverage' \
        --exclude='coverage.xml' \
        --exclude='htmlcov/' \
        --exclude='build/' \
        --exclude='dist/' \
        --exclude='*.egg-info/' \
        --exclude='.idea/' \
        --exclude='data/' \
        --exclude='results/' \
        "$local_code_root/" "$host:$remote_project_dir/"

    echo "[$host] synced CODE -> $remote_project_dir"
}

sync_results() {
    local host="$1"
    local remote_results_dir=""
    local output

    mkdir -p "$local_results_root"

    # Prefer results written inside the synced project tree (e.g., ~/SPFlow/results).
    # Fall back to the legacy shared results layout (e.g., ~/results/SPFlow).
    for candidate in \
        "${remote_projects_root%/}/$project_name/results" \
        "${remote_results_root%/}/$project_name"
    do
        if output="$(
            rsync -av --list-only -e "$rsync_rsh" "$host:$candidate/" "$local_results_root/" 2>&1
        )"; then
            remote_results_dir="$candidate"
            break
        fi
    done

    if [ -z "$remote_results_dir" ]; then
        echo "[$host] skipped RESULTS (missing remote dirs: ${remote_projects_root%/}/$project_name/results and ${remote_results_root%/}/$project_name)"
        return 0
    fi

    if ! output="$(
        rsync -av --inplace -e "$rsync_rsh" \
            --exclude='*.pth' \
            --exclude='*.pt' \
            --exclude='*.ckpt' \
            --exclude='*.npy' \
            --exclude='events.out.tfevents*' \
            --exclude='wandb/' \
            --exclude='.hydra/' \
            "$host:$remote_results_dir/" "$local_results_root/" 2>&1
    )"; then
        if printf '%s' "$output" | grep -q "No such file or directory"; then
            echo "[$host] skipped RESULTS (missing remote dir: $remote_results_dir)"
            return 0
        fi
        printf '%s\n' "$output" >&2
        return 1
    fi

    printf '%s\n' "$output"
    echo "[$host] synced RESULTS -> $local_results_root"
}

sync_all() {
    local host="$1"

    sync_code "$host" &
    sync_results "$host" &
    wait
}

if [ "$#" -gt 0 ]; then
    hosts=("$@")
else
    hosts=("${default_hosts[@]}")
fi

if [ "${#hosts[@]}" -eq 0 ]; then
    echo "No sync hosts configured. Pass hosts as args or set SYNC_HOSTS."
    exit 1
fi

echo "Synchronizing '$project_name' with hosts: ${hosts[*]}"
for host in "${hosts[@]}"; do
    sync_all "$host"
done
