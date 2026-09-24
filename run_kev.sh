#!/bin/bash
# Kev System One server — started by kev.service.
# systemd starts this with no login/interactive shell, so ~/.bashrc (where conda init
# usually lives) never runs and plain `conda activate` fails with "conda: command not
# found". Source conda.sh by path instead, from whichever install actually has it.
set -euo pipefail

cd /home/wiffzack/llm/kev

CONDA_BASE=""
for candidate in "/mnt/data/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="$candidate"
        break
    fi
done
if [[ -z "$CONDA_BASE" ]]; then
    echo "run_kev.sh: could not find conda.sh under \$HOME/{miniconda3,anaconda3,miniforge3} or /opt/{conda,miniconda3}" >&2
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate webui

# uv is usually a per-user install (its own installer, or pipx), not inside the conda
# env, so it's not guaranteed to be on PATH in this non-interactive shell either.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "run_kev.sh: 'uv' not found on PATH (checked \$HOME/.local/bin, \$HOME/.cargo/bin); run 'which uv' in your interactive (webui) shell and add its directory here" >&2
    exit 1
fi

export OLLAMA_HOST="http://10.0.0.10:11434"

exec uv run --extra serve python -m kev.serve --ollama qwen-hauhaucs:latest --port 8009
