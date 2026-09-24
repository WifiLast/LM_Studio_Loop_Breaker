#!/bin/bash -l
# Installs kev.service as a systemd unit and starts it.
# Run this from the checkout on the target machine (e.g. /home/wiffzack/llm/kev), as a
# user with sudo rights: ./install_kev_service.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_kev.sh"
UNIT_SRC="$SCRIPT_DIR/kev.service"
UNIT_DST="/etc/systemd/system/kev.service"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "missing $RUN_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$UNIT_SRC" ]]; then
    echo "missing $UNIT_SRC" >&2
    exit 1
fi

chmod +x "$RUN_SCRIPT"

# kev.service hardcodes /home/wiffzack/llm/kev; make sure this checkout actually lives
# there before installing a unit that points at a path other than this one.
if [[ "$SCRIPT_DIR" != "/home/wiffzack/llm/kev" ]]; then
    echo "warning: this checkout is at $SCRIPT_DIR, but kev.service points at /home/wiffzack/llm/kev" >&2
    echo "edit kev.service's WorkingDirectory/ExecStart (and run_kev.sh's cd) to match, or move the checkout." >&2
fi

echo "installing $UNIT_DST (sudo)..."
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable --now kev.service

sleep 1
sudo systemctl status --no-pager kev.service
echo
echo "logs: journalctl -u kev.service -f"
