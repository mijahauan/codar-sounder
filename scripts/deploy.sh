#!/bin/bash
# deploy.sh — refresh editable install + restart instances after a code change.
#
# Usage: sudo ./scripts/deploy.sh [--pull] [--no-restart]

set -euo pipefail

REPO_SOURCE="/opt/git/sigmond/codar-sounder"
VENV_DIR="/opt/codar-sounder/venv"
SERVICE_USER="codarsnd"

DO_PULL=false
DO_RESTART=true
for arg in "$@"; do
    case "$arg" in
        --pull)        DO_PULL=true ;;
        --no-restart)  DO_RESTART=false ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "Must run as root (sudo)" >&2; exit 1
fi

if $DO_PULL; then
    git -c "safe.directory=$REPO_SOURCE" -C "$REPO_SOURCE" pull --ff-only
fi

"$VENV_DIR/bin/pip" install -e "$REPO_SOURCE" >/dev/null
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python3" -c 'import codar_sounder' \
    || { echo "post-install verify failed" >&2; exit 1; }

if $DO_RESTART; then
    mapfile -t units < <(systemctl list-units --plain --no-legend 'codar-sounder@*.service' \
                         2>/dev/null | awk '{print $1}')
    for u in "${units[@]:-}"; do
        [[ -n "$u" ]] || continue
        echo "Restarting $u"
        systemctl restart "$u"
    done
fi
echo "deploy complete"
