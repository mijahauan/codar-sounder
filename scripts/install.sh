#!/bin/bash
# install.sh — first-run bootstrap for codar-sounder (Pattern A editable install)
#
# Usage: sudo ./scripts/install.sh [--pull] [--yes]
#
# What it does:
#   1. Creates service user codarsnd:codarsnd
#   2. Clones/links repo to /opt/git/sigmond/codar-sounder
#   3. Creates venv at /opt/codar-sounder/venv with editable install
#   4. Renders config template (non-destructive — never overwrites)
#   5. Installs systemd unit template
#
# Idempotent: safe to re-run.

set -euo pipefail

SERVICE_USER="codarsnd"
SERVICE_GROUP="codarsnd"
REPO_SOURCE="/opt/git/sigmond/codar-sounder"
VENV_DIR="/opt/codar-sounder/venv"
CONFIG_DIR="/etc/codar-sounder"
CONFIG_FILE="${CONFIG_DIR}/codar-sounder-config.toml"
SPOOL_DIR="/var/lib/codar-sounder"
LOG_DIR="/var/log/codar-sounder"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ui_info()  { echo "[INFO]  $*"; }
ui_warn()  { echo "[WARN]  $*" >&2; }
ui_error() { echo "[ERROR] $*" >&2; }

DO_PULL=false
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        --pull) DO_PULL=true ;;
        --yes)  AUTO_YES=true ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    ui_error "Must run as root (sudo)"
    exit 1
fi

# --- Phase 0.5: ensure uv is on PATH (canonical sigmond-suite installer) ---
_ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        ui_info "uv $(uv --version 2>/dev/null | awk '{print $2}') at $(command -v uv)"
        return
    fi
    ui_info "uv not found -- installing system-wide to /usr/local/bin"
    command -v curl >/dev/null || { ui_error "curl not found (apt install curl)"; exit 1; }
    if ! curl -LsSf https://astral.sh/uv/install.sh | env XDG_BIN_HOME=/usr/local/bin UV_NO_MODIFY_PATH=1 sh; then
        ui_error "uv installer failed"
        exit 1
    fi
    command -v uv >/dev/null || { ui_error "uv installer ran but uv is still not on PATH"; exit 1; }
    ui_info "uv $(uv --version 2>/dev/null | awk '{print $2}') installed"
}
_ensure_uv

# --- Phase 0.6: ensure ka9q-python sibling repo is on disk ---
# pyproject.toml's [tool.uv.sources] declares ka9q-python as a path-based
# editable dep at ../ka9q-python.  uv sync needs the directory to exist
# at /opt/git/sigmond/ka9q-python or it fails.
if [[ ! -f /opt/git/sigmond/ka9q-python/pyproject.toml ]]; then
    ui_info "ka9q-python sibling repo not at /opt/git/sigmond/ka9q-python -- cloning"
    mkdir -p /opt/git/sigmond
    git clone https://github.com/mijahauan/ka9q-python /opt/git/sigmond/ka9q-python \
        || { ui_error "Failed to clone ka9q-python"; exit 1; }
fi

# --- Phase 1: service user ---
if ! id -u "$SERVICE_USER" &>/dev/null; then
    ui_info "Creating service user $SERVICE_USER"
    useradd --system --shell /usr/sbin/nologin \
            --home-dir /nonexistent --no-create-home \
            "$SERVICE_USER"
fi

# --- Phase 2: repo + venv ---
if [[ ! -d "$REPO_SOURCE" ]] && [[ ! -L "$REPO_SOURCE" ]]; then
    ui_info "Linking $REPO_ROOT -> $REPO_SOURCE"
    mkdir -p "$(dirname "$REPO_SOURCE")"
    ln -sfn "$REPO_ROOT" "$REPO_SOURCE"
fi

# Pattern A traversability check
if ! sudo -u "$SERVICE_USER" test -r "$REPO_SOURCE/src/codar_sounder/__init__.py"; then
    ui_error "Service user $SERVICE_USER cannot read $REPO_SOURCE/src/codar_sounder/__init__.py"
    ui_error "Fix: ensure the repo is at /opt/git/sigmond/codar-sounder (not under a mode-700 home),"
    ui_error "  or chmod g+rx the path and add $SERVICE_USER to the owner's group"
    exit 1
fi

if $DO_PULL; then
    ui_info "Pulling latest from origin"
    git -C "$REPO_SOURCE" pull --ff-only
fi

if [[ ! -d "$VENV_DIR" ]]; then
    ui_info "Creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    # --seed populates pip/setuptools/wheel for compatibility with tooling
    # that shells out to pip; harmless overhead otherwise.
    uv venv "$VENV_DIR" --python 3.11 --seed --quiet
fi

# uv sync reads pyproject.toml + uv.lock, resolves [tool.uv.sources]
# (ka9q-python editable from ../ka9q-python), installs codar-sounder
# itself editable, and pins exactly what's in uv.lock.  --no-dev skips
# dev extras (pytest etc.); --frozen requires uv.lock to be current
# (regenerate locally with `uv lock` if siblings or deps have shifted).
ui_info "Syncing codar-sounder + ka9q-python (editable) into $VENV_DIR"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    uv sync --project "$REPO_SOURCE" --frozen --no-dev --quiet

# sigmond is the host-wide orchestrator; codar-sounder lazy-imports
# sigmond.hamsci_sink.Writer for the spots SQLite sink (with a no-op
# fallback when absent).  Not declared in codar-sounder's pyproject
# so uv sync doesn't install it; explicit uv pip install when the
# sibling exists.  uv pip install needs --python (UV_PROJECT_ENVIRONMENT
# only applies to project commands like uv sync).
if [[ -d /opt/git/sigmond/sigmond ]]; then
    ui_info "Installing sigmond (editable) into venv"
    uv pip install --quiet --python "$VENV_DIR/bin/python3" -e /opt/git/sigmond/sigmond
else
    ui_info "sigmond repo not found at /opt/git/sigmond/sigmond -- SQLite spot sink"
    ui_info "  will resolve to a no-op fallback."
fi

# Post-install verify
if ! sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python3" -c 'import codar_sounder' 2>/dev/null; then
    ui_error "Post-install verify failed: $SERVICE_USER cannot import codar_sounder"
    exit 1
fi
ui_info "Post-install verify OK"

# --- Phase 3: config ---
mkdir -p "$CONFIG_DIR"
if [[ ! -f "$CONFIG_FILE" ]]; then
    ui_info "Rendering config template -> $CONFIG_FILE"
    cp "$REPO_SOURCE/config/codar-sounder-config.toml.template" "$CONFIG_FILE"
    ui_warn "Edit $CONFIG_FILE: callsign, grid, receiver_lat/lon, [[radiod]] + [[radiod.transmitter]] blocks"
else
    ui_info "Config exists at $CONFIG_FILE — not overwriting"
fi

# --- Phase 4: directories ---
for dir in "$SPOOL_DIR" "$LOG_DIR"; do
    mkdir -p "$dir"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$dir"
done

# --- Phase 5: systemd ---
ui_info "Installing systemd unit template"
install -o root -g root -m 644 \
    "$REPO_SOURCE/systemd/codar-sounder@.service" \
    /etc/systemd/system/codar-sounder@.service
systemctl daemon-reload

ui_info "Install complete. Edit $CONFIG_FILE then enable an instance with:"
ui_info "  sudo systemctl enable --now codar-sounder@<radiod-id>"
ui_info ""
ui_info "Note: v0.1 daemon is a stub — it idles and logs what it WOULD dechirp."
ui_info "Signal-processing engine (FMCW dechirping per Kaeppler 2022) lands in v0.2."
