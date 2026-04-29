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
    python3 -m venv "$VENV_DIR"
fi

ui_info "Installing codar-sounder (editable) into venv"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel >/dev/null
"$VENV_DIR/bin/pip" install -e "$REPO_SOURCE" >/dev/null

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
