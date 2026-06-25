#!/bin/bash
#
# codar-sounder config wizard (whiptail).
#
# Called by `codar-sounder config init` / `config edit` when stdout is a
# TTY and whiptail is installed (gate in configurator.py via
# sigmond.wizard_dispatch).  The centrepiece is the TRANSMITTER PICKER:
# instead of hand-writing [[radiod.transmitter]] blocks, the operator
# multi-selects from the known CODAR sites ranked by distance from the
# receiver (`codar-sounder stations --json`).  The chosen sites are
# grouped by band into one [[radiod]] block each and written through
# `codar-sounder config apply --json -`, so the Python side owns
# schema/type validation and the units bridge (MHz → Hz) is never done
# by hand.
#
# Usage:
#   config-wizard.sh init [--config <path>]
#   config-wizard.sh edit [--config <path>]
#
# Env (set by configurator.py before exec):
#   CODAR_SOUNDER_CLI         path to the codar-sounder binary to use
# Reads (read-only) for pre-fills:
#   /etc/sigmond/coordination.env   STATION_CALL/GRID/LAT/LON, radiod status

set -euo pipefail

MODE="${1:-init}"; shift || true
CONFIG_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        *) echo "config-wizard: unknown arg: $1" >&2; exit 2 ;;
    esac
done

CODAR="${CODAR_SOUNDER_CLI:-codar-sounder}"
COORD_ENV="/etc/sigmond/coordination.env"

# -------- shared shell helpers ----------------------------------------
SIGMOND_WIZARD_LIB_SH="${SIGMOND_WIZARD_LIB_SH:-/opt/git/sigmond/sigmond/lib/sigmond/wizard_dispatch/wizard_dispatch.sh}"
if [[ -r "$SIGMOND_WIZARD_LIB_SH" ]]; then
    # shellcheck disable=SC1090
    . "$SIGMOND_WIZARD_LIB_SH"
else
    HEIGHT=20; WIDTH=78; LIST_HEIGHT=12
    _info() { printf '  %s\n'                "$*" >&2; }
    _warn() { printf '  \033[33m⚠\033[0m %s\n' "$*" >&2; }
    _err()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; }
    preflight_or_exit_2() {
        command -v whiptail >/dev/null 2>&1 \
            || { _err "whiptail not on PATH"; exit 2; }
        [[ -t 1 ]] || { _err "stdout is not a TTY"; exit 2; }
    }
fi
BACKTITLE="codar-sounder configuration"

preflight_or_exit_2

# -------- read coordination.env defaults (best effort) ----------------
ENV_CALL=""; ENV_GRID=""; ENV_LAT=""; ENV_LON=""; ENV_STATUS=""
if [[ -r "$COORD_ENV" ]]; then
    # shellcheck disable=SC1090
    set -a; . "$COORD_ENV" 2>/dev/null || true; set +a
fi
ENV_CALL="${STATION_CALL:-}"
ENV_GRID="${STATION_GRID:-}"
ENV_LAT="${STATION_LAT:-}"
ENV_LON="${STATION_LON:-}"
ENV_STATUS="${SIGMOND_RADIOD_STATUS:-}"

# If editing, prefer current config values for the prefills.
cur_json=""
if [[ "$MODE" == "edit" || -n "$CONFIG_PATH" ]]; then
    cur_json=$("$CODAR" config show --json ${CONFIG_PATH:+--config "$CONFIG_PATH"} 2>/dev/null || echo "{}")
fi
prefill() {  # prefill <jq-ish python path> <env fallback>
    PYJSON="$cur_json" python3 - "$1" "$2" <<'PYEOF' 2>/dev/null || true
import json, os, sys
path, fallback = sys.argv[1], sys.argv[2]
try:
    d = json.loads(os.environ.get("PYJSON") or "{}")
except Exception:
    d = {}
cur = d
for part in path.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = None
        break
print(cur if cur not in (None, "") else fallback)
PYEOF
}

# -------- ask helpers --------------------------------------------------
ask_input() {  # ask_input <title> <text> <default> -> stdout
    whiptail --title "$1" --backtitle "$BACKTITLE" \
             --inputbox "$2" "$HEIGHT" "$WIDTH" "$3" 3>&1 1>&2 2>&3
}

# -------- 1. station identity + receiver location ---------------------
CALL=$(ask_input "Station callsign" "Your station callsign (reporter id)." \
                 "$(prefill station.callsign "$ENV_CALL")") || exit 0
GRID=$(ask_input "Grid square" "Maidenhead grid (optional)." \
                 "$(prefill station.grid_square "$ENV_GRID")") || exit 0
RX_LAT=$(ask_input "Receiver latitude" \
    "Receiver latitude in decimal degrees (drives TX-RX distance + secant law)." \
    "$(prefill station.receiver_lat "$ENV_LAT")") || exit 0
RX_LON=$(ask_input "Receiver longitude" \
    "Receiver longitude in decimal degrees." \
    "$(prefill station.receiver_lon "$ENV_LON")") || exit 0

if ! python3 -c "float('$RX_LAT'); float('$RX_LON')" 2>/dev/null; then
    whiptail --title "Invalid coordinates" --backtitle "$BACKTITLE" \
        --msgbox "Receiver lat/lon must be decimal degrees.  Aborting; config unchanged." \
        10 "$WIDTH"
    exit 1
fi

# -------- 2. radiod source --------------------------------------------
RADIOD_STATUS=$(ask_input "Radiod status (mDNS)" \
    "The radiod control/status multicast name to consume from
(e.g. bee1-status.local)." \
    "$(prefill radiod.0.status "$ENV_STATUS")") || exit 0
if [[ -z "$RADIOD_STATUS" || "$RADIOD_STATUS" == "<configure-via-config-init>" ]]; then
    whiptail --title "Radiod required" --backtitle "$BACKTITLE" \
        --msgbox "A radiod status (mDNS) name is required.  Aborting; config unchanged." \
        10 "$WIDTH"
    exit 1
fi

# -------- 3. transmitter picker ---------------------------------------
# Pull the ranked inventory for this receiver.  The list is the heart of
# the wizard: the operator multi-selects which CODAR sites to record.
STN_JSON=$("$CODAR" stations --receiver-lat "$RX_LAT" --receiver-lon "$RX_LON" \
                    --max-range-km 3000 --json 2>/dev/null || echo "")
if [[ -z "$STN_JSON" ]]; then
    whiptail --title "Station inventory unavailable" --backtitle "$BACKTITLE" \
        --msgbox "Could not read the CODAR station inventory.  Is data/codar-stations.toml present?  Aborting." \
        10 "$WIDTH"
    exit 1
fi

# Build whiptail --checklist args: TAG=id, ITEM="freq dist band assoc",
# STATE=on for prime one-hop sites.  Emitted by python, read with mapfile
# so spaces in ITEM survive (NUL-delimited).
mapfile -d '' CHECK_ARGS < <(STN_JSON="$STN_JSON" python3 <<'PYEOF'
import json, os, sys
d = json.loads(os.environ["STN_JSON"])
out = []
for t in d["transmitters"]:
    item = (f"{t['freq_mhz']:7.3f} MHz  {t['distance_km']:5.0f} km "
            f"{t['bearing_deg']:3.0f}°  {t['band']}  [{t['association']}]")
    state = "on" if t["in_prime_range"] else "off"
    out += [t["id"], item, state]
sys.stdout.write("\0".join(out))
sys.stdout.write("\0" if out else "")
PYEOF
)

if [[ ${#CHECK_ARGS[@]} -eq 0 ]]; then
    whiptail --title "No transmitters in range" --backtitle "$BACKTITLE" \
        --msgbox "No CODAR transmitters within 3000 km of the receiver.  Check the receiver coordinates.  Aborting." \
        10 "$WIDTH"
    exit 1
fi

n_sites=$(( ${#CHECK_ARGS[@]} / 3 ))
PICKED=$(whiptail --title "Select CODAR transmitters to record" \
    --backtitle "$BACKTITLE" --separate-output \
    --checklist "Pre-checked sites are in the prime one-hop window (200–2000 km).
Sites in the same band are grouped into one radiod channel automatically." \
    "$HEIGHT" "$WIDTH" "$LIST_HEIGHT" \
    "${CHECK_ARGS[@]}" \
    3>&1 1>&2 2>&3) || exit 0

# --separate-output gives one TAG per line.
mapfile -t PICKED_IDS <<< "$PICKED"
# Drop empty lines (no selection).
TMP=(); for x in "${PICKED_IDS[@]}"; do [[ -n "$x" ]] && TMP+=("$x"); done
PICKED_IDS=("${TMP[@]}")
if [[ ${#PICKED_IDS[@]} -eq 0 ]]; then
    whiptail --title "Nothing selected" --backtitle "$BACKTITLE" \
        --msgbox "No transmitters selected; config unchanged." 10 "$WIDTH"
    exit 0
fi

# -------- 4. build payload + apply ------------------------------------
# Group the chosen sites by band into one [[radiod]] block each (a single
# radiod IQ channel covers one band).  All blocks share the operator's
# radiod status; channel_name is derived per band.  config apply does the
# schema validation; stations.to_tx_block does the MHz→Hz units bridge.
PAYLOAD=$(STN_JSON="$STN_JSON" CALL="$CALL" GRID="$GRID" \
          RX_LAT="$RX_LAT" RX_LON="$RX_LON" STATUS="$RADIOD_STATUS" \
          PICKED="${PICKED_IDS[*]}" python3 <<'PYEOF'
import json, os, re
d = json.loads(os.environ["STN_JSON"])
by_id = {t["id"]: t for t in d["transmitters"]}
picked = [p for p in os.environ["PICKED"].split() if p in by_id]

def chan_name(band: str) -> str:
    m = re.search(r"([\d.]+)", band)
    return f"codar-{m.group(1)}mhz" if m else "codar"

# band label -> list of transmitter blocks
bands: dict[str, list] = {}
for pid in picked:
    t = by_id[pid]
    bands.setdefault(t["band"], []).append({
        "id": t["id"],
        "center_freq_hz": t["center_freq_hz"],
        "sweep_rate_hz_per_s": t["sweep_rate_hz_per_s"],
        "sweep_bw_hz": int(round(t["sweep_bw_hz"])),
        "sweep_repetition_hz": t["sweep_repetition_hz"],
        "tx_lat_deg": t["tx_lat_deg"],
        "tx_lon_deg": t["tx_lon_deg"],
    })

radiod = []
for band, txs in sorted(bands.items()):
    radiod.append({
        "status": os.environ["STATUS"],
        "channel_name": chan_name(band),
        "transmitter": txs,
    })

payload = {
    "station": {
        "callsign": os.environ["CALL"],
        "grid_square": os.environ["GRID"],
        "receiver_lat": float(os.environ["RX_LAT"]),
        "receiver_lon": float(os.environ["RX_LON"]),
    },
    "radiod": radiod,
}
print(json.dumps(payload))
PYEOF
)

if ! printf '%s' "$PAYLOAD" | \
        "$CODAR" config apply --json - ${CONFIG_PATH:+--config "$CONFIG_PATH"}; then
    whiptail --title "config apply failed" --backtitle "$BACKTITLE" \
        --msgbox "codar-sounder config apply rejected the input.  See stderr for details.  Existing config was not modified." \
        12 "$WIDTH"
    exit 1
fi

n_bands=$(printf '%s' "$PAYLOAD" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["radiod"]))')
whiptail --title "Configuration written" --backtitle "$BACKTITLE" \
    --msgbox "Wrote ${#PICKED_IDS[@]} transmitter(s) across ${n_bands} band/channel(s).

Next:
  codar-sounder validate --json
  sudo systemctl start codar-sounder@${RADIOD_STATUS%%-status.local}.service" \
    14 "$WIDTH"
exit 0
