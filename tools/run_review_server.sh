#!/usr/bin/env bash
# Supervised BabyTalk review server.
#
# Keeps http://127.0.0.1:8765 up across Cursor agent shell deaths and crashes.
# Reuses an already-healthy listener instead of killing/restarting it.
#
# Usage:
#   tools/run_review_server.sh              # ensure running (detached)
#   tools/run_review_server.sh --status     # print health + pids
#   tools/run_review_server.sh --stop       # stop supervisor + server
#   tools/run_review_server.sh --foreground # restart loop in this terminal
#   tools/run_review_server.sh --restart    # stop then start fresh
#   tools/run_review_server.sh [path] [port]  # passed through to review_server.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
PORT="${BABYTALK_REVIEW_PORT:-8765}"
RUN_DIR="$TOOLS/.run"
PIDFILE="$RUN_DIR/supervisor.pid"
LOGFILE="$RUN_DIR/review_server.log"
SERVER_PY="$TOOLS/review_server.py"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

if [[ -x "$TOOLS/.venv/bin/python" ]]; then
  PYTHON="$TOOLS/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

mkdir -p "$RUN_DIR"

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

health_ok() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    return 0
  fi
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:${PORT}/" 2>/dev/null || true)"
  [[ "$code" == "200" ]]
}

listener_pids() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

supervisor_alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

print_status() {
  if health_ok; then
    echo "review UI: healthy  http://127.0.0.1:${PORT}/"
  else
    echo "review UI: down     http://127.0.0.1:${PORT}/"
  fi
  if supervisor_alive; then
    echo "supervisor: running pid=$(cat "$PIDFILE")"
  else
    echo "supervisor: not running"
  fi
  local listeners
  listeners="$(listener_pids | tr '\n' ' ')"
  if [[ -n "${listeners// /}" ]]; then
    echo "listeners:  ${listeners}"
  else
    echo "listeners:  none"
  fi
  echo "log:        $LOGFILE"
}

stop_all() {
  if supervisor_alive; then
    local spid
    spid="$(cat "$PIDFILE")"
    # Kill process group (supervisor is session leader after double-fork).
    kill -TERM "-$spid" 2>/dev/null || kill -TERM "$spid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$spid" 2>/dev/null || break
      sleep 0.2
    done
    kill -KILL "-$spid" 2>/dev/null || kill -KILL "$spid" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  local pids
  pids="$(listener_pids)"
  if [[ -n "$pids" ]]; then
    # Only free :PORT — do not sweep unrelated user processes.
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    sleep 0.3
    pids="$(listener_pids)"
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $pids 2>/dev/null || true
    fi
  fi
}

detach_supervisor() {
  # Double-fork + setsid so we leave the Cursor agent process tree.
  # Ephemeral agent shells often SIGTERM their descendants on exit (exit 143);
  # a new session with PPID 1 survives that.
  local PASS_JSON
  if [[ ${#PASS_ARGS[@]} -eq 0 ]]; then
    PASS_JSON='[]'
  else
    PASS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' -- "${PASS_ARGS[@]}")"
  fi
  PORT="$PORT" ROOT="$ROOT" SELF="$SELF" LOGFILE="$LOGFILE" PASS_JSON="$PASS_JSON" \
    "$PYTHON" - <<'PY'
import json, os, sys

script = os.environ["SELF"]
logfile = os.environ["LOGFILE"]
root = os.environ["ROOT"]
pass_args = json.loads(os.environ.get("PASS_JSON") or "[]")

if os.fork() > 0:
    raise SystemExit(0)
os.setsid()
if os.fork() > 0:
    raise SystemExit(0)

os.chdir(root)
os.environ["PYTHONUNBUFFERED"] = "1"
with open(logfile, "a", encoding="utf-8") as log:
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
# stdin from /dev/null
devnull = os.open("/dev/null", os.O_RDONLY)
os.dup2(devnull, 0)
os.close(devnull)

os.execv("/bin/bash", ["bash", script, "--foreground", *pass_args])
PY
}

# Parse flags; remaining args go to review_server.py
MODE="ensure"
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --status) MODE="status"; shift ;;
    --stop) MODE="stop"; shift ;;
    --restart) MODE="restart"; shift ;;
    --foreground) MODE="foreground"; shift ;;
    --) shift; PASS_ARGS+=("$@"); break ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

# Allow trailing numeric arg / BABYTALK_REVIEW_PORT override via argv port.
if [[ ${#PASS_ARGS[@]} -ge 1 && "${PASS_ARGS[-1]}" =~ ^[0-9]+$ ]]; then
  PORT="${PASS_ARGS[-1]}"
elif [[ ${#PASS_ARGS[@]} -ge 2 && "${PASS_ARGS[1]}" =~ ^[0-9]+$ ]]; then
  PORT="${PASS_ARGS[1]}"
fi

case "$MODE" in
  status)
    print_status
    exit 0
    ;;
  stop)
    stop_all
    echo "stopped review server on :${PORT}"
    exit 0
    ;;
  restart)
    stop_all
    MODE="ensure"
    ;;
esac

if [[ "$MODE" == "ensure" ]]; then
  if health_ok; then
    echo "review UI already healthy at http://127.0.0.1:${PORT}/"
    if ! supervisor_alive; then
      echo "(listener is up but unsupervised — for auto-restart: tools/run_review_server.sh --restart)"
    fi
    print_status
    exit 0
  fi
  if supervisor_alive; then
    echo "supervisor alive; waiting for health on :${PORT}..."
    for _ in $(seq 1 20); do
      if health_ok; then
        print_status
        exit 0
      fi
      sleep 0.25
    done
    echo "supervisor not recovering — restarting" >&2
    stop_all
  fi

  detach_supervisor

  for _ in $(seq 1 40); do
    if health_ok; then
      echo "started supervised review UI at http://127.0.0.1:${PORT}/"
      print_status
      exit 0
    fi
    sleep 0.25
  done
  echo "failed to become healthy within 10s — see $LOGFILE" >&2
  tail -n 40 "$LOGFILE" >&2 || true
  exit 1
fi

# --- foreground supervised loop ---
echo $$ >"$PIDFILE"
cleanup() {
  rm -f "$PIDFILE"
}
trap cleanup EXIT
trap 'cleanup; exit 0' INT TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] supervisor start port=${PORT} python=${PYTHON} pid=$$"

backoff=1
while true; do
  if health_ok; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] :${PORT} already healthy — supervising existing listener"
    backoff=1
    while health_ok; do
      sleep 2
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] health lost — will start server"
  fi

  # If something holds the port but is unhealthy, free only that listener.
  stale="$(listener_pids)"
  if [[ -n "$stale" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] clearing stale listener(s): $stale"
    # shellcheck disable=SC2086
    kill -TERM $stale 2>/dev/null || true
    sleep 0.4
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching review_server.py"
  set +e
  PYTHONUNBUFFERED=1 "$PYTHON" "$SERVER_PY" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
  rc=$?
  set -e
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] review_server exited rc=${rc}; restart in ${backoff}s"

  sleep "$backoff"
  if [[ $backoff -lt 8 ]]; then
    backoff=$((backoff * 2))
  fi
done
