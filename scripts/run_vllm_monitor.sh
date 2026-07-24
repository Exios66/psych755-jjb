#!/usr/bin/env bash
#
# Tail / status helper for scripts/run_vllm.sh logs.
#
# Usage:
#   ./scripts/run_vllm_monitor.sh
#   ./scripts/run_vllm_monitor.sh --status
#   ./scripts/run_vllm_monitor.sh --list
#   ./scripts/run_vllm_monitor.sh --poll <PID>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logging"

mode="${1:-tail}"

latest_log() {
  if [[ -f "${LOG_DIR}/latest_vllm.logpath" ]]; then
    cat "${LOG_DIR}/latest_vllm.logpath"
    return
  fi
  ls -1t "${LOG_DIR}"/*_ca_vllm_*.log 2>/dev/null | head -1 || true
}

case "$mode" in
  --list)
    mkdir -p "$LOG_DIR"
    printf "%-10s %-8s %s\n" "SIZE" "MTIME" "LOG"
    find "$LOG_DIR" -maxdepth 1 -name '*_ca_vllm_*.log' -printf '%s %TY-%Tm-%Td_%TH:%TM %p\n' 2>/dev/null \
      | sort -k2 \
      | while read -r size mtime path; do
          printf "%-10s %-8s %s\n" "$size" "$mtime" "$(basename "$path")"
        done
    ;;
  --status)
    LOG="$(latest_log)"
    PID_FILE="${LOG_DIR}/latest_vllm.pid"
    echo "log: ${LOG:-<none>}"
    if [[ -f "$PID_FILE" ]]; then
      PID="$(cat "$PID_FILE")"
      if kill -0 "$PID" 2>/dev/null; then
        echo "pid: $PID (running)"
      else
        echo "pid: $PID (not running)"
      fi
    else
      echo "pid: <none>"
    fi
    if [[ -n "${LOG:-}" && -f "$LOG" ]]; then
      echo "---- last 20 lines ----"
      tail -n 20 "$LOG"
    fi
    ;;
  --poll)
    PID="${2:-}"
    if [[ -z "$PID" ]]; then
      echo "Usage: $0 --poll <PID>" >&2
      exit 1
    fi
    while kill -0 "$PID" 2>/dev/null; do
      echo "[$(date +%H:%M:%S)] PID $PID still running..."
      sleep 30
    done
    echo "PID $PID finished"
    LOG="$(latest_log)"
    [[ -n "${LOG:-}" && -f "$LOG" ]] && tail -n 40 "$LOG"
    ;;
  tail|--tail|"")
    LOG="$(latest_log)"
    if [[ -z "${LOG:-}" || ! -f "$LOG" ]]; then
      echo "No vLLM logs found under $LOG_DIR" >&2
      exit 1
    fi
    echo "Tailing $LOG"
    tail -f "$LOG"
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    echo "Use: --list | --status | --poll <PID> | (default tail)" >&2
    exit 1
    ;;
esac
