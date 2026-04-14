#!/usr/bin/env bash
# Benchmark NumKong SIMD kernels vs Lance's existing implementations.
#
# Usage:
#   ./scripts/bench_numkong.sh              # all metrics
#   ./scripts/bench_numkong.sh dot          # dot only
#   ./scripts/bench_numkong.sh dot l2       # dot and l2
#   ./scripts/bench_numkong.sh --filter f16 # all metrics, f16 only

set -euo pipefail

METRICS=()
FILTER=""
TARGET_TIME="${TARGET_TIME:-5}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter) FILTER="$2"; shift 2 ;;
        dot|l2|cosine) METRICS+=("$1"); shift ;;
        *) echo "Usage: $0 [--filter PATTERN] [dot] [l2] [cosine]"; exit 1 ;;
    esac
done

if [[ ${#METRICS[@]} -eq 0 ]]; then
    METRICS=(dot l2 cosine)
fi

FILTER_ARG=""
if [[ -n "$FILTER" ]]; then
    FILTER_ARG="-- $FILTER"
fi

export TARGET_TIME

for metric in "${METRICS[@]}"; do
    echo "========================================"
    echo " ${metric^^}: baseline (no numkong)"
    echo "========================================"
    cargo bench -p lance-linalg --bench "$metric" -- --save-baseline lance $FILTER_ARG

    echo ""
    echo "========================================"
    echo " ${metric^^}: numkong"
    echo "========================================"
    cargo bench -p lance-linalg --bench "$metric" --features numkong -- --baseline lance $FILTER_ARG

    echo ""
done
