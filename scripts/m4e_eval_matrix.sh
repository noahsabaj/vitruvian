#!/usr/bin/env bash
# M4.5 evaluation matrix.
# 10 goals (5 from original v3 H5 for comparability, 5 from new diverse H5)
# × {sigma=0.02, sigma=0.05} × {vf-on, vf-off} = 40 runs.
# Also includes a v4 baseline with σ=0 (pure PPO replay) for direct
# comparison to the σ=0 baseline we recorded in M4.4c close.

set -u  # don't -e: we want to keep going on individual run failures

ORIG_H5="$HOME/.stable_worldmodel/g1_joystick_expert.h5"
DIVERSE_H5="$HOME/.stable_worldmodel/g1_diverse_v1.h5"
JEPA_V4="$HOME/.vitruvian/m4e_v4/best.pt"
VF_HER="$HOME/.vitruvian/m4e_v4/vf_her.pt"
POLICY="checkpoints/m1-g1-full/000043253760"

OUT_DIR=/tmp/vitruvian/m4e_eval
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
: > "$SUMMARY"   # start fresh
echo -e "scenario\tsigma\tvf\th5\tgoal_ep\tupright\tmacros\tdxy\tmean_cos\tmax_cos\tlog" >> "$SUMMARY"

run_one() {
    local label="$1" sigma="$2" vf="$3" h5="$4" ep="$5"
    local log="$OUT_DIR/${label}_sigma${sigma}_vf${vf}_ep${ep}.log"
    local vf_flag=""
    [ "$vf" = "on" ] && vf_flag="--vf-ckpt $VF_HER"
    echo "=== $label  σ=$sigma  vf=$vf  ep=$ep  h5=$(basename "$h5") ==="
    uv run python scripts/m4c_hierarchical_plan.py \
        --encoder dinov3-v4 \
        --ckpt-jepa-v4 "$JEPA_V4" \
        --policy-ckpt "$POLICY" \
        --h5 "$h5" \
        --decoder flat --warm-start-policy \
        --l1-num-samples 64 --l1-noise-sigma "$sigma" \
        --vel-cmd "0.5,0,0" \
        --goal-ep-idx "$ep" --goal-idx 0 \
        $vf_flag \
        2>&1 | tee "$log" | grep -E "macros run|Δxy|mean cos|^  macro|terminate" | tail -6
    # Extract summary row.
    local macros=$(grep -oE "macros run: [0-9]+/[0-9]+" "$log" | head -1 | awk '{print $3}')
    local upright=$(grep -oE "upright: [0-9]+" "$log" | head -1 | awk '{print $2}')
    local mean_cos=$(grep -oE "mean cos: [+\-0-9.]+" "$log" | head -1 | awk '{print $3}')
    local dxy=$(grep -oE "Δxy from start: [0-9.]+" "$log" | head -1 | awk '{print $4}')
    # max cos from the per-macro rows
    local max_cos=$(grep -oE "cos=[+\-0-9.]+" "$log" | awk -F= '{if ($2 > max) max=$2} END {printf "%+.3f", max}')
    echo -e "${label}\t${sigma}\t${vf}\t$(basename "$h5")\t${ep}\t${upright:-NA}\t${macros:-NA}\t${dxy:-NA}\t${mean_cos:-NA}\t${max_cos:-NA}\t$(basename "$log")" >> "$SUMMARY"
    echo
}

# --- original-H5 goals (comparable to prior M4.4c runs) ---
for sigma in 0.02 0.05; do
    for vf in off on; do
        for ep in 0 20 40 60 80; do
            run_one "orig" "$sigma" "$vf" "$ORIG_H5" "$ep"
        done
    done
done

# --- diverse-H5 goals ---
for sigma in 0.02 0.05; do
    for vf in off on; do
        for ep in 0 10 20 30 40; do
            run_one "diverse" "$sigma" "$vf" "$DIVERSE_H5" "$ep"
        done
    done
done

echo
echo "=== Full matrix complete. Summary: $SUMMARY ==="
column -t -s $'\t' "$SUMMARY"
