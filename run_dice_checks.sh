#!/usr/bin/env bash
# Run from the existing lab in its working Python environment.
# Examples: bash run_dice_checks.sh smoke 0
#           bash run_dice_checks.sh both_shifted 1
set -uo pipefail
cd -- "$(dirname -- "$0")" || exit 2
cw2_python="${CW2_PYTHON:-python}"
cw2_selection="${1:-smoke}"
cw2_seed="${2:-0}"
"$cw2_python" check_settled_robustness.py --verify || exit 2
mkdir -p settled_robustness_logs
cw2_console="settled_robustness_logs/console_$(date +%Y%m%d_%H%M%S)_${RANDOM}.txt"
case "$cw2_selection" in
  screen|smoke|full) cw2_args=(--suite "$cw2_selection");;
  *) cw2_args=(--case "$cw2_selection");;
esac
"$cw2_python" -u check_settled_robustness.py "${cw2_args[@]}" --seed "$cw2_seed" 2>&1 | tee "$cw2_console"
cw2_result=${PIPESTATUS[0]}
"$cw2_python" - "$cw2_console" <<'PY'
from pathlib import Path
import re
import subprocess
import sys
text = Path(sys.argv[1]).read_text(errors='replace')
match = re.search(r'^\[SUITE\] summary: (.+)$', text, re.MULTILINE)
if match:
    raise SystemExit(subprocess.call([sys.executable, 'pack_settled_results.py',
                                     str(Path(match[1]).parent)]))
print('No suite started; see the console log above.')
raise SystemExit(2)
PY
cw2_pack_result=$?
if [ "$cw2_pack_result" -ne 0 ]; then
  exit "$cw2_pack_result"
fi
exit "$cw2_result"
