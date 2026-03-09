#!/usr/bin/env bash
set  -euo pipefail

# Keep window open at the end of execution
pause_always() {
	local exit_status=$?
	echo
	if [[ -r /dev/tty ]]; then
		read -r -p "Press enter to close the window..."
	else
		echo "/dev/tty not available, press crtl+C to close or wait 60s..."
		sleep 60
	fi
	exit "$exit_status"
}
trap pause_always EXIT

# Select path relative to this file
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PY_EXE="$VENV_DIR/Scripts/python.exe"

# Parameters
CAB="cab1"
TEST_STEP="Move TBC to position B"
CONFIG="predict"

# Ausführen
echo "Starte run.py mit:"
echo "--cab \"$CAB\" --test_step \"$TEST_STEP\" --config \"$CONFIG\""
echo

PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
"$PY_EXE" "$SCRIPT_DIR/run.py" \
	--cab "$CAB" \
	--test_step "$TEST_STEP" \
	--config "$CONFIG"