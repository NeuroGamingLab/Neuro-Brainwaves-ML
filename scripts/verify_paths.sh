#!/usr/bin/env bash
# Validate scripts/ layout and targets referenced by launch wrappers.
# Run from anywhere: bash scripts/verify_paths.sh  OR  ./scripts/verify_paths.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NB_AI="$PROJECT_ROOT/neuro_brainwave_ai_project"

fail=0
err() { echo "ERROR: $*"; fail=1; }

EXPECTED_SCRIPTS=(
  setup_env.sh
  setup_brain_agents.sh
  run_streamlit_app.sh
  run_consumer.sh
  run_producer.sh
  cleanup.sh
  test_connection.sh
  shutdown_neurosity.sh
  launch_dashboard.sh
  launch_dynamic_generator.sh
  launch_enhanced_generator.sh
  verify_paths.sh
)

echo "Project root: $PROJECT_ROOT"
echo "Checking scripts and targets..."

for f in "${EXPECTED_SCRIPTS[@]}"; do
  if [[ ! -f "$SCRIPT_DIR/$f" ]]; then
    err "missing script: scripts/$f"
  fi
done

[[ -f "$PROJECT_ROOT/brain_signal_streamlit_app.py" ]] || err "missing: brain_signal_streamlit_app.py (run_streamlit_app.sh)"
[[ -f "$PROJECT_ROOT/brain_signal_producer.py" ]] || err "missing: brain_signal_producer.py (test_connection, agents)"
[[ -f "$PROJECT_ROOT/simple_brain_consumer.py" ]] || err "missing: simple_brain_consumer.py (test_connection, agents)"
[[ -f "$NB_AI/neuro_brainwave_dashboard.py" ]] || err "missing: neuro_brainwave_ai_project/neuro_brainwave_dashboard.py (launch_dashboard.sh)"
[[ -f "$NB_AI/dynamic_data_generator.py" ]] || err "missing: neuro_brainwave_ai_project/dynamic_data_generator.py (launch_dynamic_generator.sh)"
[[ -f "$NB_AI/enhanced_data_generator.py" ]] || err "missing: neuro_brainwave_ai_project/enhanced_data_generator.py (launch_enhanced_generator.sh)"

echo ""
echo "bash -n (syntax) on all scripts..."
shopt -s nullglob
for s in "$SCRIPT_DIR"/*.sh; do
  if ! bash -n "$s"; then
    err "bash -n failed: $s"
  fi
done
shopt -u nullglob

if (( fail )); then
  echo ""
  echo "verify_paths: FAILED"
  exit 1
fi

echo ""
echo "verify_paths: OK (all expected files present, bash -n passed)"
