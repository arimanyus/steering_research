#!/bin/bash
ST=/workspace/steer-spine/QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
while kill -0 969148 2>/dev/null; do sleep 30; done
S "(B) step2 pid 969148 exited: $(tail -2 /workspace/steer-spine/step2.log | tr '\n' '|')"
if pgrep -f "tbL_venv/bin/python" >/dev/null; then S "ABORT chain7: model alive"; exit 1; fi
S "(F) start: Leila fit_concept_dir.py"
cd /workspace/neuron-bench && /workspace/tbL_venv/bin/python fit_concept_dir.py > fit_concept_dir.log 2>&1
S "(F) finished exit=$?; next: Maya calibration+stock timing (launched by Maya). C (l23rand s10..29) deferred until neuron-bench screen."
