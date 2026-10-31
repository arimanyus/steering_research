#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python; ST=QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
while kill -0 969148 2>/dev/null; do sleep 30; done
S "(B) step2 exited: $(tail -2 step2.log | tr '\n' '|')"
for s in $(seq 0 19); do
  if pgrep -f "tbL_venv/bin/python" >/dev/null; then S "ABORT chain8: model alive before E1 s$s"; exit 1; fi
  S "(E1) l23rand fp seed $s start (seed 3000+s)"
  NT=4 $PY step1_l23.py l23rand $s > step1_l23rand_s$s.log 2>&1
  S "(E1) seed $s exit=$?: $(python3 summ_arm.py arm_l23rand_fp_s$s.jsonl 2>&1 | grep -E '^t[2-6]:' | tr '\n' '|')"
done
S "(E1) DONE 20 seeds"
if pgrep -f "tbL_venv/bin/python" >/dev/null; then S "ABORT chain8 before F"; exit 1; fi
S "(F) start Leila fit_concept_dir.py"
cd /workspace/neuron-bench && $PY fit_concept_dir.py > fit_concept_dir.log 2>&1
cd /workspace/steer-spine; S "(F) finished exit=$?; box free for Maya"
