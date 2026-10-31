#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python
ST=/workspace/steer-spine/QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
guard() { if pgrep -f "tbL_venv/bin/python (step|run)" >/dev/null; then S "ABORT chain4: model process alive before launching $1: $(pgrep -af 'tbL_venv/bin/python (step|run)' | tr '\n' ';')"; exit 1; fi; }
while kill -0 728103 2>/dev/null; do sleep 5; done
S "(A) l23rand s0..s9 (pid 728103) EXITED: $(grep '^SEED' /workspace/steer-withheld/tbL/run_l23rand_A.out | tr '\n' ';') files=$(ls /workspace/steer-withheld/tbL_l23rand_s?.jsonl 2>/dev/null | wc -l)/10"
guard D
S "(D) start: step1_stock_neutral.py"
$PY step1_stock_neutral.py > step1_stock_neutral.log 2>&1
S "(D) finished exit=$? rows=$(wc -l < arm_stock_fp_neutral.jsonl)"
for s in 0 1 2 3 4 5 6 7 8 9; do
  guard E$s
  S "(E) randstack fp seed $s start (random 3-dir stack L20/L23/L25 slots, seed 2000+s, QR; unit/orthonormal so projection needs no extra norm matching)"
  $PY step1.py randstack $s > step1_randstack_s$s.log 2>&1
  S "(E) seed $s finished exit=$?: $(python3 summ_arm.py arm_randstack_fp_s$s.jsonl 2>&1 | grep -E '^t[456]:|dlp|factual' | tr '\n' '|')"
done
guard A2
S "(A2) start: step1_l23.py l23"
$PY step1_l23.py l23 > step1_l23.log 2>&1
S "(A2) finished exit=$?: $(python3 summ_arm.py arm_l23_fp_turns.jsonl | head -12 | tr '\n' '|')"
guard B
S "(B) start: step2.py sustained pilot (NT=4)"
NT=4 $PY step2.py > step2.log 2>&1
S "(B) finished exit=$?: $(tail -2 step2.log | tr '\n' '|')"
guard C
S "(C) start: l23rand_s10..s29"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 $PY run_l23rand.py $(for s in $(seq 10 29); do echo -n "l23rand_s$s "; done) > run_l23rand_C.out 2>&1
S "(C) finished exit=$?: $(grep '^SEED' run_l23rand_C.out | tr '\n' ';')"
S "CHAIN4 DONE"
