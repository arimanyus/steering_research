#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python
ST=/workspace/steer-spine/QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
guard() { if pgrep -f "tbL_venv/bin/python (step|run)" >/dev/null; then S "ABORT chain5: model process alive before launching $1: $(pgrep -af 'tbL_venv/bin/python (step|run)' | tr '\n' ';')"; exit 1; fi; }

S "chain6 start (A already stopped by parent at 08:50; s0,s1 kept)"
guard D
S "(D) start: step1_l23.py stock = full 6-turn fp, NO edit -> arm_stock_fp_turns.jsonl (+acts); rows log suite path+sha256"
$PY step1_l23.py stock > step1_stock_full.log 2>&1
S "(D) finished exit=$?: $(python3 summ_arm.py arm_stock_fp_turns.jsonl 2>&1 | grep -E '^(baseline|t[456]|neutral|factual)' | tr '\n' '|')"
for s in 0 1 2 3 4 5 6 7 8 9; do
  guard E$s
  S "(E) randstack fp seed $s start (seed 2000+s)"
  $PY step1.py randstack $s > step1_randstack_s$s.log 2>&1
  S "(E) seed $s finished exit=$?: $(python3 summ_arm.py arm_randstack_fp_s$s.jsonl 2>&1 | grep -E '^t[456]:|dlp|factual' | tr '\n' '|')"
done
guard Arest
S "(Arest) start: l23rand_s2..s9 (s0,s1 files kept)"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 $PY run_l23rand.py $(for s in $(seq 2 9); do echo -n "l23rand_s$s "; done) > run_l23rand_Arest.out 2>&1
S "(Arest) finished exit=$?: $(grep '^SEED' run_l23rand_Arest.out | tr '\n' ';')"
cd /workspace/steer-spine
guard A2
S "(A2) start: step1_l23.py l23"
$PY step1_l23.py l23 > step1_l23.log 2>&1
S "(A2) finished exit=$?: $(python3 summ_arm.py arm_l23_fp_turns.jsonl | head -12 | tr '\n' '|')"
guard B
S "(B) start: step2.py (NT=4)"
NT=4 $PY step2.py > step2.log 2>&1
S "(B) finished exit=$?: $(tail -2 step2.log | tr '\n' '|')"
guard C
S "(C) start: l23rand_s10..s29"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 $PY run_l23rand.py $(for s in $(seq 10 29); do echo -n "l23rand_s$s "; done) > run_l23rand_C.out 2>&1
S "(C) finished exit=$?: $(grep '^SEED' run_l23rand_C.out | tr '\n' ';')"
S "CHAIN6 DONE"
