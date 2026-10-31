#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python
ST=/workspace/steer-spine/QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
guard() { # abort if any model process is alive
  if pgrep -f "tbL_venv/bin/python (step|run)" >/dev/null; then S "ABORT chain3: model process alive before launching $1: $(pgrep -af 'tbL_venv/bin/python (step|run)' | tr '\n' ';')"; exit 1; fi; }
while kill -0 716516 2>/dev/null; do sleep 5; done
S "step1 stack (pid 716516) EXITED. $(tail -1 step1_stack.log)"
S "step1 summary (auto labels; hand-check pending): $(python3 summ_arm.py arm_stack_fp_turns.jsonl | head -12 | tr '\n' '|')"
guard A
S "(A) start: l23rand_s0..s9 (seed 3000+s) tbL harness MAXNEW 512, torch $($PY -c 'import torch;print(torch.__version__)')"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 $PY run_l23rand.py $(for s in $(seq 0 9); do echo -n "l23rand_s$s "; done) > run_l23rand_A.out 2>&1
S "(A) finished exit=$?: $(grep '^SEED' run_l23rand_A.out | tr '\n' ';')"
cd /workspace/steer-spine; guard L23FP
S "(A2) start: step1_l23.py l23 (fp 6-turn, refusal_dir_L23.pt) -> arm_l23_fp_turns.jsonl"
$PY step1_l23.py l23 > step1_l23.log 2>&1
S "(A2) finished exit=$?: $(python3 summ_arm.py arm_l23_fp_turns.jsonl | head -12 | tr '\n' '|')"
guard B
S "(B) start: step2.py sustained pilot mr-401..430 (NT=4)"
NT=4 $PY step2.py > step2.log 2>&1
S "(B) finished exit=$?: $(tail -2 step2.log | tr '\n' '|')"
guard C
S "(C) start: l23rand_s10..s29"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 $PY run_l23rand.py $(for s in $(seq 10 29); do echo -n "l23rand_s$s "; done) > run_l23rand_C.out 2>&1
S "(C) finished exit=$?: $(grep '^SEED' run_l23rand_C.out | tr '\n' ';')"
cd /workspace/steer-spine; guard SN
S "(D) start: step1_stock_neutral.py (stock fp neutral replay)"
rm -f step1_stock_neutral.log.bak; mv step1_stock_neutral.log step1_stock_neutral.log.bak 2>/dev/null
$PY step1_stock_neutral.py > step1_stock_neutral.log 2>&1
S "(D) finished exit=$?"
S "CHAIN3 DONE (random-stack 1c dropped per parent; fp randstack not run)"
