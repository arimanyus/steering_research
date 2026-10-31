#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python
while pgrep -f "run1b.py noedit" >/dev/null; do sleep 10; done
echo "$(date +%H:%M) IST Step 1b finished (see tbL_noedit/l23only.jsonl). Starting Step 1: step1.py stack" >> QUEUE_STATUS.md
pgrep -f "[p]ython.*(step|run)" && echo "WARN other proc" >> QUEUE_STATUS.md
$PY step1.py stack > step1_stack.log 2>&1
N=$(python3 -c "
import json
print(sum(1 for l in open('/workspace/steer-spine/arm_stack_fp_turns.jsonl') if (r:=json.loads(l)) and r['turn']==5 and r['auto_label']=='fold'))")
T4=$(python3 -c "
import json
print(sum(1 for l in open('/workspace/steer-spine/arm_stack_fp_turns.jsonl') if (r:=json.loads(l)) and r['turn']==4 and r['auto_label']=='fold'))")
echo "$(date +%H:%M) IST Step 1 stack done: auto folds t4=$T4/25 t5=$N/25 (stock t4 21/25, t5 1/25 hand)" >> QUEUE_STATUS.md
echo "$(date +%H:%M) IST Step 1c start: tbL random-stack s10..s29 on tb items (TBL_THREADS=4)" >> QUEUE_STATUS.md
cd /workspace/steer-withheld/tbL
SN=""; for s in $(seq 10 29); do SN="$SN stack_random_s$s"; done
TBL_THREADS=4 $PY run1b.py $SN > run1c.out 2>&1
echo "$(date +%H:%M) IST Step 1c finished" >> /workspace/steer-spine/QUEUE_STATUS.md
cd /workspace/steer-spine
$PY step1_stock_neutral.py > step1_stock_neutral.log 2>&1
echo "$(date +%H:%M) IST stock fp neutral replay finished; launching Step 2" >> QUEUE_STATUS.md
$PY step2.py > step2.log 2>&1
echo "$(date +%H:%M) IST Step 2 finished" >> QUEUE_STATUS.md
if [ "$N" -ge 7 ]; then
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo "$(date +%H:%M) IST fp randstack seed $s start" >> QUEUE_STATUS.md
    $PY step1.py randstack $s > step1_randstack_s$s.log 2>&1
  done
  echo "$(date +%H:%M) IST fp randstack s0..s9 finished" >> QUEUE_STATUS.md
else
  echo "$(date +%H:%M) IST t5 folds <7 -> fp randstack seeds skipped per rule" >> QUEUE_STATUS.md
fi
