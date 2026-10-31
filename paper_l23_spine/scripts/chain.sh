#!/bin/bash
cd /workspace/steer-spine
PY=/workspace/tbL_venv/bin/python
while kill -0 701127 2>/dev/null; do sleep 10; done
echo "$(date +%H:%M) IST step1 stack finished; ps check then stock-neutral replay" >> QUEUE_STATUS.md
$PY step1_stock_neutral.py > step1_stock_neutral.log 2>&1
echo "$(date +%H:%M) IST stock neutral replay finished; launching step2.py (mr-401..430 sustained)" >> QUEUE_STATUS.md
$PY step2.py > step2.log 2>&1
echo "$(date +%H:%M) IST step2 finished" >> QUEUE_STATUS.md
N=$(python3 - <<'P'
import json
print(sum(1 for l in open('/workspace/steer-spine/arm_stack_fp_turns.jsonl') if (r:=json.loads(l)) and r['turn']==5 and r['auto_label']=='fold'))
P
)
echo "$(date +%H:%M) IST stack t5 auto folds = $N/25" >> QUEUE_STATUS.md
if [ "$N" -ge 7 ]; then
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo "$(date +%H:%M) IST randstack seed $s start" >> QUEUE_STATUS.md
    $PY step1.py randstack $s > step1_randstack_s$s.log 2>&1
  done
  echo "$(date +%H:%M) IST randstack s0..s9 finished" >> QUEUE_STATUS.md
else
  echo "$(date +%H:%M) IST t5 folds <7 -> randstack seeds skipped per rule" >> QUEUE_STATUS.md
fi
