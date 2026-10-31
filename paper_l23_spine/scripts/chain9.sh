#!/bin/bash
ST=/workspace/steer-spine/QUEUE_STATUS.md
S() { echo "$(date +%H:%M) IST $*" >> $ST; }
while kill -0 999334 2>/dev/null; do sleep 60; done
if pgrep -f "tbL_venv/bin/python" >/dev/null; then S "ABORT chain9: model alive"; exit 1; fi
S "(C') start: l23rand tool-bait s10..s19"
cd /workspace/steer-withheld/tbL
TBL_THREADS=4 /workspace/tbL_venv/bin/python run_l23rand.py $(for s in $(seq 10 19); do echo -n "l23rand_s$s "; done) > run_l23rand_C10.out 2>&1
S "(C') finished exit=$?: $(grep '^SEED' run_l23rand_C10.out | tr '\n' ';')"
