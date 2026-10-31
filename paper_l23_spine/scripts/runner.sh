#!/bin/bash
# restarts stageD (resumable per arm) if it is OOM-killed; waits for free RAM first
cd /workspace/steer-spine
for try in $(seq 1 40); do
  while [ $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) -lt 6500 ]; do sleep 30; done
  echo "$(date +%T) launch try $try" >> runner.log
  NT=4 BS=8 MAXNEW=32 /workspace/tbL_venv/bin/python stageD.py >> stageD.log 2>&1
  rc=$?; echo "$(date +%T) exit $rc" >> runner.log
  [ $rc -eq 0 ] && break
  sleep 60
done
