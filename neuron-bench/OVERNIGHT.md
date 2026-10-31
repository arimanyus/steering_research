# Overnight driver log

## 2026-10-31 02:06 IST: E1 l23rand fp seed 1 row check
- QUEUE_STATUS row (01:43 IST): `(E1) seed 1 exit=0: t2: n=25 hold=25 hedge=0 fold=0 other=0 capped=0|t3: n=25 hold=25 hedge=0 fold=0 other=0 capped=0|t4: n=25 hold=2 hedge=0 fold=23 other=0 capped=0|t5: n=25 hold=24 hedge=0 fold=1 other=0 capped=0|t6: n=25 hold=25 hedge=0 fold=0 other=0 capped=0|`
- File: /workspace/steer-spine/arm_l23rand_fp_s1.jsonl, 203 rows, 28 unique ids overall.
- false_premise: 200 rows, 25 unique ids (fp-001..fp-025), no duplicate (turn,id) pairs. PASS.
- fact: 3 rows, 3 unique ids (f1, f2, f3). Same layout as s0 (203 rows / 28 unique).
- Status at check: E1 seed 2 running (pid 1034719, ~10.9 GB RSS); Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy).

## 2026-10-31 02:38 IST: status check
- E1 seed 2 exit=0 at 02:20; seed 3 running (pid 1051713, ~10.4 GB RSS). Seed-1 row check already done (PASS); no repeat.
- 02:20 refit_L23 rc=1 was its own memory guard (MemAvailable 5.9 GB < 11 GB while E1 ran), not a crash; Kenji re-slotted it via after_chain_refit.sh (pid 1053038) to run after chain8 (E1 s19 + Leila fit). No post.
- Waiting for E1, (F), and (R). Neuron-bench not started (computer busy).

## 2026-10-31 03:06 IST: status check
- E1 seeds 3 and 4 exit=0 (02:43, 02:59); seed 5 running (pid 1068449, ~8.1 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run (refit waits on after_chain_refit.sh). Neuron-bench not started (computer busy). No post.

## 2026-10-31 03:31 IST: status check
- E1 seed 5 exit=0 (03:24); seed 6 running (pid 1077473, ~9.9 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 04:05 IST: status check
- E1 seed 6 exit=0 (03:51); seed 7 running (pid 1087598, ~10.0 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 04:34 IST: status check
- E1 seed 7 exit=0 (04:16); seed 8 running (pid 1099105, ~10.6 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 05:01 IST: status check
- E1 seed 8 exit=0 (04:36); seed 9 running (pid 1106931, ~10.6 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 05:30 IST: status check
- E1 seeds 9 and 10 exit=0 (05:05, 05:18); seed 11 running (pid 1125051, ~10.4 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 06:01 IST: status check
- E1 seeds 11 and 12 exit=0 (05:31, 05:54); seed 13 running (pid 1140764, ~5.7 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 06:35 IST: status check
- E1 seed 13 exit=0 (06:25); seed 14 running (pid 1151767, ~10.6 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 07:09 IST: status check
- E1 seeds 14 and 15 exit=0 (06:41, 07:02); seed 16 running (pid 1169394, ~8.9 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 07:38 IST: status check
- E1 seed 16 exit=0 (07:29); seed 17 running (pid 1179977, ~10.8 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 08:07 IST: status check
- E1 seed 17 exit=0 (07:43); seed 18 running (pid 1188108, ~10.2 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run. Neuron-bench not started (computer busy). No post.

## 2026-10-31 08:37 IST: status check
- E1 seed 18 exit=0 (08:12); seed 19 (last E1 seed) running (pid 1197318, ~10.6 GB RSS). Seed-1 row check already done (PASS); no repeat.
- Leila fit (F) and refit (R) not yet run (re-slotted after chain8). Neuron-bench not started (computer busy). No post.

## 2026-10-31 09:11 IST: status check
- E1 DONE 20 seeds (08:38). Refit (R) finished rc=0 (08:40). Leila fit (F2) finished rc=0 pt_files=4 (08:46). All prerequisites exited; no python process over 2 GB.
- MemAvailable = 2741980 kB (~2.6 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 09:43 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6610736 kB (~6.3 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 10:11 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 5492972 kB (~5.2 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 10:45 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 7100252 kB (~6.8 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 11:13 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6475896 kB (~6.2 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 11:15 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6475896 kB (~6.2 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 11:40 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6299688 kB (~6.0 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. Not the 12:00 check, so no post.

## 2026-10-31 12:06 IST: status check (12:00 IST check)
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6198060 kB (~5.9 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. This is the 12:00 check with memory still below 11 GB, so one room post (screen blocked) handed to the parent.

## 2026-10-31 12:36 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2945628 kB (~2.8 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 13:06 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2537100 kB (~2.4 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 13:13 IST: status check
- No python process over 2 GB; E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6407660 kB (~6.1 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 13:27 IST: status check
- Python procs over 2 GB: 0.
- MemAvailable = 2559484 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 13:42 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2782972 kB (~2.7 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 14:10 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 3073352 kB (~2.9 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 14:26 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2391156 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 14:37 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 6226192 kB (~5.9 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 15:06 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2546564 kB (~2.4 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 15:06 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2566116 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 15:34 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2882428 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 15:39 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2805120 kB (~2.7 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 16:02 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2849488 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 16:09 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2817708 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 16:35 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2854288 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 16:38 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2828212 kB, below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.

## 2026-10-31 17:06 IST: status check
- Python procs over 2 GB: 0. E1, refit (R) and Leila fit (F2) all exited earlier.
- MemAvailable = 2852124 kB (~2.7 GB), below the 11 GB gate. Neuron-bench not started; waiting, nothing killed. The 12:00 screen-blocked post was already handed off at 12:06, so no further post.
