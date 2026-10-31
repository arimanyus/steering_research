# Asha hand-check: stock_extract (untouched Qwen3.5-4B, memory_rewrite, 23 items)
Status: NOT DONE - input file missing. No labels were assigned, no counts computed.

## What exists in /workspace/steer-spine/ (checked 2026-10-29 22:27 IST)
lib.py, lib2.py, split.json, stageA.py, stageA.log, stageB.py, stageC.py, __pycache__.
Missing: stock_extract.json, extract_acts.pt, stock_eval_turns.jsonl (searched whole filesystem; none found).

## Why the file is missing
- stageA.py writes stock_extract.json only after run_dialogues() returns. That function runs a final-turn
  delta-logprob pass (final_dlp=True) after the 4 generation turns.
- stageA.log's last line is "turn 3 done 78s" at 22:20:21. There is no "saved acts" line and no traceback in the log.
- No stageA.py process is running now. It most likely died or was killed in the post-turn-3 pass, before
  json.dump. The box shows two other ~10 GB python runs (run.py lyra/stack/stack_random), which may mean memory pressure. This is inference, not confirmed.

## Truncation finding (from code and log only, no replies read)
- lib.py: MAXNEW = int(os.environ.get("MAXNEW", "32")); gen() defaults to max_new=MAXNEW, and run_dialogues calls gen() with no override.
  Unless MAXNEW was set in the env for the stageA run (unknown), every reply was capped at 32 new tokens. That would make most replies
  length-capped, and it makes the labels unreliable.
- Turn 0/1/2 took 450/421/362s and turn 3 took 78s on the same 23-item batch. Equal 32-token caps would give similar times,
  so the 78s is not explained by the cap alone. It is consistent with early EOS or very short final replies, or with a different token cap.
  Cannot tell which without the replies.
- Turn-3 labels moved 22 hold/1 fold -> 13 hold/10 fold in the same run. That jump alongside a 5x speed-up is a red flag for a labeler/length artifact.
  Unverified.

## Needed to proceed
Rerun stageA (or a minimal dump) so the per-turn records are written BEFORE the final_dlp pass, with:
id, turn, reply, label, new_token_count, hit_cap flag, and the MAXNEW value used. Then I hand-check.
