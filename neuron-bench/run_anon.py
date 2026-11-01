#!/usr/bin/env python3
"""Runs bench.py unchanged (same CLI arguments) after patching AutoModelForCausalLM.from_pretrained to copy every weight out of the
safetensors mmap into anonymous memory. On this VM the memory balloon reclaims file-backed pages, so mmapped weights get evicted and
re-read from disk during inference (~10x slowdown); anonymous pages cannot be reclaimed (no swap). The copy is bit-exact, so bench.py,
its arguments and its config hashes are unchanged. The copy is throttled on MemAvailable so the balloon has time to deflate.
Usage: python run_anon.py <bench.py arguments>"""
import os, sys, time, runpy
import transformers

def mem_available_gb(): return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20

_orig = transformers.AutoModelForCausalLM.from_pretrained
def from_pretrained(*args, **kwargs):
    import torch
    m = _orig(*args, **kwargs); t0 = time.time(); waited = 0.0
    with torch.no_grad():
        for p in list(m.parameters()) + list(m.buffers()):
            while mem_available_gb() < 1.5 and waited < 600: time.sleep(0.5); waited += 0.5
            p.data = p.data.clone()
    print(f"[run_anon] weights copied to anonymous memory in {time.time() - t0:.0f}s (waited {waited:.0f}s for MemAvailable)", flush=True)
    return m
transformers.AutoModelForCausalLM.from_pretrained = from_pretrained

sys.argv = [f"{os.path.dirname(os.path.abspath(__file__))}/bench.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
