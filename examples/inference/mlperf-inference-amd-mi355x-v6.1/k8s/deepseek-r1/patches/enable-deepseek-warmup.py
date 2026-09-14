#!/usr/bin/env python3
# enable-deepseek-warmup.py - inject the 'deepseek-r1' entry into
# harness_llm/backends/common/constants.py WarmUp.ENCODED_SAMPLES.
#
# Our deepseek_sglang_v0.5.15 image already carries AMD's full warmup code path
# (run_warmup/warmup_generate in distributed_async_server.py, SGLang build 20260722),
# but its constants.py has no 'deepseek-r1' encoded-sample entry, so warmup self-disables
# (enable_warmup AND warmup_sample is not None → False). This adds the entry AMD shipped in
# dsr_code6_1 (the "warmup" delivery), flipping the warmup path on for the deepseek Server.
#
# Idempotent: does nothing if a deepseek-r1 entry already exists.
import io, os, re, sys

CONST = "/lab-mlperf-inference/code/harness_llm/backends/common/constants.py"

# AMD's DSR-tokenizer "Lorem ipsum" warmup prompt (constants.py:46-50 in dsr_code6_1)
DSR_SAMPLE = ("[83240, 55848, 39208, 10434, 57037, 14, 67956, 109387, 51320, 16, 70398, 686, "
              "10605, 1024, 14415, 223, 94849, 14, 312, 1133, 15932, 395, 223, 94849, 39672, "
              "349, 120184, 22047, 279, 260, 16, 6153, 29690, 309, 57734, 11748, 10154, 376, "
              "403, 668, 314, 3253, 67, 12459, 48036, 716, 399, 604, 601, 72493, 14, 716, 295, "
              "1377, 376, 1440, 267, 3814, 376, 16]")

def main():
    if not os.path.exists(CONST):
        print(f"WARN: {CONST} not found; skipping warmup patch"); return 0
    src = io.open(CONST, encoding="utf-8").read()
    if re.search(r"['\"]deepseek-r1['\"]\s*:", src):
        print("[ok] deepseek-r1 warmup sample already present - no change"); return 0
    # insert right after `ENCODED_SAMPLES = {`
    m = re.search(r"(ENCODED_SAMPLES\s*=\s*\{)", src)
    if not m:
        print("WARN: could not find WarmUp.ENCODED_SAMPLES dict; skipping"); return 0
    ins = m.group(1) + "\n        'deepseek-r1': " + DSR_SAMPLE + ",\n"
    out = src[:m.start(1)] + ins + src[m.end(1):]
    io.open(CONST, "w", encoding="utf-8").write(out)
    # verify it parses
    import py_compile
    try:
        py_compile.compile(CONST, doraise=True)
    except Exception as e:
        io.open(CONST, "w", encoding="utf-8").write(src)  # revert
        print(f"ERROR: patched constants.py failed to compile ({e}); reverted"); return 1
    print("[ok] injected deepseek-r1 warmup sample into WarmUp.ENCODED_SAMPLES")
    return 0

if __name__ == "__main__":
    sys.exit(main())
