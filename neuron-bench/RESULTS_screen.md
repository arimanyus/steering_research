# MMLU damage screen: Qwen3.5-4B, CPU

Stage `screen` = `data/mmlu_screen200.jsonl` (200 MMLU test items, all 57 subjects, disjoint from calib200 and mmlu500). Zero-shot, score = log-softmax over ' A'..' D' at the last position, bs=1, no generation. dtype bfloat16, torch 2.15.0+cpu, threads 4.

Arms done here: 52/211. Not in this table: rand_L8_k08_n9011_d0, rand_L8_k08_n9011_d2, rand_L8_k08_n9011_d5, rand_L8_k08_n9011_d10, rand_L8_k09_n6614_d-5, rand_L8_k09_n6614_d0, rand_L8_k09_n6614_d2, rand_L8_k09_n6614_d5, rand_L8_k09_n6614_d10, rand_L8_k10_n277_d-5, rand_L8_k10_n277_d0, rand_L8_k10_n277_d2, rand_L8_k10_n277_d5, rand_L8_k10_n277_d10, rand_L8_k11_n4357_d-5, rand_L8_k11_n4357_d0, rand_L8_k11_n4357_d2, rand_L8_k11_n4357_d5, rand_L8_k11_n4357_d10, rand_L8_k12_n7000_d-5, rand_L8_k12_n7000_d0, rand_L8_k12_n7000_d2, rand_L8_k12_n7000_d5, rand_L8_k12_n7000_d10, rand_L8_k13_n8488_d-5, rand_L8_k13_n8488_d0, rand_L8_k13_n8488_d2, rand_L8_k13_n8488_d5, rand_L8_k13_n8488_d10, rand_L8_k14_n7505_d-5, rand_L8_k14_n7505_d0, rand_L8_k14_n7505_d2, rand_L8_k14_n7505_d5, rand_L8_k14_n7505_d10, rand_L8_k15_n9195_d-5, rand_L8_k15_n9195_d0, rand_L8_k15_n9195_d2, rand_L8_k15_n9195_d5, rand_L8_k15_n9195_d10, rand_L8_k16_n7277_d-5, rand_L8_k16_n7277_d0, rand_L8_k16_n7277_d2, rand_L8_k16_n7277_d5, rand_L8_k16_n7277_d10, rand_L8_k17_n305_d-5, rand_L8_k17_n305_d0, rand_L8_k17_n305_d2, rand_L8_k17_n305_d5, rand_L8_k17_n305_d10, rand_L8_k18_n116_d-5, rand_L8_k18_n116_d0, rand_L8_k18_n116_d2, rand_L8_k18_n116_d5, rand_L8_k18_n116_d10, rand_L8_k19_n3711_d-5, rand_L8_k19_n3711_d0, rand_L8_k19_n3711_d2, rand_L8_k19_n3711_d5, rand_L8_k19_n3711_d10, randpair_0_d-5, randpair_0_d0, randpair_0_d2, randpair_0_d5, randpair_0_d10, randpair_1_d-5, randpair_1_d0, randpair_1_d2, randpair_1_d5, randpair_1_d10, randpair_2_d-5, randpair_2_d0, randpair_2_d2, randpair_2_d5, randpair_2_d10, randpair_3_d-5, randpair_3_d0, randpair_3_d2, randpair_3_d5, randpair_3_d10, randpair_4_d-5, randpair_4_d0, randpair_4_d2, randpair_4_d5, randpair_4_d10, randpair_5_d-5, randpair_5_d0, randpair_5_d2, randpair_5_d5, randpair_5_d10, randpair_6_d-5, randpair_6_d0, randpair_6_d2, randpair_6_d5, randpair_6_d10, randpair_7_d-5, randpair_7_d0, randpair_7_d2, randpair_7_d5, randpair_7_d10, randpair_8_d-5, randpair_8_d0, randpair_8_d2, randpair_8_d5, randpair_8_d10, randpair_9_d-5, randpair_9_d0, randpair_9_d2, randpair_9_d5, randpair_9_d10, randpair_10_d-5, randpair_10_d0, randpair_10_d2, randpair_10_d5, randpair_10_d10, randpair_11_d-5, randpair_11_d0, randpair_11_d2, randpair_11_d5, randpair_11_d10, randpair_12_d-5, randpair_12_d0, randpair_12_d2, randpair_12_d5, randpair_12_d10, randpair_13_d-5, randpair_13_d0, randpair_13_d2, randpair_13_d5, randpair_13_d10, randpair_14_d-5, randpair_14_d0, randpair_14_d2, randpair_14_d5, randpair_14_d10, randpair_15_d-5, randpair_15_d0, randpair_15_d2, randpair_15_d5, randpair_15_d10, randpair_16_d-5, randpair_16_d0, randpair_16_d2, randpair_16_d5, randpair_16_d10, randpair_17_d-5, randpair_17_d0, randpair_17_d2, randpair_17_d5, randpair_17_d10, randpair_18_d-5, randpair_18_d0, randpair_18_d2, randpair_18_d5, randpair_18_d10, randpair_19_d-5, randpair_19_d0, randpair_19_d2, randpair_19_d5, randpair_19_d10
This table covers arms.jsonl lines 1-80 (this run); lines 81-211 are run by shard agents and land under results/screen/shard{1..4}/.

`acc` = fraction correct on the n items. `d_acc` = acc minus stock acc on the same items. `d_gold_lp` = mean over items of (gold-letter logprob under the arm minus under stock), paired per item, with standard error. Gold logprob is within the 4-letter log-softmax.

## Caveats

- Doses are in units of each neuron's own mean|act| over calib200 (stock model, all token positions), NOT Lyra run.py's resid-norm/||down_proj col|| units. The lyra_pair_* arms therefore do not reproduce Lyra's dose scale and the comparison to Lyra is inexact.
- The screen measures MMLU damage only (accuracy and gold logprob on 200 items); it says nothing about whether an arm produces the claimed behavioural effect.
- n=200 items per arm: 1 item = 0.5 pp of accuracy; the binomial SE of an acc near 0.76 is about 3 pp.

## Per-arm results

| arm | n | acc | d_acc | d_gold_lp (mean +/- se) | peak RSS MB |
|---|---|---|---|---|---|
| stock | 200 | 0.760 | +0.000 | +0.0000 +/- 0.0000 | 9574 |
| nisten_L8N1955_d-5 | 200 | 0.755 | -0.005 | +0.0497 +/- 0.0076 | 9571 |
| nisten_L8N1955_d0 | 200 | 0.750 | -0.010 | -0.0058 +/- 0.0070 | 9697 |
| nisten_L8N1955_d2 | 200 | 0.755 | -0.005 | -0.0209 +/- 0.0069 | 9755 |
| nisten_L8N1955_d5 | 200 | 0.755 | -0.005 | -0.0529 +/- 0.0073 | 9755 |
| nisten_L8N1955_d10 | 200 | 0.755 | -0.005 | -0.0916 +/- 0.0090 | 9766 |
| lyra_pair_d-5 | 200 | 0.755 | -0.005 | +0.0273 +/- 0.0077 | 9786 |
| lyra_pair_d0 | 200 | 0.750 | -0.010 | +0.0003 +/- 0.0068 | 9786 |
| lyra_pair_d2 | 200 | 0.755 | -0.005 | +0.0021 +/- 0.0071 | 9794 |
| lyra_pair_d5 | 200 | 0.755 | -0.005 | -0.0316 +/- 0.0068 | 9805 |
| lyra_pair_d10 | 200 | 0.760 | +0.000 | -0.0558 +/- 0.0074 | 9805 |
| rand_L8_k00_n4014_d-5 | 200 | 0.755 | -0.005 | -0.0054 +/- 0.0073 | 9575 |
| rand_L8_k00_n4014_d0 | 200 | 0.760 | +0.000 | +0.0084 +/- 0.0072 | 9705 |
| rand_L8_k00_n4014_d2 | 200 | 0.750 | -0.010 | +0.0102 +/- 0.0069 | 9739 |
| rand_L8_k00_n4014_d5 | 200 | 0.755 | -0.005 | +0.0099 +/- 0.0078 | 9762 |
| rand_L8_k00_n4014_d10 | 200 | 0.760 | +0.000 | +0.0282 +/- 0.0081 | 9762 |
| rand_L8_k01_n3596_d-5 | 200 | 0.755 | -0.005 | +0.0160 +/- 0.0073 | 9780 |
| rand_L8_k01_n3596_d0 | 200 | 0.760 | +0.000 | +0.0013 +/- 0.0071 | 9795 |
| rand_L8_k01_n3596_d2 | 200 | 0.760 | +0.000 | +0.0062 +/- 0.0068 | 9795 |
| rand_L8_k01_n3596_d5 | 200 | 0.760 | +0.000 | +0.0007 +/- 0.0074 | 9795 |
| rand_L8_k01_n3596_d10 | 200 | 0.760 | +0.000 | +0.0030 +/- 0.0079 | 9795 |
| rand_L8_k02_n831_d-5 | 200 | 0.760 | +0.000 | +0.0389 +/- 0.0078 | 9795 |
| rand_L8_k02_n831_d0 | 200 | 0.755 | -0.005 | +0.0144 +/- 0.0076 | 9795 |
| rand_L8_k02_n831_d2 | 200 | 0.750 | -0.010 | -0.0093 +/- 0.0071 | 9795 |
| rand_L8_k02_n831_d5 | 200 | 0.755 | -0.005 | -0.0271 +/- 0.0071 | 9795 |
| rand_L8_k02_n831_d10 | 200 | 0.760 | +0.000 | -0.0548 +/- 0.0084 | 9795 |
| rand_L8_k03_n9063_d-5 | 200 | 0.760 | +0.000 | +0.0049 +/- 0.0070 | 9795 |
| rand_L8_k03_n9063_d0 | 200 | 0.760 | +0.000 | -0.0014 +/- 0.0069 | 9796 |
| rand_L8_k03_n9063_d2 | 200 | 0.755 | -0.005 | +0.0079 +/- 0.0074 | 9796 |
| rand_L8_k03_n9063_d5 | 200 | 0.755 | -0.005 | +0.0100 +/- 0.0073 | 9796 |
| rand_L8_k03_n9063_d10 | 200 | 0.755 | -0.005 | +0.0081 +/- 0.0084 | 9796 |
| rand_L8_k04_n4918_d-5 | 200 | 0.755 | -0.005 | +0.0163 +/- 0.0073 | 9796 |
| rand_L8_k04_n4918_d0 | 200 | 0.755 | -0.005 | -0.0071 +/- 0.0071 | 9796 |
| rand_L8_k04_n4918_d2 | 200 | 0.755 | -0.005 | -0.0083 +/- 0.0070 | 9796 |
| rand_L8_k04_n4918_d5 | 200 | 0.760 | +0.000 | -0.0187 +/- 0.0073 | 9796 |
| rand_L8_k04_n4918_d10 | 200 | 0.765 | +0.005 | -0.0103 +/- 0.0079 | 9796 |
| rand_L8_k05_n4711_d-5 | 200 | 0.755 | -0.005 | +0.0185 +/- 0.0076 | 14406 |
| rand_L8_k05_n4711_d0 | 200 | 0.755 | -0.005 | +0.0040 +/- 0.0073 | 14406 |
| rand_L8_k05_n4711_d2 | 200 | 0.760 | +0.000 | -0.0061 +/- 0.0069 | 14406 |
| rand_L8_k05_n4711_d5 | 200 | 0.760 | +0.000 | +0.0033 +/- 0.0081 | 14406 |
| rand_L8_k05_n4711_d10 | 200 | 0.760 | +0.000 | -0.0023 +/- 0.0085 | 14406 |
| rand_L8_k06_n338_d-5 | 200 | 0.765 | +0.005 | +0.0248 +/- 0.0078 | 14406 |
| rand_L8_k06_n338_d0 | 200 | 0.760 | +0.000 | +0.0027 +/- 0.0069 | 14406 |
| rand_L8_k06_n338_d2 | 200 | 0.755 | -0.005 | -0.0038 +/- 0.0069 | 14406 |
| rand_L8_k06_n338_d5 | 200 | 0.750 | -0.010 | -0.0116 +/- 0.0086 | 14406 |
| rand_L8_k06_n338_d10 | 200 | 0.745 | -0.015 | -0.0115 +/- 0.0094 | 14699 |
| rand_L8_k07_n5752_d-5 | 200 | 0.755 | -0.005 | +0.0057 +/- 0.0069 | 14699 |
| rand_L8_k07_n5752_d0 | 200 | 0.755 | -0.005 | +0.0027 +/- 0.0076 | 14699 |
| rand_L8_k07_n5752_d2 | 200 | 0.755 | -0.005 | +0.0045 +/- 0.0069 | 14699 |
| rand_L8_k07_n5752_d5 | 200 | 0.760 | +0.000 | +0.0119 +/- 0.0077 | 14699 |
| rand_L8_k07_n5752_d10 | 200 | 0.755 | -0.005 | +0.0094 +/- 0.0080 | 14699 |
| rand_L8_k08_n9011_d-5 | 118 | 0.712 | -0.008 | -0.0130 +/- 0.0104 | 14699 |
