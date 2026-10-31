# arms.jsonl
111 rows: stock(1) + nisten L8N1955 x5 doses + Lyra pair x5 + 20 random L8 neurons x5 doses.
Doses (multiples of the neuron's own mean|activation| on the calibration set; Maya computes scale): [-5, 0, 2, 5, 10].
layer = 0-indexed model.model.layers[layer]; neuron = index into MLP intermediate act (input to down_proj); op: dose_mult>0 or <0 => ADD dose_mult * mean|act| (this neuron, calibration set) to the activation at all positions; dose_mult==0 => ABLATE (set activation to 0), not add-0.
Validation: all layer < 32, all neuron < intermediate_size 9216 (config.json text_config).
Random neurons: `g = torch.Generator().manual_seed(4242); perm = torch.randperm(9216, generator=g).tolist(); rand = [n for n in perm if n != 1955][:20]` -> [4014, 3596, 831, 9063, 4918, 4711, 338, 5752, 9011, 6614, 277, 4357, 7000, 8488, 7505, 9195, 7277, 305, 116, 3711]
Lyra sign convention copied from /workspace/steer-withheld/tbL/run.py lines 54-55: (L14, n8949, +1), (L13, n7956, -1); applied as one combined arm, per-component dose_mult = dose*sign.
Regenerate: python3 make_arms.py (torch from /workspace/tbL_venv or system).
randpair: add_pairs.py seed 4343, 20 L14(+1)/L13(-1) pairs x 5 doses appended
