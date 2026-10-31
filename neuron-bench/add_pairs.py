import json,torch
g=torch.Generator().manual_seed(4343)
a=[i for i in torch.randperm(9216,generator=g).tolist() if i!=8949][:20]
b=[i for i in torch.randperm(9216,generator=g).tolist() if i!=7956][:20]
with open("arms.jsonl","a") as f:
  for k,(x,y) in enumerate(zip(a,b)):
    for d in [-5,0,2,5,10]:
      f.write(json.dumps({"arm_id":f"randpair_{k}_d{d}","layer":[14,13],"neuron":[x,y],"dose_mult":[d*1,d*-1] if d else [0,0],"op":"ablate" if d==0 else "add","sign_note":"random L14 +1 / L13 -1 control for Lyra pair, seed 4343"})+"\n")
print(list(zip(a,b)))
