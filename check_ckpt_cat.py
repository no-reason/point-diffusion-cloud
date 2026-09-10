import torch
import sys

for p in sys.argv[1:]:
    try:
        ckpt = torch.load(p, map_location='cpu')
        print(f"{p}: {ckpt['args'].categories}")
    except Exception as e:
        pass
