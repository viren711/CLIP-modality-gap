"""Turn the released NegCLIP training checkpoint into a plain state_dict.

Usage: .venv/bin/python scripts/convert_negclip.py checkpoints/negclip.pth checkpoints/negclip_state_dict.pt

The release (1.8 GB) also holds optimizer / AMP-scaler state with pickled
numpy scalars, which torch.load(weights_only=True) rejects. We allow just
that one numpy type (instead of weights_only=False, which would run
arbitrary pickle code), keep only the model tensors (~600 MB), and from
then on the model loads through the normal safe path.
"""
import sys

import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
           type(np.dtype(np.float64)), type(np.dtype(np.int64))]
with torch.serialization.safe_globals(allowed):
    ck = torch.load(src, map_location="cpu", weights_only=True)
sd = ck["state_dict"]
assert all(isinstance(v, torch.Tensor) for v in sd.values())
sd = {k.removeprefix("module."): v for k, v in sd.items()}
torch.save(sd, dst)
print(f"{ck['name']} (epoch {ck['epoch']}): saved {len(sd)} tensors to {dst}")
