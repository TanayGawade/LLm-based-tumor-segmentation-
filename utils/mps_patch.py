import torch
import gc

# Create a dummy MPS module
class MPSModule:
    def empty_cache(self):
        gc.collect()

# Patch torch to include mps module
if not hasattr(torch, 'mps'):
    torch.mps = MPSModule()

# Patch torch.cuda to handle MPS devices
original_empty_cache = torch.cuda.empty_cache
def patched_empty_cache():
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    else:
        original_empty_cache()
torch.cuda.empty_cache = patched_empty_cache 