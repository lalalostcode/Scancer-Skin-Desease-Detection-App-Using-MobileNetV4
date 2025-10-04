import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

in_path = r"C:\Local D\Galeri Belajar\Project\Computer Vision\scancer\model\model_mobile.ptl"  # file ptl lama kamu
out_path = r"C:\Local D\Galeri Belajar\Project\Computer Vision\scancer\model\model_mobile_lite.ptl"

# load traced/script module
ts = torch.jit.load(in_path, map_location="cpu")
ts.eval()

optimized_traced_module = optimize_for_mobile(ts)
optimized_traced_module._save_for_lite_interpreter(out_path)

print("Lite model saved to:", out_path)