import os

import numpy as np


_MODELS = {}


def dino_cls_images(images, device="cuda", batch=32):
    import torch
    from PIL import Image
    if "dino" not in _MODELS:
        os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
        _MODELS["dino"] = torch.hub.load(os.path.join(os.environ["TORCH_HOME"], "hub", "facebookresearch_dinov2_main"),
                                         "dinov2_vitl14_reg", source="local").to(device).eval()
    model = _MODELS["dino"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(images), batch):
        ims = []
        for im in images[s:s + batch]:
            if im.mode == "RGBA":
                im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
            ims.append(torch.from_numpy(np.asarray(im.convert("RGB").resize((518, 518), Image.LANCZOS))).permute(2, 0, 1).float() / 255)
        with torch.no_grad():
            out.append(model((torch.stack(ims).to(device) - mean) / std).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def dino_cls(png_paths, device="cuda", batch=32):
    from PIL import Image
    return np.concatenate([dino_cls_images([Image.open(p) for p in png_paths[s:s + batch]], device, batch)
                           for s in range(0, len(png_paths), batch)], 0)


def lpips_pairs(a_imgs, b_imgs, device="cuda", batch=32):
    import lpips
    import torch
    from PIL import Image
    if "lpips" not in _MODELS:
        _MODELS["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    net = _MODELS["lpips"]

    def _t(ims):
        def rgb(im):
            if im.mode == "RGBA":
                im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
            return np.asarray(im.convert("RGB"))
        return torch.stack([torch.from_numpy(rgb(im)).permute(2, 0, 1).float() / 127.5 - 1 for im in ims]).to(device)
    out = []
    for s in range(0, len(a_imgs), batch):
        with torch.no_grad():
            out.append(net(_t(a_imgs[s:s + batch]), _t(b_imgs[s:s + batch])).flatten().cpu().numpy())
    return np.concatenate(out)


