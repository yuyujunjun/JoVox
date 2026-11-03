import os

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RES = 1024
SIMPLIFY_TARGET = 1_000_000
_KEEP = ("shape_slat_flow_model_1024", "shape_slat_decoder")
_pipe = None


def mesh_pipeline():
    global _pipe
    if _pipe is None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        class _GeometryOnly(Trellis2ImageTo3DPipeline):
            model_names_to_load = list(_KEEP)
        p = _GeometryOnly.from_pretrained("microsoft/TRELLIS.2-4B")
        assert sorted(p.models) == sorted(_KEEP), f"loaded models {sorted(p.models)} != {sorted(_KEEP)}"
        p.low_vram = False
        p.cuda()
        print(f"[stage2] TRELLIS.2-4B '1024'; models {sorted(p.models)}; shape sampler {p.shape_slat_sampler_params}",
              flush=True)
        _pipe = p
    return _pipe


def voxels_to_mesh(p, image, seed, coords_np):
    import torch
    coords = torch.as_tensor(np.concatenate([np.zeros((len(coords_np), 1), np.int64), coords_np], 1),
                             dtype=torch.int32, device="cuda")
    with torch.no_grad():
        cond = p.get_cond([p.preprocess_image(image)], RES)
        torch.manual_seed(seed)
        slat = p.sample_shape_slat(cond, p.models["shape_slat_flow_model_1024"], coords)
        meshes, _ = p.decode_shape_slat(slat, RES)
        m = meshes[0]
        del slat, meshes, cond
        torch.cuda.empty_cache()
        m.fill_holes()
        m.simplify(SIMPLIFY_TARGET)
    return m.vertices.float().cpu().numpy(), m.faces.int().cpu().numpy()


def background_remover():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    class _RembgOnly(Trellis2ImageTo3DPipeline):
        model_names_to_load = []
    p = _RembgOnly.from_pretrained("microsoft/TRELLIS.2-4B")
    p.low_vram = False
    p.cuda()
    assert getattr(p, "rembg_model", None) is not None, "the TRELLIS.2 pipeline has no background removal model (rembg_model)"
    return p.rembg_model
