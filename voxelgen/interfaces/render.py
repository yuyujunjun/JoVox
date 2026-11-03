import math

import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr


class VoxelMesh:
    def __init__(self, v_pos, t_pos_idx, v_nrm=None, t_nrm_idx=None):
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self.v_nrm = v_nrm
        self.t_nrm_idx = t_nrm_idx


def _normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def vertex_normals(mesh: VoxelMesh) -> VoxelMesh:
    i0, i1, i2 = mesh.t_pos_idx[:, 0], mesh.t_pos_idx[:, 1], mesh.t_pos_idx[:, 2]
    v0, v1, v2 = mesh.v_pos[i0, :], mesh.v_pos[i1, :], mesh.v_pos[i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    n = torch.zeros_like(mesh.v_pos)
    for i in (i0, i1, i2):
        n.scatter_add_(0, i[:, None].repeat(1, 3), face_normals)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=n.device)
    n = torch.where(torch.sum(n * n, -1, keepdim=True) > 1e-20, n, up)
    return VoxelMesh(mesh.v_pos, mesh.t_pos_idx, _normalize(n), mesh.t_pos_idx)


def render_mesh(glctx, mesh, mtx, resolution, shading_func, antialias=True, **kwargs):
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        faces = mesh.t_pos_idx.type(torch.int32)
        vert_hom = torch.cat((mesh.v_pos, torch.ones(mesh.v_pos.shape[0], 1, device=mesh.v_pos.device)), axis=-1)
        clip = vert_hom @ mtx.transpose(-2, -1)
        rast, _ = dr.rasterize(glctx, clip[None], faces, resolution=[resolution, resolution], grad_db=False)
        buffers = shading_func(rast, None, mesh, **kwargs)
        if antialias:
            buffers = {k: dr.antialias(v.contiguous(), rast, clip, faces) for k, v in buffers.items()}
        return buffers


def render_mesh_batch(glctx, mesh, mtx, resolution, shading_func, antialias=True, **kwargs):
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        assert mesh.v_pos.dim() == 3 and mtx.dim() == 3, "a batched render takes v_pos [B, V, 3] and one matrix per view [B, 4, 4]"
        B, V = mesh.v_pos.shape[:2]
        faces = mesh.t_pos_idx.type(torch.int32)
        vert_hom = torch.cat((mesh.v_pos, torch.ones(B, V, 1, device=mesh.v_pos.device)), axis=-1)
        clip = torch.bmm(vert_hom, mtx.transpose(-2, -1)).to(torch.float32)
        rast, _ = dr.rasterize(glctx, clip, faces, resolution=[resolution, resolution], grad_db=False)
        buffers = shading_func(rast, None, mesh, **kwargs)
        if antialias:
            buffers = {k: dr.antialias(v.contiguous(), rast, clip, faces) for k, v in buffers.items()}
        return buffers


def coords_to_cube_mesh(coords: torch.Tensor, sparse_res: int, device: str = "cuda") -> VoxelMesh:
    coords_b = coords[coords[:, 0] == 0][:, 1:].float()
    if coords_b.numel() == 0:
        return None

    pitch = 1.0 / float(sparse_res)
    centers = (coords_b.cpu().numpy() + 0.5) * pitch - 0.5
    mesh = trimesh.voxel.ops.multibox(centers, pitch=pitch)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return None

    v_pos = torch.as_tensor(mesh.vertices, device=device, dtype=torch.float32)
    t_pos_idx = torch.as_tensor(mesh.faces, device=device, dtype=torch.long)
    return VoxelMesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


def _render_to_image(buffers: dict, key: str) -> np.ndarray:
    data = buffers[key][0]
    alpha = buffers.get("alpha")
    if alpha is not None:
        a = alpha[0, ..., 0]
        mask = a < 0.1
    else:
        mask = None

    img = data.detach().cpu().numpy()

    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = img.repeat(3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    if mask is not None:
        img[mask.cpu().numpy()] = 0

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _empty_image(height: int = 512, width: int = 512) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _pbr_env_shading(rast, rast_db, mesh, v_pos_clip=None, triangle_colors=None,
                      metallic=0.05, roughness=0.4, sky_top=None, sky_horizon=None,
                      ground_color=None, cam_pos=None, **kwargs):
    nrm, _ = dr.interpolate(mesh.v_nrm, rast, mesh.t_nrm_idx.int())
    nrm = nrm[..., :3]
    mask = rast[..., -1] > 0
    nrm[mask] = _normalize(nrm[mask])

    pos, _ = dr.interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    pos = pos[..., :3]

    if cam_pos is not None:
        view_dir = _normalize(pos - cam_pos.to(pos.device))
    else:
        view_dir = _normalize(-pos)

    frontface = torch.sum(nrm * view_dir, dim=-1, keepdim=True) > 0
    nrm = torch.where(frontface, nrm, -nrm)

    if triangle_colors is not None:
        tri_id = rast[..., 3].long() - 1
        valid_tri = tri_id >= 0
        albedo = torch.zeros(*tri_id.shape, 3, device=rast.device, dtype=torch.float32)
        if valid_tri.any():
            albedo[valid_tri] = triangle_colors[tri_id[valid_tri].clamp(min=0)]
    else:
        albedo = torch.ones_like(nrm) * 0.85

    def _ensure_tensor(c, default):
        if c is None:
            return default
        if torch.is_tensor(c):
            return c
        return torch.tensor(c, device=nrm.device, dtype=torch.float32)

    sky_top_c = _ensure_tensor(sky_top, torch.tensor([0.55, 0.65, 0.87], device=nrm.device))
    sky_horizon_c = _ensure_tensor(sky_horizon, torch.tensor([0.82, 0.85, 0.90], device=nrm.device))
    ground_c = _ensure_tensor(ground_color, torch.tensor([0.48, 0.42, 0.34], device=nrm.device))
    sun_color = torch.tensor([1.0, 0.93, 0.82], device=nrm.device)
    sun_dir = _normalize(torch.tensor([0.3, 0.25, -0.85], device=nrm.device))

    def env_sample(direction):
        z = direction[..., 2:3]
        sky = torch.lerp(sky_horizon_c, sky_top_c, torch.clamp(-z, 0.0, 1.0))
        gnd = torch.lerp(sky_horizon_c, ground_c, torch.clamp(z, 0.0, 1.0))
        return torch.where(z > 0, gnd, sky)

    NdotV = torch.clamp(torch.sum(nrm * view_dir, dim=-1, keepdim=True), min=0.0)
    NdotL = torch.clamp(torch.sum(nrm * sun_dir, dim=-1, keepdim=True), min=0.0)

    F0 = torch.lerp(torch.tensor(0.04, device=nrm.device), albedo, metallic)
    F = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

    kD = (1.0 - F) * (1.0 - metallic)

    sun_diffuse = kD * albedo * sun_color * NdotL * 0.85

    env_diffuse = kD * albedo * env_sample(nrm) * 0.35

    half_vec = _normalize(view_dir + sun_dir)
    NdotH = torch.clamp(torch.sum(nrm * half_vec, dim=-1, keepdim=True), min=0.0)

    r2 = roughness * roughness
    a2 = r2 * r2
    denom = NdotH * NdotH * (a2 - 1.0) + 1.0
    D = a2 / (torch.pi * denom * denom + 1e-6)

    k = (roughness + 1.0) * (roughness + 1.0) / 8.0
    gv = NdotV / (NdotV * (1.0 - k) + k + 1e-6)
    gl = NdotL / (NdotL * (1.0 - k) + k + 1e-6)
    G = gv * gl

    sun_specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-6) * sun_color * NdotL

    refl = _normalize(2.0 * NdotV * nrm - view_dir)
    import math
    _exp = torch.exp if torch.is_tensor(r2) else math.exp
    env_spec_strength = _exp(-4.0 * r2) * (1.0 - r2)
    env_specular = F * env_sample(refl) * env_spec_strength * 0.3

    ambient = albedo * torch.lerp(ground_c, sky_top_c, 0.35) * 0.07

    color = sun_diffuse + env_diffuse + sun_specular * 0.55 + env_specular + ambient
    color[mask] = torch.clamp(color[mask], 0.0, 1.0)
    color[~mask] = 1.0

    alpha = torch.clamp(rast[..., -1:], max=1)
    return {"shaded": torch.cat((color, alpha), dim=-1), "alpha": alpha}


def render_voxel_pbr(
    coords: torch.Tensor,
    glctx,
    mtx: torch.Tensor,
    sparse_res: int,
    **kwargs,
) -> np.ndarray:
    mesh = coords_to_cube_mesh(coords, sparse_res, device=coords.device)
    if mesh is None:
        return _empty_image()

    mesh = vertex_normals(mesh)

    metallic = kwargs.get("metallic", 0.05)
    roughness = kwargs.get("roughness", 0.4)
    sky_top = kwargs.get("sky_top", None)
    sky_horizon = kwargs.get("sky_horizon", None)
    ground_color = kwargs.get("ground_color", None)

    buffers = render_mesh(
        glctx, mesh, mtx, 512,
        shading_func=_pbr_env_shading,
        antialias=True,
        triangle_colors=None,
        metallic=metallic,
        roughness=roughness,
        sky_top=sky_top,
        sky_horizon=sky_horizon,
        ground_color=ground_color,
        cam_pos=kwargs.get("cam_pos", None),
    )
    return _render_to_image(buffers, "shaded")


def perspective(fov: float, znear: float, zfar: float) -> torch.Tensor:
    half_w = math.tan(fov / 2) * znear
    width = half_w - (-half_w)
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / width
    P[1, 1] = 2.0 * znear / width
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def make_projection_mtx(device):
    import utils3d
    fov_rad = float(40.0 * torch.pi / 180.0)
    proj = perspective(fov_rad, 0.01, 100.0).to(device)
    orig = torch.tensor([0.0, -1.7, 0.5], device=device, dtype=torch.float32)
    extr = utils3d.torch.extrinsics_look_at(
        orig, torch.zeros(3, device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
    )
    return proj @ extr, orig


def render_voxel_frame(coords, glctx, mtx, sparse_res, cam_pos):
    img_np = render_voxel_pbr(coords, glctx, mtx, sparse_res, cam_pos=cam_pos)
    img = torch.from_numpy(img_np).float() / 255.0
    return img.permute(2, 0, 1)


