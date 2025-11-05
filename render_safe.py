# 把下面这两个路径改成你机器上的真实绝对路径
DATA_DIR  = Path("/Users/yangjiyue/Desktop/NeRF/my_scene/processed")
CKPT_PATH = Path("/Users/yangjiyue/Desktop/NeRF/outputs/processed/nerfacto/2025-10-29_154943/nerfstudio_models/step-000002999.ckpt")
import os
import torch
from pathlib import Path
import imageio.v2 as imageio
import torch.serialization
import numpy as np

# 防止 Mac 上 OpenMP 多副本崩溃
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====== Patch torch.load: 允许完整还原 ckpt (PyTorch 2.6 安全限制) ======
_orig_load = torch.load

def trusted_load(f, map_location=None, **kwargs):
    # 我们信任自己训练出来的 checkpoint，所以允许加载完整对象
    kwargs["weights_only"] = False

    # PyTorch 2.6 限制反序列化某些 numpy 的类，这里手动放行
    with torch.serialization.safe_globals({
        np.core.multiarray.scalar: np.core.multiarray.scalar
    }):
        return _orig_load(f, map_location=map_location, **kwargs)

torch.load = trusted_load
# =====================================================================


def main():
    # --------- 路径配置（按你机器的实际训练输出）---------
    DATA_DIR  = Path("/Users/yangjiyue/Desktop/NeRF/my_scene/processed")
    CKPT_PATH = Path("/Users/yangjiyue/Desktop/NeRF/outputs/processed/nerfacto/2025-10-29_154943/nerfstudio_models/step-000002999.ckpt")

    OUTPUT_DIR = Path("renders/stills_manual")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 使用数据目录:", DATA_DIR)
    print(">>> 使用 checkpoint:", CKPT_PATH)

    # --------- 读 checkpoint: 会得到一个 dict ---------
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # ckpt.keys() 应该包含: ['step','pipeline','optimizers','schedulers','scalers']
    pipeline_state = ckpt["pipeline"]

    print(">>> ckpt keys:", ckpt.keys())
    print(">>> pipeline_state 例子:", list(pipeline_state.keys())[:10])

    # --------- 导入 nerfstudio 的必要组件 ---------
    # pipeline / datamanager / model config 这些类
    from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
    from nerfstudio.data.datamanagers.parallel_datamanager import (
        ParallelDataManager,
        ParallelDataManagerConfig,
    )
    from nerfstudio.data.pixel_samplers import PixelSampler, PixelSamplerConfig
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
        Nerfstudio as NerfstudioDataParser,
        NerfstudioDataParserConfig,
    )
    from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

    # 渲染要用到的相机和射线工具
    from nerfstudio.cameras.cameras import Cameras
    from nerfstudio.cameras.rays import RayBundle

    print(">>> 搭建推理配置(ParallelDataManager / 单进程模式)...")

    # dataparser: 负责从 processed 数据中读取相机位姿 / intrinsics / imgs
    dataparser_cfg = NerfstudioDataParserConfig(
        _target=NerfstudioDataParser,
        data=Path("."),      # 会相对于 DATA_DIR 解析
        scale_factor=1.0,
        scene_scale=1.0,
        orientation_method="up",
        center_method="poses",
        auto_scale_poses=True,

        eval_mode="fraction",
        train_split_fraction=0.9,
        eval_interval=8,
    )

    # datamanager: 把数据喂给模型
    # 我们用 ParallelDataManager 但把 num_processes=0，这样它不会开子进程（Mac 上安全）
    datamanager_cfg = ParallelDataManagerConfig(
        _target=ParallelDataManager,
        data=DATA_DIR,
        dataparser=dataparser_cfg,

        # batch 大小小一点，因为我们跑 CPU
        train_num_rays_per_batch=1024,
        eval_num_rays_per_batch=1024,

        pixel_sampler=PixelSamplerConfig(
            _target=PixelSampler,
            num_rays_per_batch=1024,
            keep_full_image=False,
            is_equirectangular=False,
            rejection_sample_mask=True,
        ),

        camera_res_scale_factor=1.0,

        # 💡 关键：不走多进程
        num_processes=0,
        queue_size=1,
    )

    # Nerfacto 模型配置（CPU 版本，implementation="torch"）
    model_cfg = NerfactoModelConfig(
        _target=NerfactoModel,
        implementation="torch",   # 不用 tiny-cuda-nn，纯 PyTorch
        eval_num_rays_per_chunk=4096,
        hidden_dim=64,
        hidden_dim_color=64,
        near_plane=0.05,
        far_plane=1000.0,
        background_color="last_sample",
    )

    # pipeline = datamanager + model
    pipeline_cfg = VanillaPipelineConfig(
        _target=VanillaPipeline,
        datamanager=datamanager_cfg,
        model=model_cfg,
    )

    print(">>> 根据 config 实例化 pipeline (CPU)...")
    pipeline: VanillaPipeline = pipeline_cfg.setup(device=torch.device("cpu"))

    model = pipeline.model
    datamanager = pipeline.datamanager
    model.eval()  # 关掉训练模式，推理用

    # --------- 把 ckpt 的权重塞回 model ---------
    # ckpt["pipeline"] 里是各种 module 的 state_dict，但 key 带前缀，比如 "_model.xxx"
    model_state = {}
    for k, v in pipeline_state.items():
        if k.startswith("_model."):
            new_k = k[len("_model."):]
            model_state[new_k] = v
        elif k.startswith("model."):
            new_k = k[len("model."):]
            model_state[new_k] = v
        # 其他 key (比如 datamanager 里的状态) 我们忽略

    print(">>> 从 ckpt 提取到的模型参数个数:", len(model_state))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(">>> missing keys:", missing)
    print(">>> unexpected keys:", unexpected)

    # --------- 找到我们要渲染的相机视角 ---------
    # datamanager 通常会有 eval_dataset
    eval_dataset = getattr(datamanager, "eval_dataset", None)
    if eval_dataset is None and hasattr(datamanager, "get_eval_dataset"):
        eval_dataset = datamanager.get_eval_dataset()
    if eval_dataset is None:
        print(">>> 没有 eval_dataset，尝试 train_dataset")
        eval_dataset = getattr(datamanager, "train_dataset", None)
    if eval_dataset is None:
        raise RuntimeError("无法拿到可渲染的数据集 (eval/train 都为空)")

    num_views_to_render = min(5, len(eval_dataset))
    print(f">>> 开始渲染前 {num_views_to_render} 个视角到 PNG ...")

    # --------- 循环每个视角，生成RGB并保存 ---------
    for i in range(num_views_to_render):
        sample = eval_dataset[i]

        # 不同 nerfstudio 版本返回的结构不完全一样
        # 1) 有的返回 dict{'cameras': Cameras, 'image': ...}
        # 2) 有的直接是一个对象，属性里有 .cameras
        if isinstance(sample, dict) and "cameras" in sample:
            cam: Cameras = sample["cameras"]
        elif hasattr(sample, "cameras"):
            cam: Cameras = sample.cameras
        else:
            raise RuntimeError(
                "这个 nerfstudio 版本的 eval_dataset[i] 拿不到 cameras。"
            )

        # 用相机生成射线 RayBundle (每个像素对应一条光线，包含方向/原点等)
        ray_bundle: RayBundle = cam.generate_rays(device=torch.device("cpu"))

        with torch.no_grad():
            # NerfactoModel 有方法 get_outputs_for_camera_ray_bundle
            outputs = model.get_outputs_for_camera_ray_bundle(ray_bundle)
            # 里面通常有 'rgb' 这个 key: 形状 [H, W, 3]，范围在 [0,1]
            rgb = outputs["rgb"]

        rgb_np = rgb.cpu().numpy()
        rgb_np = (rgb_np * 255).astype(np.uint8)

        out_path = OUTPUT_DIR / f"manual_frame_{i:03d}.png"
        imageio.imwrite(out_path, rgb_np)
        print(f"✅ 已保存 {out_path}")

    print(">>> 全部完成！去看看 renders/stills_manual/ 里的 PNG 吧。")


if __name__ == "__main__":
    main()
