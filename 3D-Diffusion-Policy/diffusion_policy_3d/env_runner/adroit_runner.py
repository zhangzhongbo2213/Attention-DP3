import wandb
import numpy as np
import torch
import collections
import tqdm
import os
import subprocess
import tempfile
import json
import imageio
import pycocotools.mask as mask_util
import base64
from io import BytesIO
from pathlib import Path
from diffusion_policy_3d.env import AdroitEnv
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

# Try to import requests, fallback to subprocess if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    cprint("[warn] requests library not found. Install with: pip install requests", "yellow")


class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 seg_type='gs2',  # 'gs2' or 'env'
                 num_distraction_nails=0,  # Number of distraction nails for hammer task
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.seg_type = seg_type
        self.num_distraction_nails = num_distraction_nails

        cprint(f"[AdroitRunner] Initialized with seg_type='{seg_type}' for task '{task_name}'", "cyan")

        # Load target geom ids for env-based segmentation
        self.target_geom_ids = None
        if seg_type == 'env' and task_name:
            try:
                # Use absolute path to targets directory
                targets_path = f"/mnt/disk2/ycb/AEDP3/targets/{task_name}_geom_ids.json"
                # attempted targets_path (absolute) is used silently

                if os.path.exists(targets_path):
                    with open(targets_path, "r") as f:
                        self.target_geom_ids = json.load(f)
                    cprint(f"[info] Loaded target geom ids for task '{task_name}' from {targets_path} ({len(self.target_geom_ids)} ids): {self.target_geom_ids}", "cyan")
                else:
                    cprint(f"[warn] Target geom ids file not found: {targets_path}, will use fallback segmentation", "yellow")
            except Exception as e:
                cprint(f"[warn] Failed to load target geom ids: {e}, will use fallback segmentation", "yellow")

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            # Enable segmentation rendering if using env-based segmentation
            render_segmentation = (seg_type == 'env')
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True, render_segmentation=render_segmentation, num_distraction_nails=num_distraction_nails),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        # Buffer to store historical attn_3d (for efficient GS2 usage)
        # Each step generates 1 attn, and we take the last n_obs_steps for the window
        self.attn_history = collections.deque(maxlen=n_obs_steps)

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        # Configuration for Grounded-SAM-2 inference
        # Option 1: Use API server (recommended for performance)
        # Set gs2_api_url to use API server, e.g., "http://127.0.0.1:5000"
        # If None, falls back to subprocess mode
        self.gs2_api_url = os.getenv("GS2_API_URL", "http://127.0.0.1:5000")  # e.g., "http://127.0.0.1:5000"
        # Control verbose output (set GS2_VERBOSE=0 to disable)
        self.gs2_verbose = os.getenv("GS2_VERBOSE", "0").lower() in ("1", "true", "yes")

        # Option 2: Subprocess mode (original, slower)
        self.gs2_conda_env = "aedp3_vis"
        self.gs2_text_prompt = self._get_text_prompt_for_task(task_name)
        # Get project root (3 levels up from this file)
        # __file__: .../3D-Diffusion-Policy/diffusion_policy_3d/env_runner/adroit_runner.py
        # project_root: .../AEDP3
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.gs2_script_path = os.path.join(
            project_root, "Grounded-SAM-2", "infer_grounded_sam2_single.py"
        )
        # Default paths (can be overridden via config)
        self.gs2_sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"
        self.gs2_sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.gs2_gdino_cfg = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        self.gs2_gdino_ckpt = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
        self.gs2_device = "cuda"
        self.gs2_box_thr = 0.25
        self.gs2_text_thr = 0.15

        # Create temp directory for inference images in project root
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    def _point_cloud_sampling(self, point_cloud: np.ndarray, num_points: int, method: str = 'fps'):
        """
        Point cloud sampling function consistent with training
        point_cloud: (N, D) where D can be 6 (xyz+rgb), 8 (xyz+rgb+uv), or 3 (xyz)
        """
        if num_points == 'all':  # use all points
            return point_cloud

        if point_cloud.shape[0] <= num_points:
            # pad with zeros
            point_cloud_dim = point_cloud.shape[-1]
            point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
            return point_cloud

        if method == 'uniform':
            # uniform sampling
            sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
            point_cloud = point_cloud[sampled_indices]
        elif method == 'fps':
            # fast point cloud sampling using torch3d (consistent with training)
            try:
                import pytorch3d.ops as torch3d_ops
                point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
                num_points_tensor = torch.tensor([num_points]).cuda()
                # remember to only use coord to sample
                _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud_tensor[..., :3], K=num_points_tensor)
                point_cloud = point_cloud_tensor.squeeze(0).cpu().numpy()
                point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
            except ImportError:
                # fallback to random sampling if torch3d not available
                cprint("[warn] pytorch3d not available, falling back to random sampling for inference", "yellow")
                sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
                point_cloud = point_cloud[sampled_indices]
        else:
            raise NotImplementedError(f"point cloud sampling method {method} not implemented")

        return point_cloud

    def _get_text_prompt_for_task(self, task_name):
        """Get text prompt for Grounded-SAM-2 based on task name."""
        prompts = {
            'door': 'door handle. door.',
            'hammer': 'hammer. handle.',
            "pen": "blue pen in hand.",
            # Add more task prompts as needed
        }
        return prompts.get(task_name, 'object.')  # Default fallback
        # Option 1: Use API server (recommended for performance)
        # Set gs2_api_url to use API server, e.g., "http://127.0.0.1:5000"
        # If None, falls back to subprocess mode
        self.gs2_api_url = os.getenv("GS2_API_URL", "http://127.0.0.1:5000")  # e.g., "http://127.0.0.1:5000"
        # Control verbose output (set GS2_VERBOSE=0 to disable)
        self.gs2_verbose = os.getenv("GS2_VERBOSE", "0").lower() in ("1", "true", "yes")
        
        # Option 2: Subprocess mode (original, slower)
        self.gs2_conda_env = "aedp3_vis"
        self.gs2_text_prompt = self._get_text_prompt_for_task(task_name)
        # Get project root (3 levels up from this file)
        # __file__: .../3D-Diffusion-Policy/diffusion_policy_3d/env_runner/adroit_runner.py
        # project_root: .../AEDP3
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.gs2_script_path = os.path.join(
            project_root, "Grounded-SAM-2", "infer_grounded_sam2_single.py"
        )
        # Default paths (can be overridden via config)
        self.gs2_sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"
        self.gs2_sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.gs2_gdino_cfg = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.gs2_gdino_ckpt = "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.gs2_device = "cuda"
        self.gs2_box_thr = 0.35
        self.gs2_text_thr = 0.25
        
        # Create temp directory for inference images in project root
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir
    
    def _get_text_prompt_for_task(self, task_name):
        """Get text prompt for Grounded-SAM-2 based on task name."""
        prompts = {
            'door': 'door handle. door.',
            'hammer': 'hammer. handle.',
            "pen": "blue pen in hand.",
            # Add more task prompts as needed
        }
        return prompts.get(task_name, 'object.')  # Default fallback
    
    def _build_attn_from_mask(self, point_cloud, mask_json, img_res=(84, 84), n_points=512, n_channels=3):
        """
        Build attn_3d from point cloud and mask JSON (same logic as convert_zarr_with_attn3d.py).
        point_cloud: (N_pc, 8) xyzrgbuv, where uv is normalized [0, 1]
        mask_json: json dict from grounded_sam2
        img_res: (H, W) of original image
        """
        H, W = img_res

        # Sample or pad point cloud using FPS (consistent with training)
        pc = self._point_cloud_sampling(point_cloud, n_points, method='fps')
        
        xyz = pc[:, :3]
        
        # Extract UV coordinates (normalized [0, 1])
        if pc.shape[1] >= 8:
            u_norm = pc[:, 6]  # normalized u
            v_norm = pc[:, 7]  # normalized v
        else:
            # Fallback: if no UV, use simple method
            u_norm = np.zeros(pc.shape[0], dtype=np.float32)
            v_norm = np.zeros(pc.shape[0], dtype=np.float32)
        
        # Convert normalized UV to pixel coordinates
        u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
        v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)
        
        # Initialize attention field
        attn = np.zeros((n_channels, n_points), dtype=np.float32)
        
        # Check if points are inside any mask
        mask_hit = np.zeros(n_points, dtype=bool)
        
        if mask_json and "annotations" in mask_json and len(mask_json["annotations"]) > 0:
            # Decode all masks and check which points are inside
            for ann in mask_json["annotations"]:
                rle = ann["segmentation"]
                if isinstance(rle, dict) and "counts" in rle:
                    try:
                        # Decode RLE mask
                        mask = mask_util.decode(rle).astype(bool)  # (H, W)
                        # Check which points are inside this mask
                        mask_hit |= mask[v_pix, u_pix]
                    except Exception as e:
                        cprint(f"[warn] Failed to decode mask: {e}", "yellow")
                        continue
        
        # Channel 0: Binary mask hit (1 if inside any mask, 0 otherwise)
        attn[0] = mask_hit.astype(np.float32)
        
        # Channel 1: Distance-based attention (closer to center = higher weight)
        x_center = xyz[:, 0].mean()
        x_dist = np.abs(xyz[:, 0] - x_center)
        x_dist_norm = x_dist / (x_dist.max() + 1e-6)
        attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)  # Only for points in mask
        
        # Channel 2: Inverse distance (for obstacle/background attention)
        attn[2] = (1.0 - mask_hit.astype(np.float32))  # Points NOT in mask
        
        # NOTE: Channel 3 (normalized x coordinate) removed — only channels 0-2 are kept.
        
        return attn


    def _build_attn_from_env_seg(self, point_cloud, seg_data, n_points=512, n_channels=3):
        """
        Build attention from environment-provided segmentation data.
        seg_data: (T, H, W, 2) - MuJoCo segmentation data over T timesteps, where each pixel contains [objtype, objid]
        point_cloud: (T, N_pc, 8) xyzrgbuv, where uv is normalized [0, 1]
        """
        T, H, W, _ = seg_data.shape
        T_pc, N_pc, C_pc = point_cloud.shape

        # Ensure T matches
        assert T == T_pc, f"Time steps mismatch: seg_data has {T}, point_cloud has {T_pc}"

        # Initialize attention field for all timesteps
        attn_all = []

        for t in range(T):
            pc_t = point_cloud[t]  # (N_pc, 8)
            seg_t = seg_data[t]    # (H, W, 2)

            # Sample or pad point cloud using FPS (consistent with training)
            pc_sampled = self._point_cloud_sampling(pc_t, n_points, method='fps')

            xyz = pc_sampled[:, :3]

            # Extract UV coordinates (normalized [0, 1])
            if pc_sampled.shape[1] >= 8:
                u_norm = pc_sampled[:, 6]  # normalized u
                v_norm = pc_sampled[:, 7]  # normalized v

                # UV coordinate ranges checked during development; suppressed in production
            else:
                # Fallback: if no UV, use simple method
                u_norm = np.zeros(pc_sampled.shape[0], dtype=np.float32)
                v_norm = np.zeros(pc_sampled.shape[0], dtype=np.float32)

            # Convert normalized UV to pixel coordinates
            u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
            v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)

            # Get segmentation IDs at point locations
            seg_ids = seg_t[v_pix, u_pix, 1]  # objid

            # For adroit tasks, we prefer to use a precomputed target geom id list (by body)
            # If available, only those geom ids will be treated as targets. Otherwise,
            # fall back to treating any non-zero objid as target.
            if self.target_geom_ids:
                # seg_ids correspond to geom ids (objid) from MuJoCo; keep only those in target list
                try:
                    mask_hit = np.isin(seg_ids, np.array(self.target_geom_ids, dtype=seg_ids.dtype))

                    # Matching and mask statistics computed during development; suppressed in production

                except Exception:
                    mask_hit = seg_ids > 0
                    # Exception in mask computation handled by fallback; suppressed debug message
            else:
                mask_hit = seg_ids > 0
                # No target_geom_ids: using all non-zero segmentation IDs (no debug print)

            # Initialize attention field for this timestep
            attn = np.zeros((n_channels, n_points), dtype=np.float32)

            # Channel 0: Binary mask hit (1 if target object, 0 otherwise)
            attn[0] = mask_hit.astype(np.float32)

            # Channel 1: Distance-based attention (closer to center = higher weight)
            # Use normalized x coordinate as proxy for distance
            x_center = xyz[:, 0].mean()
            x_dist = np.abs(xyz[:, 0] - x_center)
            x_dist_norm = x_dist / (x_dist.max() + 1e-6)
            attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)  # Only for points in mask

            # Channel 2: Inverse mask (for background attention)
            attn[2] = (1.0 - mask_hit.astype(np.float32))  # Points NOT in mask

            attn_all.append(attn)

        # Stack all timesteps: (T, n_channels, n_points)
        return np.stack(attn_all, axis=0)

    def _generate_attn_3d_inference(self, rgb_img, point_cloud_with_uv, img_res=(84, 84), seg_data=None):
        """
        Generate attn_3d during inference.
        If seg_type='env', use environment segmentation data.
        Otherwise, use Grounded-SAM-2.
        Uses API server if available, otherwise falls back to subprocess mode.

        Args:
            rgb_img: (H, W, 3) uint8 RGB image
            point_cloud_with_uv: (N, 8) point cloud with UV coordinates (xyzrgbuv)
            img_res: (H, W) image resolution
            seg_data: (H, W, 2) environment segmentation data (only used when seg_type='env')

        Returns:
            attn_3d: (C, N) attention field
        """
        if self.seg_type == 'env':
            # Use environment segmentation data
            if seg_data is not None:
                cprint(f"[AdroitRunner] Using environment segmentation for inference", "green")
                return self._build_attn_from_env_seg(point_cloud_with_uv, seg_data,
                                                   n_points=512, n_channels=3)
            else:
                cprint(f"[warn] seg_type='env' but no segmentation data provided, using GS2 fallback", "yellow")
                # Fall back to GS2
                if self.gs2_api_url and HAS_REQUESTS:
                    return self._generate_attn_3d_via_api(rgb_img, point_cloud_with_uv, img_res)
                else:
                    return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)
        else:
            # Use Grounded-SAM-2 (original behavior)
            # cprint(f"[AdroitRunner] Using Grounded-SAM-2 for inference", "blue")
            if self.gs2_api_url and HAS_REQUESTS:
                return self._generate_attn_3d_via_api(rgb_img, point_cloud_with_uv, img_res)
            else:
                return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)
    
    def _generate_attn_3d_via_api(self, rgb_img, point_cloud_with_uv, img_res=(84, 84)):
        """Generate attn_3d using HTTP API (faster, no model reload)."""
        try:
            # Encode image to base64
            from PIL import Image
            img_pil = Image.fromarray(rgb_img)
            img_bytes = BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
            
            # Prepare API request
            api_url = f"{self.gs2_api_url}/infer"
            payload = {
                "image_base64": img_base64,
                "text": self.gs2_text_prompt,
                "box_thr": self.gs2_box_thr,
                "text_thr": self.gs2_text_thr,
            }
            
            # Make API call
            if self.gs2_verbose:
                cprint(f"[GM2-API] Calling Grounded-SAM-2 API...", "cyan")
            response = requests.post(api_url, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_msg = response.json().get("error", f"HTTP {response.status_code}")
                if self.gs2_verbose:
                    cprint(f"[error] Grounded-SAM-2 API error: {error_msg}", "red")
                return np.zeros((3, 512), dtype=np.float32)
            
            # Parse response
            mask_json = response.json()
            
            # Build attn_3d from mask
            attn_3d = self._build_attn_from_mask(
                point_cloud_with_uv, mask_json, img_res=img_res, n_points=512, n_channels=3
            )
            
            return attn_3d
            
        except requests.exceptions.RequestException as e:
            cprint(f"[warn] Grounded-SAM-2 API request failed: {e}", "yellow")
            cprint(f"[warn] Falling back to subprocess mode...", "yellow")
            return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)
        except Exception as e:
            cprint(f"[warn] Failed to generate attn_3d via API: {e}", "yellow")
            return np.zeros((3, 512), dtype=np.float32)
    
    def _generate_attn_3d_via_subprocess(self, rgb_img, point_cloud_with_uv, img_res=(84, 84)):
        """
        Generate attn_3d using subprocess (original method, slower due to model reload).
        """
        # Save temporary image
        temp_img_path = os.path.join(self.temp_dir, f"temp_img_{os.getpid()}_{np.random.randint(0, 1000000)}.png")
        temp_json_path = temp_img_path.replace(".png", ".json")
        
        try:
            # Save image
            imageio.imwrite(temp_img_path, rgb_img)
            
            # Get absolute paths for script and checkpoints
            script_path = os.path.abspath(self.gs2_script_path)
            # Script is now in Grounded-SAM-2 directory
            # script_path: .../AEDP3/Grounded-SAM-2/infer_grounded_sam2_single.py
            gs2_root = os.path.dirname(script_path)  # Grounded-SAM-2 directory
            
            # Verify path exists
            if not os.path.exists(gs2_root):
                cprint(f"[warn] Grounded-SAM-2 not found at {gs2_root}", "yellow")
                return np.zeros((3, 512), dtype=np.float32)
            
            # Build command to run in aedp3_vis environment
            # Use conda run to execute in the correct environment
            # Note: conda run requires conda to be in PATH
            # Note: --no-capture-output may not be available in older conda versions
            cmd = [
                "conda", "run", "-n", self.gs2_conda_env,
                "python", script_path,
                "--img_path", temp_img_path,
                "--output_json", temp_json_path,
                "--text", self.gs2_text_prompt,
                "--sam2_ckpt", self.gs2_sam2_ckpt,
                "--sam2_cfg", self.gs2_sam2_cfg,
                "--gdino_cfg", self.gs2_gdino_cfg,
                "--gdino_ckpt", self.gs2_gdino_ckpt,
                "--device", self.gs2_device,
                "--box_thr", str(self.gs2_box_thr),
                "--text_thr", str(self.gs2_text_thr),
            ]
            
            # Run Grounded-SAM-2 inference with real-time output
            # Set cwd to gs2_root so relative paths in script work correctly
            if self.gs2_verbose:
                cprint(f"[GM2] Running Grounded-SAM-2 inference (subprocess mode)...", "cyan")
            
            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                cwd=gs2_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            
            # Read output line by line and print (filter warnings)
            import time
            import re
            
            stdout_lines = []
            timeout_seconds = 60
            start_time = time.time()
            
            # Patterns to filter out
            warning_patterns = [
                r"UserWarning:",
                r"Triggered internally",
                r"Falling back to",
                r"Memory efficient",
                r"Flash attention",
                r"CuDNN attention",
                r"Expected query, key and value",
                r"torch\.meshgrid",
                r"torch\.utils\.checkpoint",
                r"requires_grad=True",
                r"NumPy array is not writable",
            ]
            
            def should_print_line(line):
                """Check if line should be printed (not a warning)."""
                if not line.strip():
                    return False
                # Filter out warning lines
                for pattern in warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        return False
                return True
            
            try:
                # Read lines until process finishes
                for line in process.stdout:
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                    
                    line = line.rstrip()
                    if should_print_line(line):  # Only print non-warning lines
                        if self.gs2_verbose:
                            cprint(f"[GM2] {line}", "cyan")
                        stdout_lines.append(line)
                
                # Wait for process to finish
                process.wait(timeout=max(1, timeout_seconds - (time.time() - start_time)))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                cprint(f"[error] Grounded-SAM-2 inference timeout", "red")
                return np.zeros((3, 512), dtype=np.float32)
            
            if process.returncode != 0:
                cprint(f"[error] Grounded-SAM-2 inference failed with return code {process.returncode}", "red")
                # Return zero attention field as fallback
                return np.zeros((3, 512), dtype=np.float32)
            
            # Load JSON result
            if not os.path.exists(temp_json_path):
                cprint(f"[warn] Grounded-SAM-2 output JSON not found: {temp_json_path}", "yellow")
                return np.zeros((3, 512), dtype=np.float32)
            
            with open(temp_json_path, "r") as f:
                mask_json = json.load(f)
            
            # Build attn_3d from mask
            attn_3d = self._build_attn_from_mask(
                point_cloud_with_uv, mask_json, img_res=img_res, n_points=512, n_channels=3
            )
            
            return attn_3d
            
        except subprocess.TimeoutExpired:
            cprint(f"[warn] Grounded-SAM-2 inference timeout", "yellow")
            return np.zeros((3, 512), dtype=np.float32)
        except Exception as e:
            cprint(f"[warn] Failed to generate attn_3d: {e}", "yellow")
            return np.zeros((3, 512), dtype=np.float32)
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)
            except Exception:
                pass

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        


        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
            
            # Reset attn history for each episode
            self.attn_history.clear()
                
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    # Extract only first 6 channels (xyzrgb) if point cloud has 8 channels (xyzrgbuv)
                    # This ensures compatibility with normalizer which expects 6 channels (trained on 6-channel data)
                    point_cloud = obs_dict['point_cloud']
                    if point_cloud.shape[-1] >= 8:
                        point_cloud = point_cloud[..., :6]  # Keep only xyzrgb, remove UV
                    obs_dict_input['point_cloud'] = point_cloud.unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    
                    # Check if policy expects attn_3d
                    # NOTE: This is fully compatible with original methods that don't use attn_3d.
                    # If policy doesn't need attn_3d, we skip all attn_3d generation logic.
                    needs_attn_3d = False
                    if hasattr(policy, 'shape_meta') and 'attn_3d' in policy.shape_meta.get('obs', {}):
                        needs_attn_3d = True
                    elif hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'use_attn_3d') and policy.obs_encoder.use_attn_3d:
                        needs_attn_3d = True
                    
                    if needs_attn_3d:
                        # Only generate attn_3d if policy requires it
                        # Check if environment provides attn_3d
                        if 'attn_3d' in obs_dict:
                            # Environment provides attn_3d, use it
                            obs_dict_input['attn_3d'] = obs_dict['attn_3d'].unsqueeze(0)
                        else:
                            # Generate attn_3d during inference
                            # Generate attn_3d during inference using Grounded-SAM-2
                            # Get RGB image by rendering from environment
                            # Navigate through wrapper chain to get to AdroitEnv
                            base_env = env.env
                            while hasattr(base_env, 'env'):
                                base_env = base_env.env
                            
                            # Render RGB image from environment
                            rgb_img = None
                            try:
                                rgb_img = base_env.render('rgb_array')  # (H, W, 3), uint8
                            except Exception as e:
                                cprint(f"[warn] Failed to render RGB image: {e}, using fallback", "yellow")
                            
                            # Get point cloud with UV (before truncating to 6 channels)
                            # We need to get the full point cloud with UV from the wrapper
                            point_cloud_full = None
                            try:
                                # Navigate to MujocoPointcloudWrapperAdroit
                                pc_wrapper = env.env  # This should be MujocoPointcloudWrapperAdroit
                                if hasattr(pc_wrapper, 'pc_generator'):
                                    # Get fresh point cloud with UV (8 channels)
                                    pc_full, _ = pc_wrapper.get_point_cloud(use_RGB=True)
                                    # Check if it has UV (8 channels) - our modified generateCroppedPointCloud returns 8 channels
                                    if pc_full.shape[-1] >= 8:
                                        # Stack to match T dimension (use same point cloud for all timesteps)
                                        point_cloud_full = np.stack([pc_full] * self.n_obs_steps, axis=0)  # (T, N, 8)
                                    else:
                                        cprint(f"[warn] Point cloud from wrapper doesn't have UV (shape: {pc_full.shape})", "yellow")
                            except Exception as e:
                                cprint(f"[warn] Failed to get point cloud with UV from wrapper: {e}", "yellow")
                            
                            # If we couldn't get point cloud with UV, check if obs_dict has it (before truncation)
                            if point_cloud_full is None:
                                # Check original observation before truncation
                                pc_orig = np_obs_dict.get('point_cloud')
                                if pc_orig is not None and pc_orig.shape[-1] >= 8:
                                    point_cloud_full = pc_orig  # (T, N, 8)
                                else:
                                    cprint(f"[warn] Cannot get point cloud with UV, using fallback zero attention", "yellow")
                                    point_cloud_full = None
                            
                            # Get segmentation data if using env-based segmentation
                            seg_data = None
                            if self.seg_type == 'env':
                                # Get segmentation data from observation
                                seg_data = np_obs_dict.get('segmentation')
                                if seg_data is None:
                                    cprint(f"[warn] seg_type='env' but no segmentation data in observation, using GS2 fallback", "yellow")
                                    self.seg_type = 'gs2'  # Temporarily fall back to GS2
                                else:
                                    # Mirror/flip segmentation vertically to match training data orientation
                                    # Training pipeline saved segmentation with a vertical flip (seg = seg[::-1, :, :]).
                                    try:
                                        if isinstance(seg_data, np.ndarray):
                                            if seg_data.ndim == 4:
                                                # (T, H, W, C)
                                                seg_data = seg_data[:, ::-1, :, :]
                                            elif seg_data.ndim == 3:
                                                # (H, W, C)
                                                seg_data = seg_data[::-1, :, :]
                                            # put back into np_obs_dict for consistency downstream
                                            np_obs_dict['segmentation'] = seg_data
                                    except Exception as e:
                                        cprint(f"[warn] Failed to flip segmentation for orientation: {e}", "yellow")

                            # Generate attn_3d
                            if rgb_img is not None and point_cloud_full is not None:
                                # Check if we can use environment segmentation
                                if self.seg_type == 'env' and seg_data is not None:
                                    # Use environment segmentation data - process current frame only
                                    # Get the current frame's data
                                    pc_current = point_cloud_full[-1]  # (N, 8) - most recent
                                    seg_current = seg_data[-1]  # (H, W, C) - most recent

                                    # Build attention for current frame
                                    attn_current = self._build_attn_from_env_seg(
                                        np.expand_dims(pc_current, 0), 
                                        np.expand_dims(seg_current, 0), 
                                        n_points=512, n_channels=3
                                    )
                                    attn_current = attn_current.squeeze(0)  # (C, N)

                                    # Store in history buffer
                                    self.attn_history.append(attn_current)

                                    # Build window from history (pad with zeros if not enough history)
                                    attn_list = list(self.attn_history)
                                    n_missing = self.n_obs_steps - len(attn_list)
                                    if n_missing > 0:
                                        padding = [np.zeros((3, 512), dtype=np.float32) for _ in range(n_missing)]
                                        attn_list = padding + attn_list

                                    attn_3d = np.stack(attn_list, axis=0)  # (T, C, N)

                                    # Check if attention is all zeros (keep warning)
                                    if attn_3d.sum() == 0:
                                        cprint(f"[warn] Attention mask is all zeros! seg_data shape: {seg_data.shape}, pc shape: {point_cloud_full.shape}", "red")
                                else:
                                    # Generate attn_3d for current frame using Grounded-SAM-2 (1 call per step)
                                    # Get the current frame's point cloud (latest timestep)
                                    pc_current = point_cloud_full[-1]  # (N, 8) - most recent

                                    # If point cloud doesn't have UV, we can't do precise mapping
                                    if pc_current.shape[-1] < 8:
                                        cprint(f"[warn] Point cloud at current frame doesn't have UV coordinates (shape: {pc_current.shape}), using fallback", "yellow")
                                        # Fallback: generate zero attention
                                        attn_current = np.zeros((3, 512), dtype=np.float32)
                                    else:
                                        # Use Grounded-SAM-2 (1 call per step instead of T calls)
                                        attn_current = self._generate_attn_3d_inference(
                                            rgb_img, pc_current, img_res=rgb_img.shape[:2], seg_data=None
                                        )

                                    # Store in history buffer
                                    self.attn_history.append(attn_current)

                                    # Build window from history (pad with zeros if not enough history)
                                    attn_list = list(self.attn_history)
                                    n_missing = self.n_obs_steps - len(attn_list)
                                    if n_missing > 0:
                                        padding = [np.zeros((3, 512), dtype=np.float32) for _ in range(n_missing)]
                                        attn_list = padding + attn_list

                                    attn_3d = np.stack(attn_list, axis=0)  # (T, C, N)
                            else:
                                # Fallback: generate zero attention
                                cprint(f"[warn] Cannot generate attn_3d (rgb_img={rgb_img is not None}, pc_full={point_cloud_full is not None}), using zero attention", "yellow")
                                attn_3d = np.zeros((self.n_obs_steps, 3, 512), dtype=np.float32)
                            
                            # Convert to torch and add to obs_dict_input
                            obs_dict_input['attn_3d'] = torch.from_numpy(attn_3d).to(device=device, dtype=dtype).unsqueeze(0)  # (1, T, C, N)
                    
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                # all_goal_achieved.append(info['goal_achieved']
                num_goal_achieved += np.sum(info['goal_achieved'])
                done = np.all(done)
                actual_step_count += 1

            all_success_rates.append(info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)


        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        del env
        
        # Clean up temp directory
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass

        return log_data
