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
import time
import re
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 num_distraction_objects=0
                 ):
        super().__init__(output_dir)
        self.task_name = task_name


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device,
                                 use_point_crop=use_point_crop, num_points=num_points,
                                 num_distraction_objects=num_distraction_objects)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        # Grounded-SAM-2 config (for realtime attn generation)
        self.gs2_api_url = os.getenv("GS2_API_URL", "http://127.0.0.1:5000")
        self.gs2_verbose = os.getenv("GS2_VERBOSE", "0").lower() in ("1", "true", "yes")
        self.gs2_conda_env = "aedp3_vis"
        self.gs2_text_prompt = self._get_text_prompt_for_task(task_name)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.gs2_script_path = os.path.join(project_root, "Grounded-SAM-2", "infer_grounded_sam2_single.py")
        self.gs2_sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"
        self.gs2_sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.gs2_gdino_cfg = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.gs2_gdino_ckpt = "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.gs2_device = "cuda"
        self.gs2_box_thr = 0.35
        self.gs2_text_thr = 0.25
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

        # Buffer to store historical attn_3d (for efficient GS2 usage)
        # Each step generates 1 attn, and we take the last n_obs_steps for the window
        self.attn_history = collections.deque(maxlen=n_obs_steps)

    def _get_text_prompt_for_task(self, task_name):
        """Get text prompt for Grounded-SAM-2 based on task name."""
        # Mapping kept consistent with scripts/make_metaworld_datasets.sh
        # prompts = {
        #     'pick-place': 'red robotic arm. a little red rectangular prism.',
        #     'shelf-place': 'red robotic arm. a little blue rectangular prism. shelf.',
        #     'soccer': 'red robotic arm. soccer. soccer goal.',
        #     'stick-pull': 'red robotic arm. blue stick. gray thermos.',
        #     'stick-push': 'red robotic arm. blue stick. gray thermos.',
        #     'sweep': 'red robotic arm. a little brown rectangular prism.',
        #     'sweep-into': 'red robotic arm. a little brown rectangular prism.',
        #     'window-close': 'red robotic arm. window.',
        #     'window-open': 'red robotic arm. window.',
        #     'hammer': 'red robotic arm. a hammer with a gray head and a red handle.',
        #     'box-close': 'red robotic arm. box.',
        #     'bin-picking': 'red robotic arm. bin.',
        #     'disassemble': 'red robotic arm. disassemble.',
        #     'reach': 'red robotic arm.'
        # }

        prompts = {
            'pick-place': 'a little red rectangular prism.',
            'shelf-place': 'a little blue rectangular prism. shelf.',
            'soccer': 'soccer. soccer goal.',
            'stick-pull': 'blue stick. gray thermos.',
            'stick-push': 'blue stick. gray thermos.',
            'sweep': 'a little rectangular prism.',
            'sweep-into': 'a little brown rectangular prism.',
            'window-close': 'window.',
            'window-open': 'window.',
            'hammer': 'a hammer with a gray head and a red handle. nail',
            'box-close': 'lid. box.',
            'bin-picking': 'bin. a little green rectangular prism.',
            'disassemble': 'a ring with handle.',
            'reach': 'red robotic arm.',
            # Default fallback
            'pick-place-wall': 'a little red rectangular prism. wall.',
            'push': 'a little rectangular prism.',
            'push-back': 'a little rectangular prism.',
            'pick-out-of-hole': 'hole. a little rectangular prism.',
            'hand-insert': 'hand. rectangular prism.',
            'assembly': 'assembly. peg. ring.',
            'push-wall': 'a little rectangular prism. wall.',
            'peg-insert-side': 'peg. hole.',
            # More task prompts (semantically consistent with make_metaworld_datasets.sh)
            'dial-turn': 'dial. knob.',
            'door-lock': 'door. lock.',
            'handle-pull': 'handle.',
            'handle-pull-side': 'handle.',
            'lever-pull': 'lever.',
            'reach-wall': 'red robotic arm.',
            'peg-unplug-side': 'peg. hole.',
            'coffee-pull': 'coffee.',
            'coffee-push': 'coffee.'
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
        if point_cloud.shape[0] >= n_points:
            idx = np.random.choice(point_cloud.shape[0], n_points, replace=False)
            pc = point_cloud[idx]
        else:
            pc = np.zeros((n_points, point_cloud.shape[1]), dtype=point_cloud.dtype)
            pc[: point_cloud.shape[0]] = point_cloud

        xyz = pc[:, :3]
        if pc.shape[1] >= 8:
            u_norm = pc[:, 6]
            v_norm = pc[:, 7]
        else:
            u_norm = np.zeros(pc.shape[0], dtype=np.float32)
            v_norm = np.zeros(pc.shape[0], dtype=np.float32)

        u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
        v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)

        attn = np.zeros((n_channels, n_points), dtype=np.float32)
        mask_hit = np.zeros(n_points, dtype=bool)

        if mask_json and "annotations" in mask_json and len(mask_json["annotations"]) > 0:
            for ann in mask_json["annotations"]:
                rle = ann["segmentation"]
                if isinstance(rle, dict) and "counts" in rle:
                    try:
                        mask = mask_util.decode(rle).astype(bool)
                        mask_hit |= mask[v_pix, u_pix]
                    except Exception:
                        continue

        attn[0] = mask_hit.astype(np.float32)
        x_center = xyz[:, 0].mean()
        x_dist = np.abs(xyz[:, 0] - x_center)
        x_dist_norm = x_dist / (x_dist.max() + 1e-6)
        attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)
        attn[2] = (1.0 - mask_hit.astype(np.float32))
        return attn

    def _generate_attn_3d_inference(self, rgb_img, point_cloud_with_uv, img_res=(84, 84)):
        """Generate attn_3d during inference by calling Grounded-SAM-2 (API preferred)."""
        try:
            import requests
            HAS_REQUESTS = True
        except Exception:
            HAS_REQUESTS = False

        if self.gs2_api_url and HAS_REQUESTS:
            try:
                from PIL import Image
                img_pil = Image.fromarray(rgb_img)
                img_bytes = BytesIO()
                img_pil.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                api_url = f"{self.gs2_api_url}/infer"
                payload = {
                    "image_base64": img_base64,
                    "text": self.gs2_text_prompt,
                    "box_thr": self.gs2_box_thr,
                    "text_thr": self.gs2_text_thr,
                }
                if self.gs2_verbose:
                    cprint(f"[GM2-API] Calling Grounded-SAM-2 API...", "cyan")
                response = requests.post(api_url, json=payload, timeout=60)
                if response.status_code != 200:
                    return np.zeros((3, 512), dtype=np.float32)
                mask_json = response.json()
                attn_3d = self._build_attn_from_mask(point_cloud_with_uv, mask_json, img_res=img_res, n_points=512, n_channels=3)
                return attn_3d
            except Exception:
                return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)
        else:
            return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)

    def _generate_attn_3d_via_subprocess(self, rgb_img, point_cloud_with_uv, img_res=(84,84)):
        try:
            temp_img_path = os.path.join(self.temp_dir, f"temp_img_{os.getpid()}_{np.random.randint(0,1000000)}.png")
            temp_json_path = temp_img_path.replace(".png", ".json")
            imageio.imwrite(temp_img_path, rgb_img)
            script_path = os.path.abspath(self.gs2_script_path)
            gs2_root = os.path.dirname(script_path)
            if not os.path.exists(gs2_root):
                return np.zeros((3,512), dtype=np.float32)
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
            process = subprocess.Popen(cmd, cwd=gs2_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            stdout_lines = []
            timeout_seconds = 60
            start_time = time.time()
            warning_patterns = [r"UserWarning:", r"Triggered internally", r"Falling back to", r"Memory efficient"]
            def should_print_line(line):
                if not line.strip():
                    return False
                for pattern in warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        return False
                return True
            try:
                for line in process.stdout:
                    if time.time() - start_time > timeout_seconds:
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                    line = line.rstrip()
                    if should_print_line(line):
                        if self.gs2_verbose:
                            cprint(f"[GM2] {line}", "cyan")
                        stdout_lines.append(line)
                process.wait(timeout=max(1, timeout_seconds - (time.time() - start_time)))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return np.zeros((3,512), dtype=np.float32)
            if process.returncode != 0:
                return np.zeros((3,512), dtype=np.float32)
            if not os.path.exists(temp_json_path):
                return np.zeros((3,512), dtype=np.float32)
            with open(temp_json_path, "r") as f:
                mask_json = json.load(f)
            attn_3d = self._build_attn_from_mask(point_cloud_with_uv, mask_json, img_res=img_res, n_points=512, n_channels=3)
            return attn_3d
        except Exception:
            return np.zeros((3,512), dtype=np.float32)
        finally:
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)
            except Exception:
                pass

    def run(self, policy: BasePolicy, save_video=True):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # Reset attn history for each episode
            self.attn_history.clear()
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    # basic fields
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)

                    # Check if policy expects attn_3d
                    needs_attn_3d = False
                    if hasattr(policy, 'shape_meta') and 'attn_3d' in policy.shape_meta.get('obs', {}):
                        needs_attn_3d = True
                    elif hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'use_attn_3d') and policy.obs_encoder.use_attn_3d:
                        needs_attn_3d = True

                    if needs_attn_3d:
                        # If environment already provides attn_3d in obs, use it
                        if 'attn_3d' in np_obs_dict:
                            attn_3d = np_obs_dict['attn_3d']  # (T, C, N) or (C, N)
                            # ensure shape (1, T, C, N)
                            if attn_3d.ndim == 2:
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, C, N)
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, 1, C, N)
                            elif attn_3d.ndim == 3:
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, T, C, N)
                            obs_dict_input['attn_3d'] = torch.from_numpy(attn_3d).to(device=device, dtype=dtype)
                        else:
                            # Generate attn_3d on the fly using Grounded-SAM-2 (API preferred)
                            rgb_img = None
                            try:
                                # try to render rgb image from env
                                base_env = env.env
                                while hasattr(base_env, 'env'):
                                    base_env = base_env.env
                                rgb_img = base_env.render('rgb_array')
                            except Exception:
                                rgb_img = None

                            # Try to get full point cloud (with UV) from wrapper if available
                            point_cloud_full = None
                            try:
                                base_wrapper = env.env
                                if hasattr(base_wrapper, 'pc_generator'):
                                    pc_full, _ = base_wrapper.get_point_cloud(use_RGB=True)
                                    if pc_full.shape[-1] >= 8:
                                        point_cloud_full = np.stack([pc_full] * self.n_obs_steps, axis=0)
                            except Exception:
                                point_cloud_full = None

                            # Fallback to check original obs dict for point_cloud with UV
                            if point_cloud_full is None:
                                pc_orig = np_obs_dict.get('point_cloud')
                                if pc_orig is not None and pc_orig.shape[-1] >= 8:
                                    point_cloud_full = pc_orig

                            # Generate attn_3d for timesteps if possible
                            if rgb_img is not None and point_cloud_full is not None:
                                # Get the current frame's point cloud (latest timestep)
                                # point_cloud_full shape: (T, N, 8) where T is n_obs_steps
                                pc_current = point_cloud_full[-1]  # (N, 8) - most recent frame

                                if pc_current.shape[-1] < 8:
                                    # No UV coordinates, generate zero attention
                                    attn_current = np.zeros((3, 512), dtype=np.float32)
                                else:
                                    # Generate attn for current frame only (1 GS2 call per step)
                                    attn_current = self._generate_attn_3d_inference(
                                        rgb_img, pc_current, img_res=rgb_img.shape[:2]
                                    )

                                # Store in history buffer
                                self.attn_history.append(attn_current)

                                # Build window from history (pad with zeros if not enough history)
                                attn_list = list(self.attn_history)
                                n_missing = self.n_obs_steps - len(attn_list)
                                if n_missing > 0:
                                    # Pad with zeros at the beginning
                                    padding = [np.zeros((3, 512), dtype=np.float32) for _ in range(n_missing)]
                                    attn_list = padding + attn_list

                                attn_3d = np.stack(attn_list, axis=0)  # (T, C, N)
                            else:
                                attn_3d = np.zeros((self.n_obs_steps, 3, 512), dtype=np.float32)
                            obs_dict_input['attn_3d'] = torch.from_numpy(attn_3d).to(device=device, dtype=dtype).unsqueeze(0)

                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
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
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
