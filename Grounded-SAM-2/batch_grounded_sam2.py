import os
import argparse
import json
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import random
from torchvision.ops import box_convert
import pycocotools.mask as mask_util

try:
    import requests
except ImportError:
    print("requests library is required. Install with: pip install requests")
    raise

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def single_mask_to_rle(mask: np.ndarray):
    """Convert binary mask (H, W) to RLE for json serialization."""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def run_on_image_via_api(
    img_path: str,
    text_prompt: str,
    api_url: str,
    box_thr: float,
    text_thr: float,
    multimask: bool = False,
    seed: int = 0,
) -> dict:
    """Run inference via HTTP API (faster, no model reload)."""
    try:
        # Prepare API request
        payload = {
            "image_path": os.path.abspath(img_path),
            "text": text_prompt,
            "box_thr": box_thr,
            "text_thr": text_thr,
            "multimask": multimask,
            "seed": int(seed),
        }
        
        # Make API call
        response = requests.post(f"{api_url}/infer", json=payload, timeout=60)
        
        if response.status_code != 200:
            error_msg = response.json().get("error", f"HTTP {response.status_code}")
            print(f"[ERROR] API error for {img_path}: {error_msg}")
            return {
                "image_path": img_path,
                "annotations": [],
            }
        
        # Parse response
        api_result = response.json()
        
        # Convert to expected format
        results = {
            "image_path": img_path,
            "annotations": api_result.get("annotations", []),
        }
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {img_path}: {e}")
        return {
            "image_path": img_path,
            "annotations": [],
        }


def run_on_image(
    img_path: str,
    text_prompt: str,
    sam2_checkpoint: str,
    sam2_config: str,
    gdino_config: str,
    gdino_checkpoint: str,
    device: str,
    box_thr: float,
    text_thr: float,
    multimask: bool = False,
    seed: int = 0,
) -> dict:
    # Set deterministic seeds for reproducibility per-image
    def set_seed(seed_val: int):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(int(seed))

    image_source, image = load_image(img_path)

    # GroundingDINO
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device,
    )
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_thr,
        text_threshold=text_thr,
        device=device,
    )

    # If no box detected, return empty result to avoid downstream SAM2 assertion failure
    if boxes.numel() == 0:
        return {
            "image_path": img_path,
            "annotations": [],
        }

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.set_image(image_source)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=multimask,
    )
    if multimask:
        best = np.argmax(scores, axis=1)
        masks = masks[np.arange(masks.shape[0]), best]

    # shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    results = {
        "image_path": img_path,
        "annotations": [],
    }
    for cls, box, mask, score in zip(labels, input_boxes.tolist(), masks, scores.tolist()):
        results["annotations"].append(
            {
                "class_name": cls,
                "bbox": box,
                "segmentation": single_mask_to_rle(mask),
                "score": score,
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True, help="export/adroit_door_frames")
    ap.add_argument("--output_root", required=True, help="export_gs2/adroit_door")
    ap.add_argument("--text", required=True, help='e.g. "door handle. door." (lowercase + dot)')
    ap.add_argument("--api_url", default="http://127.0.0.1:5000", 
                    help="GS2 API server URL (default: http://127.0.0.1:5000). "
                         "If provided, will use API instead of loading models directly.")
    ap.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument(
        "--gdino_cfg",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    ap.add_argument("--gdino_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--box_thr", type=float, default=0.35)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--multimask", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for deterministic inference (default: 0)")
    args = ap.parse_args()
    
    # Check if API mode should be used
    use_api = args.api_url is not None and args.api_url != ""
    
    if use_api:
        # Test API connection
        try:
            health_url = f"{args.api_url}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("models_loaded", False):
                    print(f"[batch_grounded_sam2] Using GS2 API at {args.api_url}")
                else:
                    print(f"[WARNING] GS2 API at {args.api_url} is running but models are not loaded!")
            else:
                print(f"[WARNING] GS2 API health check failed. Falling back to direct model loading.")
                use_api = False
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Cannot connect to GS2 API at {args.api_url}: {e}")
            print(f"[WARNING] Falling back to direct model loading.")
            use_api = False
    else:
        print("[batch_grounded_sam2] Using direct model loading (models will be reloaded for each image)")

    os.makedirs(args.output_root, exist_ok=True)

    ep_dirs = sorted(glob.glob(os.path.join(args.frames_root, "ep_*")))
    for ep_dir in ep_dirs:
        out_ep = os.path.join(args.output_root, os.path.basename(ep_dir))
        os.makedirs(out_ep, exist_ok=True)
        frame_paths = sorted(glob.glob(os.path.join(ep_dir, "frame_*.png")))
        for fp in frame_paths:
            out_json = os.path.join(out_ep, Path(fp).stem + ".json")
            if os.path.exists(out_json):
                continue
            
            if use_api:
                res = run_on_image_via_api(
                    fp,
                    args.text,
                    args.api_url,
                    args.box_thr,
                    args.text_thr,
                    multimask=args.multimask,
                    seed=args.seed,
                )
            else:
                res = run_on_image(
                    fp,
                    args.text,
                    args.sam2_ckpt,
                    args.sam2_cfg,
                    args.gdino_cfg,
                    args.gdino_ckpt,
                    args.device,
                    args.box_thr,
                    args.text_thr,
                    multimask=args.multimask,
                    seed=args.seed,
                )
            with open(out_json, "w") as f:
                json.dump(res, f)
        print(f"[done] {ep_dir} -> {out_ep}")


if __name__ == "__main__":
    main()

