import ast
import json
import re
from pathlib import Path

from prompts import build_experience_prompt, build_generation_prompt


CURVE_START = "<curves>"
CURVE_END = "</curves>"


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def safe_filename(text, max_length=60):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    if not text:
        text = "item"
    return text[:max_length]


def load_prompts(path):
    data = load_json(path)
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item.strip())
        elif isinstance(item, dict):
            value = item.get("prompt") or item.get("problem") or item.get("theme")
            if value:
                prompts.append(str(value).strip())
    return [item for item in prompts if item]


def load_experiences(path):
    if not path:
        return []
    data = load_json(path, default=[])
    if isinstance(data, dict):
        values = list(data.values())
    else:
        values = data
    return [str(item).strip() for item in values if str(item).strip()]


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_list(text):
    text = strip_code_fence(text)
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Response does not contain a JSON list.")
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not a list.")
    return data


def normalize_experience(text):
    return " ".join(text.strip().lower().split())


def merge_experiences(existing, new_items, limit=None):
    merged = []
    seen = set()
    for item in list(existing) + list(new_items):
        clean = " ".join(str(item).strip().split())
        if not clean:
            continue
        key = normalize_experience(clean)
        if key in seen:
            continue
        seen.add(key)
        merged.append(clean)
    if limit is not None:
        merged = merged[:limit]
    return merged


def truncate_text(text, max_chars=12000):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def extract_curve_block(response):
    start = response.rfind(CURVE_START)
    end = response.rfind(CURVE_END)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Missing <curves>...</curves> block.")
    content = response[start + len(CURVE_START) : end].strip()
    if not content:
        raise ValueError("Empty <curves> block.")
    content = re.sub(r"#.*", "", content)
    return content


def validate_curves(curves):
    if not isinstance(curves, list):
        raise TypeError("Top-level curves object must be a list.")

    normalized = []
    for curve in curves:
        if not isinstance(curve, list):
            raise TypeError("Each curve must be a list.")
        if len(curve) != 4:
            raise ValueError("Each curve must contain exactly 4 control points.")

        normalized_curve = []
        for point in curve:
            if not isinstance(point, list):
                raise TypeError("Each control point must be a list.")
            if len(point) != 3:
                raise ValueError("Each control point must contain exactly 3 coordinates.")

            normalized_point = []
            for value in point:
                if not isinstance(value, (int, float)):
                    raise TypeError("Each coordinate must be numeric.")
                normalized_point.append(float(value))
            normalized_curve.append(normalized_point)
        normalized.append(normalized_curve)
    return normalized


def parse_curves(response):
    block = extract_curve_block(response)
    curves = ast.literal_eval(block)
    return validate_curves(curves)


def resolve_device(device):
    import torch

    if device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def save_rendered_images(images, output_dir):
    from torchvision.utils import save_image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu().clamp(0.0, 1.0)
    if images.dim() != 4:
        raise ValueError("Rendered images must be a 4D tensor.")

    for index, image in enumerate(images):
        save_image(image, output_dir / f"view_{index:02d}.png")


class BatchRenderer:
    def __init__(self, cameras_path, device=None, canvas_size=512, stroke_width=2.0):
        import torch

        try:
            import pydiffvg
        except ImportError as exc:
            raise ImportError("pydiffvg is required for rendering.") from exc

        self.torch = torch
        self.pydiffvg = pydiffvg
        self.device = resolve_device(device)
        self.canvas_size = canvas_size
        self.stroke_width = stroke_width
        self.poses, self.focal = self._load_cameras(cameras_path)

        self.pydiffvg.set_use_gpu(self.device.startswith("cuda") and torch.cuda.is_available())

    def _load_cameras(self, cameras_path):
        data = load_json(cameras_path)
        if not data:
            raise ValueError("Camera file is empty.")

        poses = []
        for item in data:
            pose = self.torch.eye(4, dtype=self.torch.float32, device=self.device)
            pose[:3, :3] = self.torch.tensor(item["rotation"], dtype=self.torch.float32, device=self.device)
            pose[:3, 3] = self.torch.tensor(item["position"], dtype=self.torch.float32, device=self.device)
            poses.append(pose)
        return poses, float(data[0]["fx"])

    def _project(self, points_3d, pose):
        intrinsic = self.torch.tensor(
            [
                [self.focal, 0.0, self.canvas_size / 2.0],
                [0.0, self.focal, self.canvas_size / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=self.torch.float32,
            device=self.device,
        )
        extrinsic = self.torch.linalg.inv(pose)
        points_h = self.torch.cat(
            [points_3d, self.torch.ones((points_3d.shape[0], 1), dtype=self.torch.float32, device=self.device)],
            dim=-1,
        )
        world = intrinsic @ extrinsic[:3, :]
        projected = points_h @ world.T
        return projected[:, :2] / projected[:, 2:].clamp(min=1e-6)

    def _render_view(self, curves, pose):
        num_control_points = self.torch.tensor([2], dtype=self.torch.int32)
        stroke_color = self.torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=self.torch.float32, device=self.device)

        shapes = []
        shape_groups = []
        for index, curve in enumerate(curves):
            points_3d = self.torch.tensor(curve, dtype=self.torch.float32, device=self.device)
            path = self.pydiffvg.Path(
                num_control_points=num_control_points,
                points=self._project(points_3d, pose),
                stroke_width=self.torch.tensor(self.stroke_width, dtype=self.torch.float32, device=self.device),
                is_closed=False,
            )
            shapes.append(path)
            shape_groups.append(
                self.pydiffvg.ShapeGroup(
                    shape_ids=self.torch.tensor([index], dtype=self.torch.int32),
                    fill_color=None,
                    stroke_color=stroke_color,
                )
            )

        scene_args = self.pydiffvg.RenderFunction.serialize_scene(
            self.canvas_size,
            self.canvas_size,
            shapes,
            shape_groups,
        )
        render = self.pydiffvg.RenderFunction.apply
        image = render(self.canvas_size, self.canvas_size, 2, 2, 0, None, *scene_args)
        alpha = image[..., 3:4]
        rgb = image[..., :3] * alpha + (1.0 - alpha)
        return rgb.permute(2, 0, 1).unsqueeze(0)

    def render(self, curves):
        if not curves:
            raise ValueError("Cannot render an empty curve list.")
        return self.torch.cat([self._render_view(curves, pose) for pose in self.poses], dim=0)


class CLIPReward:
    def __init__(
        self,
        cameras_path,
        device=None,
        clip_model_name="ViT-B/32",
        canvas_size=512,
        stroke_width=2.0,
    ):
        self.device = resolve_device(device)
        self.clip_model_name = clip_model_name
        self.renderer = BatchRenderer(
            cameras_path=cameras_path,
            device=self.device,
            canvas_size=canvas_size,
            stroke_width=stroke_width,
        )
        self._clip_module = None
        self._clip_model = None

    def _ensure_model(self):
        if self._clip_model is not None:
            return
        try:
            import clip
        except ImportError as exc:
            raise ImportError("CLIP is required. Install it from requirements.txt.") from exc

        self._clip_module = clip
        self._clip_model, _ = clip.load(self.clip_model_name, device=self.device, jit=False)
        self._clip_model.eval()

    def _prepare_images(self, images):
        import torch
        import torch.nn.functional as F

        if images.dim() != 4:
            raise ValueError("Rendered images must be a 4D tensor.")
        if images.shape[1] != 3:
            raise ValueError("Rendered images must be in NCHW format with 3 channels.")

        images = images.to(dtype=torch.float32)
        images = images.clamp(0.0, 1.0)

        resolution = 224
        if hasattr(self._clip_model, "visual") and hasattr(self._clip_model.visual, "input_resolution"):
            resolution = self._clip_model.visual.input_resolution
        if images.shape[-2:] != (resolution, resolution):
            images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device,
            dtype=images.dtype,
        ).view(1, 3, 1, 1)
        return (images - mean) / std

    def score_curves(self, curves, theme):
        self._ensure_model()
        images = self.renderer.render(curves)
        images = self._prepare_images(images)
        text = f"{theme}, minimal line drawing, white background, black lines."
        tokens = self._clip_module.tokenize([text]).to(self.device)

        with self.renderer.torch.no_grad():
            text_features = self._clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = self._clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T
            score = (similarity.clamp(-1, 1) + 1.0) / 2.0
        return float(score.mean().item())

    def score_response(self, response, theme):
        curves = parse_curves(response)
        return self.score_curves(curves, theme)


def generate_candidates(llm, theme, experiences, num_samples, temperature, max_tokens):
    prompt = build_generation_prompt(theme, experiences)
    candidates = []
    for index in range(num_samples):
        item = {
            "sample_id": index,
            "response": "",
            "valid": False,
            "reward": 0.0,
        }
        try:
            response = llm.chat(prompt, temperature=temperature, max_tokens=max_tokens)
            item["response"] = response
            item["curves"] = parse_curves(response)
            item["valid"] = True
        except Exception as exc:
            item["curves"] = []
            item["error"] = str(exc)
        candidates.append(item)
    return candidates


def score_candidates(candidates, scorer, theme):
    for item in candidates:
        if not item["valid"]:
            continue
        try:
            item["reward"] = scorer.score_curves(item["curves"], theme)
        except Exception as exc:
            item["valid"] = False
            item["reward"] = 0.0
            item["error"] = str(exc)
    return candidates


def valid_candidates(candidates):
    return [item for item in candidates if item.get("valid")]


def best_candidate(candidates):
    items = valid_candidates(candidates)
    if not items:
        return None
    return max(items, key=lambda item: item["reward"])


def contrast_pair(candidates):
    items = sorted(valid_candidates(candidates), key=lambda item: item["reward"])
    if len(items) < 2:
        return None, None
    return items[-1], items[0]


def extract_new_experiences(
    llm,
    theme,
    best,
    worst,
    experiences,
    max_new_experiences,
    max_tokens,
):
    if best is None or worst is None:
        return []
    if best["reward"] <= worst["reward"]:
        return []

    prompt = build_experience_prompt(
        theme=theme,
        best_response=truncate_text(best["response"]),
        best_score=best["reward"],
        worst_response=truncate_text(worst["response"]),
        worst_score=worst["reward"],
        experiences=experiences,
        max_new_experiences=max_new_experiences,
    )
    response = llm.chat(prompt, temperature=0.0, max_tokens=max_tokens)
    items = parse_json_list(response)
    return [str(item).strip() for item in items if str(item).strip()]
