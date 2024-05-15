import os


def get_clip_dir() -> str:
    clip_dir = os.path.dirname(os.path.abspath(__file__))
    return clip_dir


def weight_formater(model_name: str, pretrained: str, type: str) -> str:
    return f"clip_{type}_{model_name}_{pretrained}.onnx".lower().replace("-", "_")


def get_weight_path(model_name: str, pretrained: str, type: str) -> str:
    clip_dir = get_clip_dir()
    weights_dir = os.path.join(clip_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, weight_formater(model_name, pretrained, type))
    return weight_path