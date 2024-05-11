import numpy as np
import onnxruntime
import clip
from PIL import Image
from clip_onnx import clip_onnx, attention
from clip_onnx.utils import DEFAULT_EXPORT
import torch


DEFAULT_EXPORT["opset_version"] = 15


class CLIP:
    
    def __init__(self, model_name="ViT-B/32") -> None:
        self.load_model(model_name)

    def load_model(self, model_name) -> None:
        self.visual_path = "clip_visual.onnx"
        self.textual_path = "clip_textual.onnx"
        self.model, self.preprocess = clip.load(model_name, device="cpu", jit=False)
        self.onnx_model = clip_onnx(self.model, visual_path=self.visual_path, textual_path=self.textual_path)
        self.onnx_model.convert2onnx(visual_input=torch.zeros((1, 3, 224, 224)).to(torch.float32), textual_input=torch.zeros((1, 77)).to(torch.int32), verbose=True)
        self.onnx_model.start_sessions(providers=["CPUExecutionProvider"])

    def get_image_emb(self, image: Image) -> np.ndarray:
        image = self.preprocess(image).unsqueeze(0).cpu()
        image_onnx = image.detach().cpu().numpy().astype(np.float32)
        image_features = self.onnx_model.encode_image(image_onnx)
        norm_image_features = image_features / np.linalg.norm(image_features, ord=2, 
                                                              axis=1, keepdims=True)
        return norm_image_features

    def get_prompt_emb(self, prompt: str) -> np.ndarray:
        text = clip.tokenize(prompt).cpu()
        text_onnx = text.detach().cpu().numpy().astype(np.int32)
        text_features = self.onnx_model.encode_text(text_onnx)
        norm_text_features = text_features / np.linalg.norm(text_features, ord=2, 
                                                            axis=1, keepdims=True)
        return norm_text_features