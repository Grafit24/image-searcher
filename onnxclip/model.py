import os
import numpy as np
from PIL.Image import Image as PILImage
import onnxruntime
from .clip2onnx import CLIPConverter
from .utils import get_weight_path, DEFAULT_EXPORT


class ONNXCLIP(CLIPConverter):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.visual, self.textual = self.load_torch_model(model_name, pretrained)
        visual_weight_path = get_weight_path(model_name, pretrained, "visual")
        textual_weight_path = get_weight_path(model_name, pretrained, "textual")
        self.textual_session = None
        self.visual_session = None
        if not os.path.exists(visual_weight_path) or not os.path.exists(textual_weight_path):
            self.onnx_export_visual(
                visual_weight_path, export_params=DEFAULT_EXPORT)
            self.onnx_export_textual(
                textual_weight_path, export_params=DEFAULT_EXPORT)
        self.start_session(visual_weight_path, textual_weight_path)
    
    def start_session(self, visual_path: str, textual_path: str, providers: tuple[str] = tuple(['CPUExecutionProvider'])) -> None:
        self.visual_session = onnxruntime.InferenceSession(visual_path, providers=providers)
        self.textual_session = onnxruntime.InferenceSession(textual_path, providers=providers)
    
    def encode_image(self, x: np.ndarray) -> np.ndarray:
        arg_name = self.visual_session.get_inputs()[0].name
        output, *_ = self.visual_session.run(None, {arg_name: x})
        return output
    
    def encode_text(self, x: np.ndarray) -> np.ndarray:
        arg_name = self.textual_session.get_inputs()[0].name
        output, *_ = self.textual_session.run(None, {arg_name: x})
        return output

    def get_image_emb(self, image: PILImage) -> np.ndarray:
        image = self.preprocess(image).unsqueeze(0).cpu()
        image_onnx = image.detach().cpu().numpy().astype(np.float32)
        image_features = self.encode_image(image_onnx)
        norm_image_features = image_features / np.linalg.norm(image_features, ord=2, 
                                                              axis=1, keepdims=True)
        return norm_image_features

    def get_prompt_emb(self, prompt: str) -> np.ndarray:
        text = self.tokenizer(prompt).cpu()
        text_onnx = text.detach().cpu().numpy().astype(np.int32)
        text_features = self.encode_text(text_onnx)
        norm_text_features = text_features / np.linalg.norm(text_features, ord=2, 
                                                            axis=1, keepdims=True)
        return norm_text_features
    
    def cossim(self, prompts_emb: np.ndarray, images_emb: np.ndarray) -> np.ndarray:
        logits_per_image = self.exp_logit_scale * images_emb @ prompts_emb.T
        probs = self.softmax(logits_per_image)
        return probs

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)