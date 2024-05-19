import os
import sys
import json
from open_clip import get_tokenizer
from open_clip.transform import image_transform_v2, PreprocessCfg
import numpy as np
import torch
from PIL.Image import Image as PILImage
import onnxruntime
from functools import cached_property
from .clip2onnx import CLIPConverter
from .utils import get_weight_path, get_config_path, insert_quant_to_path
from .config import DEFAULT_EXPORT

module_paths = [
    os.path.abspath(os.path.join('..', 'logger_config.py')),
    os.path.abspath(os.path.join('..', 'logger_utils.py'))
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from logger_config import LOAD_MODEL_LOGGING_LEVEL, LOGS_PATH
from logger_utils import log_execution_time, get_logger


logger = get_logger(f"{__name__}:load", LOAD_MODEL_LOGGING_LEVEL, LOGS_PATH)


class ONNXCLIP(CLIPConverter):
    @log_execution_time(logger, "[CLIP ONNX] Initialization")
    def __init__(self, 
                 model_name: str = "ViT-B-32", 
                 pretrained: str = "laion2b_s34b_b79k", 
                 providers: tuple[str] = tuple(['CPUExecutionProvider']),
                 quantized: bool = False
                 ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.providers = providers
        self.quantized = quantized
        self.load_model(self.model_name, self.pretrained, quantized=quantized)

    def load_model(self, 
                   model_name: str, 
                   pretrained: str, 
                   quantized: bool = False
                   ) -> None:
        self.visual_weight_path = get_weight_path(model_name, pretrained, "visual")
        self.textual_weight_path = get_weight_path(model_name, pretrained, "textual")
        config_path = get_config_path(model_name, pretrained)
        if not os.path.exists(self.visual_weight_path) or \
                not os.path.exists(self.textual_weight_path) or \
                not os.path.exists(config_path):
            self.visual, self.textual = self.load_torch_model(model_name, pretrained)
            self.onnx_export_visual(
                self.visual_weight_path, export_params=DEFAULT_EXPORT)
            self.onnx_export_textual(
                self.textual_weight_path, export_params=DEFAULT_EXPORT)
            self.export_config(config_path)
        else:
            with open(config_path) as f:
                cfg = json.load(f)
            assert cfg["model_name"] == self.model_name and \
                 cfg["pretrained"] == self.pretrained
            self.exp_logit_scale = cfg["exp_logit_scale"]
            self.image_size = cfg["preprocess_cfg"]["size"]
            preprocess_cfg = PreprocessCfg(**cfg["preprocess_cfg"])
            self.preprocess = image_transform_v2(preprocess_cfg, False)
            self.tokenizer = get_tokenizer(self.model_name)
        
        if quantized:
            self.visual_weight_path = self.onnx_dynamic_quantization(
                model_path=self.visual_weight_path, overwrite=False)
            self.textual_weight_path = self.onnx_dynamic_quantization(
                model_path=self.textual_weight_path, overwrite=False)
            logger.debug("[CLIP ONNX] Loading qunatized model")
    
    @cached_property
    def visual_session(self) -> None:
        return onnxruntime.InferenceSession(
            self.visual_weight_path, providers=self.providers)

    @cached_property
    def textual_session(self):
        return onnxruntime.InferenceSession(
            self.textual_weight_path, providers=self.providers)
    
    def encode_image(self, x: np.ndarray) -> np.ndarray:
        arg_name = self.visual_session.get_inputs()[0].name
        output, *_ = self.visual_session.run(None, {arg_name: x})
        return output
    
    def encode_text(self, x: np.ndarray) -> np.ndarray:
        arg_name = self.textual_session.get_inputs()[0].name
        output, *_ = self.textual_session.run(None, {arg_name: x})
        return output

    @log_execution_time(logger, "[CLIP ONNX] Inference of image")
    def get_image_emb(self, images: list[PILImage]) -> np.ndarray:
        images = torch.cat(
            [self.preprocess(image).unsqueeze(0) for image in images], dim=0).cpu()
        images_onnx = images.detach().cpu().numpy().astype(np.float32)
        images_features = self.encode_image(images_onnx)
        norm_image_features = images_features / np.linalg.norm(images_features, ord=2, 
                                                              axis=1, keepdims=True)
        return norm_image_features

    @log_execution_time(logger, "[CLIP ONNX] Inference of text")
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