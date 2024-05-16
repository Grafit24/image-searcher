import torch
import onnx
from torch import nn
from onnxruntime.quantization import quantize_dynamic, QuantType
import open_clip
from .textual_util import TextualWrapper
from .config import DEFAULT_EXPORT, IMAGES_BATCH_SIZE


class CLIPConverter:
    def load_torch_model(self, model_name: str, pretrained: str) -> tuple[nn.Module]:
        self.model_name = model_name
        self.pretrained = pretrained
        model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        self.image_size = model.visual.image_size
        # model components
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.exp_logit_scale = model.logit_scale.exp().item()
        visual = model.visual
        textual = TextualWrapper(model)
        return visual, textual
    
    def onnx_export_visual(self, out_path: str, export_params: dict = DEFAULT_EXPORT) -> onnx.ModelProto:
        dummy_input = torch.ones((IMAGES_BATCH_SIZE, 3, *self.image_size), dtype=torch.float32)
        visual_proto = self.onnx_export(
            self.visual, dummy_input, out_path, export_params)
        return visual_proto

    def onnx_export_textual(self, out_path: str, export_params: dict = DEFAULT_EXPORT) -> onnx.ModelProto:
        dummy_input = torch.ones((1, 77), dtype=torch.int32)
        textual_proto = self.onnx_export(
            self.textual, dummy_input, out_path, export_params)
        return textual_proto

    @staticmethod
    def onnx_check(model_path) -> None:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        del model

    @staticmethod
    def onnx_export(model: open_clip.model.CLIP, 
                    dummy_input: tuple[int], 
                    out_model_path: str, 
                    export_options: dict = DEFAULT_EXPORT
                    ) -> onnx.ModelProto:
        with torch.no_grad():
            onnx_program = torch.onnx.dynamo_export(
                model, dummy_input, export_options=export_options)
            onnx_program.save(out_model_path)
            onnx.checker.check_model(onnx_program.model_proto)
        return onnx_program.model_proto
