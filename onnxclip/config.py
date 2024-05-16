from torch.onnx import ExportOptions

IMAGES_BATCH_SIZE = 1
DEFAULT_EXPORT = ExportOptions(dynamic_shapes=False)