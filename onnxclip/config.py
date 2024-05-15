from torch.onnx import ExportOptions

BATCH_SIZE = 1
DEFAULT_EXPORT = ExportOptions(dynamic_shapes=False)