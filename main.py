import os
import json
from argparse import ArgumentParser
from glob import glob
from typing import Optional
import hashlib
import numpy as np
from pyexiv2 import ImageData
from PIL import Image
from PIL.Image import Image as PILImage
from onnxclip import ONNXCLIP
from logger_utils import log_execution_time, get_logger
from logger_config import INFERENCE_LOGGING_LEVEL, LOGS_PATH


logger = get_logger(f"{__name__}:inference", INFERENCE_LOGGING_LEVEL, LOGS_PATH)


class CacheProcessing:
    
    def __init__(self, files_paths: list[str], cache_dir: Optional[str] = None, use_cache: bool = True) -> None:
        self.files_paths = files_paths
        self.use_cache = use_cache

        if self.use_cache:
            self.cache_dir = cache_dir or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cache")
            self.index_fp = os.path.join(self.cache_dir, "index.json")

            # if cache doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            if not os.path.exists(self.index_fp):
                with open(self.index_fp, 'w') as f:
                    json.dump({}, f)
            
            with open(self.index_fp) as f:
                self.index = json.load(f)

    def decode_cache(self, fp: str) -> np.ndarray | None:
        if self.index.get(fp, False):
            return np.load(self.index[fp])
        return None

    def encode_cache(self, fp: str, embeddings: np.ndarray) -> None:
        name = self.unique_name_from_str(fp)
        fn = f"{name}.npy"
        np.save(arr=embeddings, file=os.path.join(self.cache_dir, f"{name}.npy"))
        self.index[fp] = os.path.join(self.cache_dir, fn)

    @log_execution_time(logger, "Processed images")
    def process(self, model: ONNXCLIP):
        images_embbedings = []
        for image_path in self.files_paths:
            img_emb = self.decode_cache(image_path) if self.use_cache else None
            if img_emb is not None:
                images_embbedings.append(img_emb)
            else:
                image = Image.open(image_path)
                img_emb = model.get_image_emb([image])
                if self.use_cache:
                    self.encode_cache(image_path, img_emb)
                images_embbedings.append(img_emb)

        if len(images_embbedings) == 0:
            raise Exception("No images .png found")
        
        if self.use_cache:
            self.save_index()
        
        images_embbedings = np.concatenate(images_embbedings, axis=0)
        return images_embbedings
    
    def save_index(self) -> None:
        with open(self.index_fp, "w") as f:
            json.dump(self.index, f)
    
    @staticmethod
    def unique_name_from_str(string: str, last_idx: int = 12) -> str:
        m = hashlib.md5()
        string = string.encode('utf-8')
        m.update(string)
        unqiue_name: str = str(int(m.hexdigest(), 16))[0:last_idx]
        return unqiue_name


@log_execution_time(logger, "The main module was executed")
def main(dir_path: str, 
         prompt: str, 
         k: Optional[int] = None,
         threshold: float = 0,
         need_json: bool = False,
         use_cache: bool = True,
         qunatized: bool = True
        ) -> None:
    if not os.path.exists(dir_path):
        raise FileNotFoundError
    if not isinstance(prompt, str):
        raise TypeError
    
    model = ONNXCLIP(quantized=qunatized)
    prompt_emb = model.get_prompt_emb(prompt)
    
    files_paths = glob(os.path.join(dir_path, "*.png"))
    cache_processing = CacheProcessing(files_paths, use_cache=use_cache)
    images_emb = cache_processing.process(model)

    probs = model.cossim(prompt_emb, images_emb)
    result = list(zip(files_paths, list(probs.flatten())))
    result = list(filter(lambda x: x[1] > threshold, result))
    result = list(reversed(sorted(result, key=lambda x: x[1])))
    result = result[:k] if k is not None else result
    if need_json:
        os.system('cls')
        print([path for path, _ in result])
    else:
        for path, score in result:
            print(f"{path} : {score}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("-d", "--dir", default=os.getcwd(), type=str)
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=0)
    parser.add_argument("-j", "--json", action='store_true')
    parser.add_argument("--no-cache", action='store_false')
    parser.add_argument("-nq", "--no-quantized", action='store_false')
    args = parser.parse_args()
    main(dir_path=args.dir.strip(), 
         prompt=args.prompt.strip(), 
         k=args.k, 
         threshold=args.threshold, 
         need_json=args.json, 
         use_cache=args.no_cache, 
         qunatized=args.no_quantized)