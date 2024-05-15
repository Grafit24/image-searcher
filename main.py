import os
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from onnxclip import ONNXCLIP
from glob import glob
from typing import Optional
from onnxclip import BATCH_SIZE


def main(dir_path: str, 
         prompt: str, 
         k: Optional[int] = None,
         threshold: float = 0,
         need_json: bool = False
        ) -> None:
    if not os.path.exists(dir_path):
        raise FileNotFoundError
    if not isinstance(prompt, str):
        raise TypeError
    
    model = ONNXCLIP()
    prompt_emb = model.get_prompt_emb(prompt)
    
    images_paths = glob(os.path.join(dir_path, "*.png"))
    images_emb = []
    bacth_images = []
    for idx, image_path in enumerate(images_paths, start=1):
        image = Image.open(image_path)
        bacth_images.append(image)
        if idx % BATCH_SIZE == 0 or len(images_paths) == idx:
            batch_emb = model.get_image_emb(bacth_images)
            images_emb.append(batch_emb)
            bacth_images = []

    if len(images_emb) == 0:
        raise Exception("No images .png found")
    
    images_emb = np.concatenate(images_emb, axis=0)
    probs = model.cossim(prompt_emb, images_emb)
    result = list(zip(images_paths, list(probs.flatten())))
    result = list(filter(lambda x: x[1] > threshold, result))
    result = list(reversed(sorted(result, key=lambda x: x[1])))
    result = result[:k] if k is not None else result
    if need_json:
        os.system('cls')
        print([path for path, _ in result])
    else:
        os.system('cls')
        for path, score in result:
            print(f"{path} : {score}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("-d", "--dir", default=os.getcwd(), type=str)
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=0)
    parser.add_argument("-j", "--json", action='store_true')
    args = parser.parse_args()
    main(args.dir.strip(), args.prompt.strip(), args.k, args.threshold, args.json)