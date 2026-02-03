from typing import Dict, List, Tuple, Optional, Union, Iterable
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None) -> np.ndarray:
    height, width = size
    resize_image = image.resize((width, height), resmaple=resample, reducing_gap=reducing_gap)
    return resize_image


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    return rescaled_image.astype(dtype)


def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image = (image - mean) / std
    return normalized_image


def process_images(
        images: List[Image.Image],
        size: Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    # convert each image to a np array
    images = [np.array(image) for image in images]
    # rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # move the channel dimension to the first dimension. The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len: int, image_token: int) -> str:
    # 引用自 Hugging Face 博客（https://huggingface.co/blog/paligemma#detailed-inference-process）：
    # 输入文本按常规方式分词（tokenized）。
    # 在开头添加一个 <bos> token，并在末尾追加一个额外的换行符 token（\n）。
    # 该换行符 token 是模型输入 prompt 中的关键组成部分（模型训练时即包含此结构），因此需显式添加。
    # 分词后的文本还会被前置固定数量的 <image> tokens。
    # 注意：根据论文，`\n` 应单独分词；但在 Hugging Face 的实现中，它被合并到前缀 prompt 中一起分词。
    # 参见 HF 实现代码：https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f3657
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)] # these tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)] # these tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # we will add BOS AND EOS tokens ourselves
        tokenizer.add_bos_token = False 
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
    
    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"received {len(images)} images for {len(text)} prompts"
        pixel_values = process_images(images, size=(self.image_size, self.image_size), resample=Image.Resampling.BICUBIC, 
                                      rescale_factor=1 / 255.0, image_mean=IMAGENET_STANDARD_MEAN, image_std = IMAGENET_STANDARD_STD)
        # convert the list of numpy arrays to a single np array with shape [Batch_Size, Channels, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.from_numpy(pixel_values)

        # prepend a 'self.image_seq_length' number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(prefix_prompt=prompt, bos_token=self.tokenizer.bos_token, image_seq_len=self.image_seq_length, image_token=self.IMAGE_TOKEN,) 
            for prompt in text]
        # returns the input_ids and attention_mask as pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation
        )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data