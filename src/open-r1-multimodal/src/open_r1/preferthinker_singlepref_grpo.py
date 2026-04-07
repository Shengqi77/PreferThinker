# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
from datasets import Dataset
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from transformers import AutoTokenizer
from transformers.integrations import deepspeed as hf_deepspeed
from transformers.utils import logging

from trainer.grpo_config import GRPOConfig
from trainer.grpo_trainer_preference_multi_turn_DAPO import VLMGRPOTrainer
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from vlm_modules import *

from dreamsim import dreamsim
from dreamsim.feature_extraction.extractor import ViTExtractor
from dreamsim.feature_extraction.vision_transformer import DINOHead, vit_base
from Flux_API_client import generate_image
from qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load,
)


logger = logging.get_logger(__name__)

monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()

_ORIGINAL_DREAMSIM_CREATE_MODEL = ViTExtractor.create_model

tokenizer = None

EVAL_MODELS = {
    "sbert": None,
    "dreamsim": None,
    "device": None,
}

GLOBAL_PATHS = {
    "save_images_base_dir": None,
    "save_txt_base_dir": None,
}


def _load_local_dreamsim_dino_vitb16(load_dir):
    ckpt_path = os.path.join(load_dir, "dino_vitb16_pretrain.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DreamSim DINO checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["student"]
    backbone_state = {
        key.removeprefix("module.backbone."): value
        for key, value in state_dict.items()
        if key.startswith("module.backbone.")
    }

    model = vit_base(patch_size=16, num_classes=0)
    model.load_state_dict(backbone_state, strict=True)

    proj = DINOHead(768, 2048)
    proj.mlp[0].weight.data = state_dict["module.head.mlp.0.weight"]
    proj.mlp[0].bias.data = state_dict["module.head.mlp.0.bias"]
    proj.mlp[2].weight.data = state_dict["module.head.mlp.2.weight"]
    proj.mlp[2].bias.data = state_dict["module.head.mlp.2.bias"]
    proj.mlp[4].weight.data = state_dict["module.head.mlp.4.weight"]
    proj.mlp[4].bias.data = state_dict["module.head.mlp.4.bias"]
    proj.last_layer.weight.data = state_dict["module.head.last_layer.weight"]
    return model, proj


def _patch_dreamsim_offline_loader():
    def _create_model_with_local_fallback(model_type, load_dir="./models"):
        if model_type == "dino_vitb16":
            ckpt_path = os.path.join(load_dir, "dino_vitb16_pretrain.pth")
            if os.path.exists(ckpt_path):
                logger.info("Using local DreamSim DINO checkpoint: %s", ckpt_path)
                return _load_local_dreamsim_dino_vitb16(load_dir)
        return _ORIGINAL_DREAMSIM_CREATE_MODEL(model_type, load_dir)

    ViTExtractor.create_model = staticmethod(_create_model_with_local_fallback)


_patch_dreamsim_offline_loader()


SYSTEM_PROMPT = (
    """
The user provides a set of historically preferred and non-preferred images. Based on this historical data, please predict the user’s visual preference and non-preference profile across five key attributes: art style, color, artistic medium, saturation, and  detail. The predicted results should be described using terms corresponding to each of these five visual attributes.
Subsequently, the user provides two candidate images: Image A and Image B. Using the predicted visual preference profile as criteria, assign an interpretability score (0–5) for each of the five attributes for both images. A higher score indicates stronger alignment with the user’s preferred attributes, while a lower score suggests greater similarity to non-preferred attributes. Each score must be accompanied by a rationale explaining the rating.
Finally, sum the scores across all five attributes for each image. The image with the higher total score should be identified as the one the user is more likely to prefer.
Below are some output examples:

output example 1:
"<visual preference profile>
"Art Styles": "Mesoamerican Art",
"Color": "Pink",
"ArtisticMedium": "Coiling",
"Saturation": "Muted",
"Detail": "Realistic"
</visual preference profile>

<visual non-preference profile>
"Art Styles": "Pixel Art",  
"Color": "Jungle Green",
"ArtisticMedium": "Digital Painting",
"Saturation": "Mid-tone",
"Detail": "Selective"
</visual non-preference profile>

<think>
1. Artistic style: 
Image A(2) - The clustered botanical forms with their geometric arrangement display a digital illustration style that lacks the ceremonial symbolism and cultural motifs characteristic of Mesoamerican Art. The stylized plant structures with their symmetrical composition and modern rendering technique align more closely with contemporary digital illustration, falling between the preferred Mesoamerican Art and non-preferred Pixel Art styles; 
Image B(5) - The vibrant fruit-like structures against the pink background incorporate organic forms reminiscent of Mesoamerican Art's natural symbolism. The decorative patterning on the pink background echoes geometric motifs found in Mesoamerican codices and textiles, while the illustrative technique maintains cultural resonance without digital flatness.

2. Color: 
Image A(1) - The predominant jungle green palette with varying shades of emerald and forest green throughout the botanical elements directly aligns with the non-preferred "Jungle Green" attribute. The monochromatic green scheme lacks the warmth and vibrancy associated with the preferred pink tones, creating a cool visual temperature that contradicts the user's color preferences; 
Image B(5) - The vibrant pink background creates an immediate visual harmony with the preferred color attribute. The gradient transitions from salmon to rose pink across the composition provide depth while maintaining the essential pink identity, complemented by contrasting green foliage that enhances rather than dominates the preferred pink palette.

3. Artistic Medium: 
Image A(2) - The smooth gradients and precise edges of the plant structures reveal digital painting techniques with algorithmic precision in the leaf arrangements. The technical execution shows clear digital origin with its perfect symmetry and computer-generated texture mapping, closely resembling the non-preferred "Digital Painting" medium rather than the organic irregularities of coiled materials; 
Image B(4) - The textured rendering of the fruit-like structures suggests dimensional craftsmanship reminiscent of coiling techniques. The visible brush texture and canvas grain beneath the illustration creates a handcrafted quality that references traditional media, though not fully achieving the three-dimensional quality of true coiling work.

4. Saturation: 
Image A(3) - The green elements display moderate saturation levels with some intensity in the central portions balanced by darker, less saturated areas in the surrounding foliage. This creates a mid-tone saturation profile that neither strongly aligns with the preferred muted quality nor pushes into high saturation territory, placing it in a neutral position between preferences; 
Image B(4) - The pink background demonstrates a restrained saturation that avoids overwhelming vibrancy while maintaining color identity. The fruit structures display a gradual desaturation from center to edges, creating a subdued color presence that aligns with the preferred "Muted" attribute while retaining necessary definition.

5. Detail: 
Image A(3) - The plant structures show moderate detail with defined leaf shapes and textural elements, but employ a stylized approach that simplifies natural complexity. The selective focus on certain botanical features while generalizing others creates a balanced but not fully realistic representation, positioning it between preferred realistic and non-preferred selective detail; 
Image B(4) - The illustration demonstrates careful attention to botanical accuracy in the fruit structures with realistic light interaction, dimensional shading, and textural variations. The leaf veining and organic transitions between elements show commitment to naturalistic representation, though slightly stylized in execution rather than photorealistic.

Total:
Image A: total_score=2+1+2+3+3=11
Image B: total_score=5+5+4+4+4=22
</think>

<answer>Image B</answer>"


output example 2:
"<visual preference profile>
"Art Styles": "Oceanic Art",
"Color": "Alloy Silver",
"ArtisticMedium": "Silkscreen",
"Saturation": "Pastel",
"Detail": "Smooth"
</visual preference profile>

<visual non-preference profile>
"Art Styles": "Romanticism",
"Color": "Charcoal Gray",
"ArtisticMedium": "3D Modeling",
"Saturation": "Intense",
"Detail": "Fine"
</visual non-preference profile>

<think>
1. Artistic style: 
Image A(4) - The sled dogs in Image A exhibit a dynamic portrayal that connects to Oceanic Art's emphasis on movement and natural forms. The flowing motion of the dogs against the snow landscape creates visual rhythms reminiscent of wave patterns found in Oceanic artistic traditions, while avoiding the dramatic emotional intensity typical of Romanticism; 
Image B(3) - The composition in Image B shows dogs in similar dynamic motion but with a more dramatic contrast between the foreground subject and background forest, creating a slight emotional tension that leans somewhat toward Romantic artistic sensibilities, though the overall treatment remains balanced between styles.

2. Color: 
Image A(4) - The color palette in Image A features subtle silver-blue tones in the snow and sky that closely align with the preferred Alloy Silver coloration, creating a metallic sheen effect across the image that enhances the dogs' natural coloring without using harsh charcoal tones; 
Image B(2) - Image B employs a darker, more contrasting palette with charcoal gray tones dominating the background forest and creating stronger shadows on the snow, moving away from the preferred silver tones toward the non-preferred charcoal gray spectrum.

3. Artistic Medium: 
Image A(4) - The rendering technique in Image A has a smooth, consistent quality with even lighting that creates a silkscreen-like effect, particularly in how the light interacts with the snow surface and the dogs' fur, giving it a flattened yet detailed appearance characteristic of silkscreen prints; 
Image B(2) - Image B displays characteristics more aligned with 3D modeling techniques, particularly in how the depth of field is rendered between foreground and background elements, with the forest creating a dimensional space that feels more computationally rendered than screen printed.

4. Saturation: 
Image A(5) - The color treatment in Image A employs pastel-like, subdued saturation levels that create a gentle visual experience, particularly in the background sky and snow areas which show subtle color variations without intensity; 
Image B(2) - Image B features more intensely saturated elements, especially in the contrast between the dogs and the background forest, creating a more dramatic visual impact that aligns with the non-preferred intense saturation attribute.

5. Detail: 
Image A(4) - The snow texture and dog fur in Image A are rendered with a smooth, flowing quality that maintains detail without excessive definition, creating a harmonious visual experience that aligns with the preferred smooth detail attribute; 
Image B(3) - Image B presents a more mixed approach to detail, with the foreground dog showing smoother rendering but with finer detail definition in the fur and harness elements, placing it between the preferred smooth and non-preferred fine detail attributes.

Total:
Image A: total_score=4+4+4+5+4=21
Image B: total_score=3+2+2+2+3=12
</think>

<answer>Image A</answer>"
"""
)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_file_paths: str = field(
        default=None, metadata={"help": "Paths to data files, separated by ':'"}
    )
    image_folders: str = field(
        default=None, metadata={"help": "Paths to image folders, separated by ':'"}
    )
    arrow_cache_dir: str = field(
        default=None, metadata={"help": "Path to arrow cache directory"}
    )
    val_split_ratio: float = field(
        default=0.0, metadata={"help": "Ratio of validation split, default 0.0"}
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'prefer_attributes', etc."},
    )
    max_pixels: Optional[int] = field(
        default=12845056, metadata={"help": "Maximum number of pixels for the image (for QwenVL)"}
    )
    min_pixels: Optional[int] = field(
        default=3136, metadata={"help": "Minimum number of pixels for the image (for QwenVL)"}
    )
    max_anyres_num: Optional[int] = field(
        default=12, metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"}
    )
    reward_method: Optional[str] = field(
        default=None, metadata={"help": "Choose reward method: 'all_match', etc."}
    )
    prefer_attributes_reward_method: Optional[str] = field(
        default=None, metadata={"help": "Choose reward method: 'prefer_attributes', etc."}
    )
    non_prefer_attributes_reward_method: Optional[str] = field(
        default=None, metadata={"help": "Choose reward method: 'non_prefer_attributes', etc."}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": "Choose task type: 'prefer_eval', etc."}
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False, metadata={"help": "Whether to use a customized reward from vlm module"}
    )
    sbert_model_path: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "Path or HuggingFace repo ID for the SBERT model."},
    )
    dreamsim_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for dreamsim model."}
    )
    generated_image_dir: str = field(
        default="./generated_images",
        metadata={"help": "Directory to save generated images and logs."},
    )
    flux_server_dir: str = field(
        default="./flux_server_data",
        metadata={"help": "Directory for FLUX server interactions."},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False



def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def _get_local_eval_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = max(torch.cuda.device_count(), 1)
    return torch.device(f"cuda:{local_rank % device_count}")


@contextmanager
def _temporarily_disable_zero3_init():
    weak_ref = getattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
    current_cfg = weak_ref() if weak_ref is not None and weak_ref() is not None else None

    if current_cfg is None:
        yield
        return

    logger.info("Temporarily disabling Hugging Face ZeRO-3 init for reward-model loading.")
    hf_deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        hf_deepspeed.set_hf_deepspeed_config(current_cfg)


def _load_sbert_model(sbert_path, device):
    with _temporarily_disable_zero3_init():
        model = SentenceTransformer(sbert_path, device=str(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _sbert_embedding_is_valid(model):
    weight = model[0].auto_model.embeddings.word_embeddings.weight
    return getattr(weight, "ndim", None) == 2


def _encode_with_sbert(text):
    model = EVAL_MODELS["sbert"]
    sbert_device = EVAL_MODELS["sbert_device"]

    if not _sbert_embedding_is_valid(model):
        logger.warning(
            "SBERT embedding weight is not 2-D on %s. Reloading the reward model on CPU for a safe fallback.",
            sbert_device,
        )
        sbert_device = torch.device("cpu")
        EVAL_MODELS["sbert_device"] = sbert_device
        EVAL_MODELS["sbert"] = _load_sbert_model(EVAL_MODELS["sbert_model_path"], sbert_device)
        model = EVAL_MODELS["sbert"]

    try:
        return model.encode(
            text or "",
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(sbert_device),
        )
    except RuntimeError as exc:
        if "'weight' must be 2-D" not in str(exc):
            raise

        logger.warning(
            "SBERT encode failed with a ZeRO-like partitioned embedding on %s. Reloading on CPU and retrying once.",
            sbert_device,
        )
        sbert_device = torch.device("cpu")
        EVAL_MODELS["sbert_device"] = sbert_device
        EVAL_MODELS["sbert"] = _load_sbert_model(EVAL_MODELS["sbert_model_path"], sbert_device)
        return EVAL_MODELS["sbert"].encode(
            text or "",
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(sbert_device),
        )


def init_eval_models(sbert_path, dreamsim_cache):
    if EVAL_MODELS["device"] is not None:
        return

    EVAL_MODELS["device"] = _get_local_eval_device()
    EVAL_MODELS["sbert_device"] = torch.device("cpu")
    EVAL_MODELS["sbert_model_path"] = sbert_path
    logger.info(
        "Initializing reward models on rank %s: SBERT on %s, DreamSim on %s",
        os.environ.get("RANK", "0"),
        EVAL_MODELS["sbert_device"],
        EVAL_MODELS["device"],
    )
    EVAL_MODELS["sbert"] = _load_sbert_model(sbert_path, EVAL_MODELS["sbert_device"])
    logger.info("Initializing DreamSim model...")
    EVAL_MODELS["dreamsim"], _ = dreamsim(
        pretrained=True,
        device=EVAL_MODELS["device"],
        cache_dir=dreamsim_cache,
    )


def setup_directories(generated_image_dir, flux_server_dir, run_name):
    save_images_base_dir = os.path.join(generated_image_dir, run_name)
    os.makedirs(save_images_base_dir, exist_ok=True)

    GLOBAL_PATHS["save_images_base_dir"] = save_images_base_dir
    GLOBAL_PATHS["save_txt_base_dir"] = f"{save_images_base_dir}.txt"

    if os.path.exists(flux_server_dir):
        logger.info("Directory '%s' exists. Recreating...", flux_server_dir)
        shutil.rmtree(flux_server_dir)
    os.makedirs(flux_server_dir, exist_ok=True)



def clean_text(text, exclue_chars=["\n", "\r"]):
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]
    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")
    return text.strip().rstrip(".").lower()


def clean_text_prefer_attributes(text, exclue_chars=["\n", "\r"]):
    answer_matches = re.findall(
        r"<visual preference profile>(.*?)</visual preference profile>",
        text,
        re.DOTALL,
    )
    if answer_matches:
        text = answer_matches[-1]
    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")
    return text.strip().rstrip(".").lower()


def clean_text_non_prefer_attributes(text, exclue_chars=["\n", "\r"]):
    answer_matches = re.findall(
        r"<visual non-preference profile>(.*?)</visual non-preference profile>",
        text,
        re.DOTALL,
    )
    if answer_matches:
        text = answer_matches[-1]
    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")
    return text.strip().rstrip(".").lower()


def clean_attributes_string(text):
    text = text.replace('"', "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


t = transforms.Compose([transforms.ToTensor()])


def preprocess(img):
    img = img.convert("RGB")
    return t(img).unsqueeze(0)


def tensor_to_image_save(tensor, save_path):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(save_path)


def _safe_save_debug_images(regenerated_image, target_image, regenerate_path, target_path, metric_name):
    if regenerated_image is None or target_image is None:
        return

    try:
        tensor_to_image_save(regenerated_image, regenerate_path)
        tensor_to_image_save(target_image, target_path)
    except Exception:
        logger.exception("Failed to save %s debug images.", metric_name)



def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, method in zip(contents, solution, kwargs.get("accu_reward_method")):
        reward = 1.0 if method == "all_match" and clean_text(content) == clean_text(sol) else 0.0
        rewards.append(reward)
    return rewards


def prefer_attributes_reward(content, attributes_gt, prompt, preferred_image_path, step, user_id, **kwargs):
    if isinstance(attributes_gt, dict):
        key_order = ["Art Styles", "Color", "Artistic Medium", "Saturation", "Detail"]
        attributes_gt = ", ".join(
            f"{key}: {attributes_gt[key]}" for key in key_order if key in attributes_gt
        )
    elif attributes_gt is None:
        attributes_gt = ""

    attributes_predict = clean_attributes_string(clean_text_prefer_attributes(content))
    attributes_gt = clean_text_prefer_attributes(attributes_gt)
    eval_device = EVAL_MODELS["device"]

    predict_embeddings = _encode_with_sbert(attributes_predict)
    gt_embeddings = _encode_with_sbert(attributes_gt)
    text_sim = EVAL_MODELS["sbert"].similarity(predict_embeddings, gt_embeddings).to(eval_device)

    recaption_prompt = prompt + attributes_predict
    img_regenerate = preprocess(generate_image(recaption_prompt)).to(eval_device)
    img_prefer = preprocess(Image.open(preferred_image_path)).to(eval_device)
    distance = EVAL_MODELS["dreamsim"](img_regenerate, img_prefer).to(eval_device)
    vision_sim = 1 - distance

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_regenerate_path = os.path.join(
        GLOBAL_PATHS["save_images_base_dir"],
        f"{timestamp}_{user_id}_step{step}_textsim:{text_sim.item()}_vissim:{vision_sim.item()}_regenerate_prefer.jpg",
    )
    img_prefer_path = os.path.join(
        GLOBAL_PATHS["save_images_base_dir"],
        f"{timestamp}_{user_id}_step{step}_prefer.jpg",
    )
    _safe_save_debug_images(
        img_regenerate, img_prefer, img_regenerate_path, img_prefer_path, "prefer_attributes"
    )

    with open(GLOBAL_PATHS["save_txt_base_dir"], "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(
            f"prefer=========Txtsim:{text_sim.item()}========={timestamp}=========Imgsim:{vision_sim.item()}========prefer\n"
        )
        f.write(f"{user_id}_step{step}_prompt: {prompt}\n")
        f.write(f"{user_id}_step{step}_attributes_predict_prefer_recaption: {recaption_prompt}\n")
        f.write(f"{user_id}_step{step}_attributes_predict_prefer: {attributes_predict}\n")
        f.write(f"{user_id}_step{step}_attributes_GT_prefer: {attributes_gt}\n")
        f.write(
            f"prefer=========Txtsim:{text_sim.item()}========={timestamp}=========Imgsim:{vision_sim.item()}========prefer\n"
        )
        f.write("\n")

    return text_sim * 0 + vision_sim * 0


def non_prefer_attributes_reward(content, attributes_gt, prompt, non_preferred_image_path, step, user_id, **kwargs):
    if isinstance(attributes_gt, dict):
        key_order = ["Art Styles", "Color", "Artistic Medium", "Saturation", "Detail"]
        attributes_gt = ", ".join(
            f"{key}: {attributes_gt[key]}" for key in key_order if key in attributes_gt
        )
    elif attributes_gt is None:
        attributes_gt = ""

    attributes_predict = clean_attributes_string(clean_text_non_prefer_attributes(content))
    attributes_gt = clean_text_non_prefer_attributes(attributes_gt)
    eval_device = EVAL_MODELS["device"]

    predict_embeddings = _encode_with_sbert(attributes_predict)
    gt_embeddings = _encode_with_sbert(attributes_gt)
    text_sim = EVAL_MODELS["sbert"].similarity(predict_embeddings, gt_embeddings).to(eval_device)

    recaption_prompt = prompt + attributes_predict
    img_regenerate = preprocess(generate_image(recaption_prompt)).to(eval_device)
    img_non_prefer = preprocess(Image.open(non_preferred_image_path)).to(eval_device)
    distance = EVAL_MODELS["dreamsim"](img_regenerate, img_non_prefer).to(eval_device)
    vision_sim = 1 - distance

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_regenerate_path = os.path.join(
        GLOBAL_PATHS["save_images_base_dir"],
        f"{timestamp}_{user_id}_step{step}_textsim:{text_sim.item()}_vissim:{vision_sim.item()}_regenerate_non-prefer.jpg",
    )
    img_non_prefer_path = os.path.join(
        GLOBAL_PATHS["save_images_base_dir"],
        f"{timestamp}_{user_id}_step{step}_non-prefer.jpg",
    )
    _safe_save_debug_images(
        img_regenerate,
        img_non_prefer,
        img_regenerate_path,
        img_non_prefer_path,
        "non_prefer_attributes",
    )

    with open(GLOBAL_PATHS["save_txt_base_dir"], "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(
            f"non_prefer=========Txtsim:{text_sim.item()}========={timestamp}=========Imgsim:{vision_sim.item()}========non_prefer\n"
        )
        f.write(f"{user_id}_step{step}_prompt: {prompt}\n")
        f.write(f"{user_id}_step{step}_attributes_predict_non_prefer_recaption: {recaption_prompt}\n")
        f.write(f"{user_id}_step{step}_attributes_predict_non_prefer: {attributes_predict}\n")
        f.write(f"{user_id}_step{step}_attributes_GT_non_prefer: {attributes_gt}\n")
        f.write(
            f"non_prefer=========Txtsim:{text_sim.item()}========={timestamp}=========Imgsim:{vision_sim.item()}========non_prefer\n"
        )
        f.write("\n")

    return text_sim * 0 + vision_sim * 0


def prefer_attribute_sim_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, method, gt, ref_prompt, img_path, step, user_id in zip(
        contents,
        kwargs.get("prefer_attributes_reward_method"),
        kwargs.get("prefer_attributes"),
        kwargs.get("reference_prompt_list"),
        kwargs.get("preferred_images_path"),
        kwargs.get("step"),
        kwargs.get("user_id"),
    ):
        reward = (
            prefer_attributes_reward(content, gt, ref_prompt[0], img_path[0], step, user_id)
            if method == "prefer_attributes"
            else 0.0
        )
        rewards.append(reward)
    return rewards


def non_prefer_attribute_sim_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, method, gt, ref_prompt, img_path, step, user_id in zip(
        contents,
        kwargs.get("non_prefer_attributes_reward_method"),
        kwargs.get("non_prefer_attributes"),
        kwargs.get("reference_prompt_list"),
        kwargs.get("non_preferred_images_path"),
        kwargs.get("step"),
        kwargs.get("user_id"),
    ):
        reward = (
            non_prefer_attributes_reward(content, gt, ref_prompt[0], img_path[0], step, user_id)
            if method == "non_prefer_attributes"
            else 0.0
        )
        rewards.append(reward)
    return rewards


def format_with_attributes_with_think_reward(completions, **kwargs):
    pattern = (
        r"<visual preference profile>.*?</visual preference profile>\s*"
        r"<visual non-preference profile>.*?</visual non-preference profile>\s*"
        r"<think>.*?</think>\s*"
        r"<answer>.*?</answer>"
    )
    contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in contents]
    return [0.3 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "prefer_attributes": prefer_attribute_sim_reward,
    "non-prefer_attributes": non_prefer_attribute_sim_reward,
    "format": format_with_attributes_with_think_reward,
}


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    if "internvl" in model_name_or_path.lower():
        return InvernVLModule
    raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    setup_directories(script_args.generated_image_dir, script_args.flux_server_dir, training_args.run_name)
    init_eval_models(script_args.sbert_model_path, script_args.dreamsim_cache_dir)

    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    logger.info("Using VLM module: %s", vlm_module_cls.__name__)

    if script_args.is_reward_customized_from_vlm_module:
        reward_funcs = [
            vlm_module_cls.select_reward_func(func, script_args.task_type)
            for func in script_args.reward_funcs
        ]
    else:
        reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    assert len(data_files) == len(image_folders), "Number of data files must match image folders"

    accu_reward_methods = ["default"] * len(data_files) if not script_args.reward_method else script_args.reward_method.split(":")
    if script_args.prefer_attributes_reward_method is None:
        raise ValueError("No prefer_attributes_reward_method")
    if script_args.non_prefer_attributes_reward_method is None:
        raise ValueError("No non_prefer_attributes_reward_method")

    prefer_attributes_reward_methods = script_args.prefer_attributes_reward_method.split(":")
    non_prefer_attributes_reward_methods = script_args.non_prefer_attributes_reward_method.split(":")

    assert len(accu_reward_methods) == len(data_files), "Number of reward methods must match number of data files"
    assert len(prefer_attributes_reward_methods) == len(data_files), "Number of reward methods must match number of data files"
    assert len(non_prefer_attributes_reward_methods) == len(data_files), "Number of reward methods must match number of data files"

    all_data = []
    for data_file, image_folder, acc_method, pref_method, non_pref_method in zip(
        data_files,
        image_folders,
        accu_reward_methods,
        prefer_attributes_reward_methods,
        non_prefer_attributes_reward_methods,
    ):
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for field in ["imageA", "imageB", "prefered_images", "non_prefered_images"]:
                    if field not in item:
                        continue
                    if isinstance(item[field], str):
                        item[f"{field}_path"] = os.path.join(image_folder, item[field])
                    elif isinstance(item[field], list):
                        item[f"{field}_path"] = [
                            os.path.join(image_folder, img) for img in item[field][:5]
                        ]
                    del item[field]

                item["problem"] = item["conversations"][0]["value"]
                solution_value = item["conversations"][1]["value"]
                item["solution"] = (
                    solution_value.replace("<answer>", "").replace("</answer>", "").strip()
                    if isinstance(solution_value, str)
                    else str(solution_value)
                )
                del item["conversations"]

                item["accu_reward_method"] = item.get("accu_reward_method", acc_method)
                item["prefer_attributes_reward_method"] = item.get(
                    "prefer_attributes_reward_method", pref_method
                )
                item["non_prefer_attributes_reward_method"] = item.get(
                    "non_prefer_attributes_reward_method", non_pref_method
                )
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        def validate_paths(field):
            if field in example:
                paths = example[field] if isinstance(example[field], list) else [example[field]]
                assert all(os.path.exists(path) for path in paths), f"Image paths do not exist: {paths}"
                return paths
            return []

        imageA_paths = validate_paths("imageA_path")
        imageB_paths = validate_paths("imageB_path")
        preferred_paths = validate_paths("prefered_images_path")
        non_preferred_paths = validate_paths("non_prefered_images_path")

        content = []
        if preferred_paths and non_preferred_paths:
            content.append(
                {
                    "type": "text",
                    "text": "First, given the following user's historical preferred images. Please integrate the common stylistic attributes from historical preference images into the <visual preference profile> </visual preference profile> tags. Focusing on FIVE attributes: artistic style, Color, Artistic Medium, Saturation, Detail. Please generate responses with varied word choices, avoiding unnecessary repetition.",
                }
            )
            for idx, _ in enumerate(preferred_paths):
                content.append({"type": "image", "text": f"Preferred Image {idx + 1}"})

            content.append(
                {
                    "type": "text",
                    "text": "Second, given the following user's historical non-preferred images. Please integrate the common stylistic attributes from historical non-preference images into the <visual non-preference profile> </visual non-preference profile> tags. Focusing on FIVE attributes: Artistic style, Color, Artistic Medium, Saturation, Detail. Please generate responses with varied word choices, avoiding unnecessary repetition.",
                }
            )
            for idx, _ in enumerate(non_preferred_paths):
                content.append({"type": "image", "text": f"Non-Preferred Image {idx + 1}"})

        content.append(
            {
                "type": "text",
                "text": "Then, given the following  two candidate images: Image A and Image B.",
            }
        )

        if imageA_paths:
            content.append({"type": "text", "text": "Image A"})
            content.append({"type": "image", "text": "Candidate Image A"})

        if imageB_paths:
            content.append({"type": "text", "text": "Image B"})
            content.append({"type": "image", "text": "Candidate Image B"})

        content.append({"type": "text", "text": example["problem"]})

        return {
            "problem": example["problem"],
            "solution": f"<answer> {example['solution']} </answer>",
            "user_id": example["user_id"],
            "reference_prompt_list": example["reference_prompt_list"],
            "preference attributes": f"<visual preference profile> {example['prefer_attributes']} </visual preference profile>",
            "non-preference attributes": f"<visual non-preference profile> {example['non_prefer_attributes']} </visual non-preference profile>",
            "accu_reward_method": example["accu_reward_method"],
            "prefer_attributes_reward_method": example["prefer_attributes_reward_method"],
            "non_prefer_attributes_reward_method": example["non_prefer_attributes_reward_method"],
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ],
            "imageA_path": example.get("imageA_path"),
            "imageB_path": example.get("imageB_path"),
            "preferred_images_path": example.get("prefered_images_path"),
            "non_preferred_images_path": example.get("non_prefered_images_path"),
        }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    splits = {"train": dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(test_size=script_args.val_split_ratio)
        splits["train"] = train_val_split["train"]
        splits["validation"] = train_val_split["test"]

    logger.info("Dataset has been loaded successfully.")

    initialize_tokenizer(model_args.model_name_or_path)

    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation") if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        logger.info("ZeRO-3 is used, Qwen2.5-VL forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()

    main(script_args, training_args, model_args)
