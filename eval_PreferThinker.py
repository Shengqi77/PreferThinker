import os
import re
import json
import random
import logging
import argparse
import warnings
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

SYSTEM_PROMPT = (
    "\nThe user provides a set of historically preferred and non-preferred images. Based on this historical data, please predict the user’s <visual preference profile> and non-preference profile across five key profile: art style, color, artistic medium, saturation, and  detail. The predicted results should be described using terms corresponding to each of these five visual profile.\nSubsequently, the user provides two candidate images: Image A and Image B. Using the predicted <visual preference profile>, assign an interpretability score (0–5) for each of the five profile for both images. A higher score indicates stronger alignment with the user’s preferred profile, while a lower score suggests greater similarity to non-preferred profile. Each score must be accompanied by a rationale explaining the rating.\nFinally, sum the scores across all five profile for each image. The image with the higher total score should be identified as the one the user is more likely to prefer.\nBelow are some output examples:\n\noutput example 1:\n\"<visual preference profile>\n\"Art Styles\": \"Mesoamerican Art\",\n\"Color\": \"Pink\",\n\"Artistic Medium\": \"Coiling\",\n\"Saturation\": \"Muted\",\n\"Detail\": \"Realistic\"\n</visual preference profile>\n\n<visual non-preference profile>\n\"Art Styles\": \"Pixel Art\",  \n\"Color\": \"Jungle Green\",\n\"Artistic Medium\": \"Digital Painting\",\n\"Saturation\": \"Mid-tone\",\n\"Detail\": \"Selective\"\n</visual non-preference profile>\n\n<think>\n1. Artistic style: \nImage A(2) - The clustered botanical forms with their geometric arrangement display a digital illustration style that lacks the ceremonial symbolism and cultural motifs characteristic of Mesoamerican Art. The stylized plant structures with their symmetrical composition and modern rendering technique align more closely with contemporary digital illustration, falling between the preferred Mesoamerican Art and non-preferred Pixel Art styles; \nImage B(5) - The vibrant fruit-like structures against the pink background incorporate organic forms reminiscent of Mesoamerican Art's natural symbolism. The decorative patterning on the pink background echoes geometric motifs found in Mesoamerican codices and textiles, while the illustrative technique maintains cultural resonance without digital flatness.\n\n2. Color: \nImage A(1) - The predominant jungle green palette with varying shades of emerald and forest green throughout the botanical elements directly aligns with the non-preferred \"Jungle Green\" attribute. The monochromatic green scheme lacks the warmth and vibrancy associated with the preferred pink tones, creating a cool visual temperature that contradicts the user's color preferences; \nImage B(5) - The vibrant pink background creates an immediate visual harmony with the preferred color attribute. The gradient transitions from salmon to rose pink across the composition provide depth while maintaining the essential pink identity, complemented by contrasting green foliage that enhances rather than dominates the preferred pink palette.\n\n3. Artistic Medium: \nImage A(2) - The smooth gradients and precise edges of the plant structures reveal digital painting techniques with algorithmic precision in the leaf arrangements. The technical execution shows clear digital origin with its perfect symmetry and computer-generated texture mapping, closely resembling the non-preferred \"Digital Painting\" medium rather than the organic irregularities of coiled materials; \nImage B(4) - The textured rendering of the fruit-like structures suggests dimensional craftsmanship reminiscent of coiling techniques. The visible brush texture and canvas grain beneath the illustration creates a handcrafted quality that references traditional media, though not fully achieving the three-dimensional quality of true coiling work.\n\n4. Saturation: \nImage A(3) - The green elements display moderate saturation levels with some intensity in the central portions balanced by darker, less saturated areas in the surrounding foliage. This creates a mid-tone saturation profile that neither strongly aligns with the preferred muted quality nor pushes into high saturation territory, placing it in a neutral position between preferences; \nImage B(4) - The pink background demonstrates a restrained saturation that avoids overwhelming vibrancy while maintaining color identity. The fruit structures display a gradual desaturation from center to edges, creating a subdued color presence that aligns with the preferred \"Muted\" attribute while retaining necessary definition.\n\n5. Detail: \nImage A(3) - The plant structures show moderate detail with defined leaf shapes and textural elements, but employ a stylized approach that simplifies natural complexity. The selective focus on certain botanical features while generalizing others creates a balanced but not fully realistic representation, positioning it between preferred realistic and non-preferred selective detail; \nImage B(4) - The illustration demonstrates careful attention to botanical accuracy in the fruit structures with realistic light interaction, dimensional shading, and textural variations. The leaf veining and organic transitions between elements show commitment to naturalistic representation, though slightly stylized in execution rather than photorealistic.\n\nTotal:\nImage A: total_score=2+1+2+3+3=11\nImage B: total_score=5+5+4+4+4=22\n</think>\n\n<answer>Image B</answer>\"\n\n\noutput example 2:\n\"<visual preference profile>\n\"Art Styles\": \"Oceanic Art\",\n\"Color\": \"Alloy Silver\",\n\"Artistic Medium\": \"Silkscreen\",\n\"Saturation\": \"Pastel\",\n\"Detail\": \"Smooth\"\n</visual preference profile>\n\n<visual non-preference profile>\n\"Art Styles\": \"Romanticism\",\n\"Color\": \"Charcoal Gray\",\n\"Artistic Medium\": \"3D Modeling\",\n\"Saturation\": \"Intense\",\n\"Detail\": \"Fine\"\n</visual non-preference profile>\n\n<think>\n1. Artistic style: \nImage A(4) - The sled dogs in Image A exhibit a dynamic portrayal that connects to Oceanic Art's emphasis on movement and natural forms. The flowing motion of the dogs against the snow landscape creates visual rhythms reminiscent of wave patterns found in Oceanic artistic traditions, while avoiding the dramatic emotional intensity typical of Romanticism; \nImage B(3) - The composition in Image B shows dogs in similar dynamic motion but with a more dramatic contrast between the foreground subject and background forest, creating a slight emotional tension that leans somewhat toward Romantic artistic sensibilities, though the overall treatment remains balanced between styles.\n\n2. Color: \nImage A(4) - The color palette in Image A features subtle silver-blue tones in the snow and sky that closely align with the preferred Alloy Silver coloration, creating a metallic sheen effect across the image that enhances the dogs' natural coloring without using harsh charcoal tones; \nImage B(2) - Image B employs a darker, more contrasting palette with charcoal gray tones dominating the background forest and creating stronger shadows on the snow, moving away from the preferred silver tones toward the non-preferred charcoal gray spectrum.\n\n3. Artistic Medium: \nImage A(4) - The rendering technique in Image A has a smooth, consistent quality with even lighting that creates a silkscreen-like effect, particularly in how the light interacts with the snow surface and the dogs' fur, giving it a flattened yet detailed appearance characteristic of silkscreen prints; \nImage B(2) - Image B displays characteristics more aligned with 3D modeling techniques, particularly in how the depth of field is rendered between foreground and background elements, with the forest creating a dimensional space that feels more computationally rendered than screen printed.\n\n4. Saturation: \nImage A(5) - The color treatment in Image A employs pastel-like, subdued saturation levels that create a gentle visual experience, particularly in the background sky and snow areas which show subtle color variations without intensity; \nImage B(2) - Image B features more intensely saturated elements, especially in the contrast between the dogs and the background forest, creating a more dramatic visual impact that aligns with the non-preferred intense saturation attribute.\n\n5. Detail: \nImage A(4) - The snow texture and dog fur in Image A are rendered with a smooth, flowing quality that maintains detail without excessive definition, creating a harmonious visual experience that aligns with the preferred smooth detail attribute; \nImage B(3) - Image B presents a more mixed approach to detail, with the foreground dog showing smoother rendering but with finer detail definition in the fur and harness elements, placing it between the preferred smooth and non-preferred fine detail profile.\n\nTotal:\nImage A: total_score=4+4+4+5+4=21\nImage B: total_score=3+2+2+2+3=12\n</think>\n\n<answer>Image A</answer>\"\n"
)

def setup_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank

def clean_attributes_string(text: str) -> str:
    text = text.replace('"', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_attributes(text: str, tag: str, exclue_chars=['\n', '\r']) -> str:
    matches = re.findall(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if matches:
        text = matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    return text.strip().rstrip('.').lower()

def evaluate(args):
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    logger = setup_logger(rank)
    
    logger.info(f"Initialized Distributed Process Group. World Size: {world_size}")
    logger.info(f"Loading Model from: {args.model_path}")

    ds_name = os.path.splitext(os.path.basename(args.data_path))[0]
    logger.info(f"Processing dataset: {ds_name} from {args.data_path}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    try:
        with open(args.data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {args.data_path}")
        return
        
    random.seed(args.seed)
    random.shuffle(data)
    data = data[:args.num_samples]

    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []

    for x in rank_data:
        num_history = args.num_history
        imageA_paths = x.get('imageA', '')
        imageB_paths = x.get('imageB', '')
        preferred_paths = x.get('prefered_images', [])[:num_history]
        non_preferred_paths = x.get('non_prefered_images', [])[:num_history]

        content = []

        if preferred_paths and non_preferred_paths:
            content.append({
                'type': 'text',
                'text': "First, given the following user's historical preferred images. Please integrate the common stylistic attributes from historical preference images into the <visual preference profile> </visual preference profile> tags. Focusing on FIVE attributes: artistic style, Color, Artistic Medium, Saturation, Detail. Please generate responses with varied word choices, avoiding unnecessary repetition."
            })

            for i, img_path in enumerate(preferred_paths):
                img_path_abs = os.path.join(args.image_root, img_path)
                content.append({
                    'type': 'image',
                    'image': f"file://{img_path_abs}", 
                    'text': f"Preferred Image {i+1}.", 
                    'label': f"Preferred Image {i+1}."
                })
            
            content.append({
                'type': 'text',
                'text': "Second, given the following user's historical non-preferred images. Please integrate the common stylistic attributes from historical non-preference images into the <visual non-preference profile> </visual non-preference profile> tags. Focusing on FIVE attributes: Artistic style, Color, Artistic Medium, Saturation, Detail. Please generate responses with varied word choices, avoiding unnecessary repetition."
            })

            for i, img_path in enumerate(non_preferred_paths):
                img_path_abs = os.path.join(args.image_root, img_path)
                content.append({
                    'type': 'image',
                    'image': f"file://{img_path_abs}",
                    'text': f"Non-Preferred Image {i+1}.", 
                    'label': f"Non-Preferred Image {i+1}."
                })

        content.append({
                'type': 'text',
                'text': "Then, given the following  two candidate images: Image A and Image B."
        })

        if imageA_paths and imageB_paths:
            content.append({'type': 'text', 'text': "Image A"})
            img_path_abs = os.path.join(args.image_root, imageA_paths)
            content.append({'type': 'image', 'image': f"file://{img_path_abs}"})

            content.append({'type': 'text', 'text': "Image B"})
            img_path_abs = os.path.join(args.image_root, imageB_paths)
            content.append({'type': 'image', 'image': f"file://{img_path_abs}"})
    
        content.append({
            'type': 'text',
            'text': '''Using the predicted preference profile and non-preference profile, please analyze and score the two images across five dimensions. Sum the scores from the five dimensions to obtain a total score, and determine which image the user prefers based on the total score. First, enclose the preferference profile within <visual preference profile> and </visual preference profile>, and the non-preferred profile within <visual non-preference profile> and </visual non-preference profile>. Then, place the multi-dimensional analysis and scoring of the two images within <think> and </think>. Finally, provide the selected answer between <answer> and </answer> tags. The answer should be in the format: <answer>Image A</answer> or <answer>Image B</answer>'''
        })

        message = [
            {'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}
        ]
        messages.append(message)

    rank_outputs = []

    logger.info(f"Starting inference on {len(messages)} batches...")
    for i in tqdm(range(0, len(messages), args.batch_size), disable=rank != 0):
        batch_messages = messages[i:i + args.batch_size]
    
        text = [processor.apply_chat_template(msg, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, 
                use_cache=True, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=False, 
                repetition_penalty=1.15
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            ground_truth = input_example['conversations'][0]['value']

            prefer_attributes = clean_attributes_string(extract_attributes(model_output, "visual preference profile"))
            non_prefer_attributes = clean_attributes_string(extract_attributes(model_output, "visual non-preference profile"))

            model_answer_ = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            model_answer = model_answer_[-1].strip() if model_answer_ else None
            
            correct = 1 if model_answer == ground_truth else 0
            correct_number += correct
            
            result = {
                'imageA': input_example.get('imageA'),
                'imageB': input_example.get('imageB'),
                'question': ground_truth,
                'ground_truth': ground_truth,
                'model_output': model_output,
                'extracted_answer': model_answer,
                'correct': correct,
                'predicted_preference_profile': prefer_attributes,
                'predicted_non_preference_profile': non_prefer_attributes
            }
            final_output.append(result)

        accuracy = correct_number / len(data) * 100
        logger.info(f"Accuracy of {ds_name}: {accuracy:.2f}%")

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"rec_results_{args.run_name}_{ds_name}.json")
            
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        logger.info(f"Results successfully saved to {output_path}")

    dist.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen VL Model on User Preference Datasets")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained/finetuned model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--image_root", type=str, default="./data/PICKAPIC/images", help="Root directory containing image files")
    parser.add_argument("--output_dir", type=str, default="./logs", help="Directory to save evaluation results")
    parser.add_argument("--run_name", type=str, default="evaluation", help="Custom name for this evaluation run")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_samples", type=int, default=10000, help="Maximum number of samples to process")
    parser.add_argument("--num_history", type=int, default=5, help="Number of historical preferred/non-preferred images to use")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens for model generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling")
    
    args = parser.parse_args()
    evaluate(args)