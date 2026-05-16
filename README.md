<div align="center">

# [ICLR 2026 🔥] PreferThinker: Reasoning-based Personalized Image Preference Assessment

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/ZhouSimple/PreferImg-Bench)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ZhouSimple/PreferThinker_model)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## :black_nib: TODO List<a name="todo"></a>

- [x] Release [PreferThinker paper](https://arxiv.org/pdf/2511.00609).
- [x] Release evaluation code.
- [x] Release PreferThinker-Bench.
- [x] Release inference checkpoints.
- [x] Release CoT-based personalized preference assessment dataset.
- [x] Release training code.

<br/>

## 🧪 Released Test Sets

We have released the **PreferImg-Bench** on Hugging Face. This benchmark is designed for evaluating personalized image preference assessment.

| Subset | Samples | Description |
| :--- | :--- | :--- |
| **TEST_MP_ID** | 250 | Multi-preference In-Domain test set |
| **TEST_MP_OD** | 250 | Multi-preference Out-of-Domain test set |
| **TEST_SP_ID** | 500 | Single-preference In-Domain test set |
| **TEST_SP_OD** | 500 | Single-preference Out-of-Domain test set |
| **TEST_PICKAPIC** | 894 | Evaluation on Pick-a-Pic distribution |

**Access the dataset here:** 👉 [ZhouSimple/PreferImg-Bench](https://huggingface.co/datasets/ZhouSimple/PreferImg-Bench)

<br/>

## Model Release

We have released the **PreferThinker** inference checkpoint on Hugging Face:

**Access the model here:** [ZhouSimple/PreferThinker_model](https://huggingface.co/ZhouSimple/PreferThinker_model)

You can also download or load the checkpoint with the Hugging Face model id:

```bash
ZhouSimple/PreferThinker_model
```

<br/>

## 🛠️ Installation

1. Clone this repository and navigate to the folder:
```bash
git clone [https://github.com/your-username/PreferThinker.git](https://github.com/your-username/PreferThinker.git)
cd PreferThinker

```

2. Create a conda environment and install the dependencies:

```bash
conda create -n preferthinker python=3.11
conda activate preferthinker

bash setup.sh

```

## 🚀 Evaluation

You can use the provided `eval_PreferThinker.py` script to evaluate the model on our PreferImg-Bench. The script supports distributed evaluation using `torchrun`.

### Running the Evaluation

To evaluate the model on a specific test set (e.g., `PICKAPIC.json`), run the following command. Make sure to adjust `--nproc_per_node` according to the number of GPUs you have available:

```bash
torchrun --nproc_per_node=8 eval_PreferThinker.py \
    --model_path "ZhouSimple/PreferThinker_model" \
    --data_path "/path/to/PreferImg-Bench/data/PICKAPIC/PICKAPIC.json" \
    --image_root "/path/to/PreferImg-Bench/data/PICKAPIC/images" \
    --output_dir "./logs" \
    --run_name "PreferThinker_eval" \
    --batch_size 8

```

### Key Arguments

* `--model_path`: Path to the pre-trained or finetuned model directory.
* `--data_path`: Path to the specific dataset JSON file you want to evaluate.
* `--image_root`: Root directory containing the images referenced in the dataset.
* `--output_dir`: Directory where the evaluation logs and JSON results will be saved.
* `--batch_size`: Batch size per GPU (default: 8).

Once the evaluation is complete, the script will automatically calculate the accuracy and save the detailed predictions (including predicted preference profiles and rationales) to the specified `--output_dir`.

## 📂 Data Release

We are incrementally releasing the training corpus for **PreferThinker**, specifically designed for **CoT-based personalized preference assessment**.

### 📊 Current Status
* **Subset:** 10K image-text pairs (Single-preference focus).
* **Methodology:** CoT-enhanced preference assessment data.
* **Hugging Face:** [🤗 ZhouSimple/PreferThinker-Dataset](https://huggingface.co/datasets/ZhouSimple/PreferThinker-Dataset)


> The remaining data will be released sequentially.

<br/>

## 🏋️ Training

We also provide the training pipeline for **PreferThinker** in [src/open-r1-multimodal/src/open_r1](./src/open-r1-multimodal/src/open_r1). The current released script targets **single-preference training** with **GRPO**.

### Training Entry

The main training files are:

* `src/open-r1-multimodal/src/open_r1/preferthinker_singlepref_grpo_run.sh`
* `src/open-r1-multimodal/src/open_r1/preferthinker_singlepref_grpo.py`

Before launching training, please edit the paths in `preferthinker_singlepref_grpo_run.sh`:

```bash
DATA_PATHS="/path/to/your/dataset.jsonl"
IMAGE_FOLDERS="/path/to/your/images"

MODEL_PATH="/path/to/your/model/Qwen2.5-VL-7B-Instruct"
SBERT_MODEL_PATH="/path/to/your/model/all-MiniLM-L6-v2"
DREAMSIM_CACHE_DIR="/path/to/your/dreamsim_cache"
DEEPSPEED_CONFIG="/path/to/your/deepspeed_config/zero3.json"
```

Then launch training with:

```bash
cd src/open-r1-multimodal/src/open_r1
bash preferthinker_singlepref_grpo_run.sh
```

### What the Training Script Does

The released training pipeline uses `torchrun` + `DeepSpeed` to optimize a multimodal preference reasoning model with GRPO. In the default script:

* `preferthinker_singlepref_grpo.py` loads one or more JSONL training files from `--data_file_paths`.
* `--image_folders` should align with `--data_file_paths` one-by-one, separated by `:`.
* The script builds multimodal prompts from the user's historical preferred images, non-preferred images, and the candidate pair (`Image A` / `Image B`).
* The default reward combination is:
  * `accuracy`
  * `prefer_attributes`
  * `non-prefer_attributes`
  * `format`
* Auxiliary evaluators include **SBERT** and **DreamSim**, so their local paths should be prepared in advance.

### Expected Data Format

Each line of the training file should be a JSON object. The current code expects fields consistent with the released pipeline, including:

* `conversations`
* `imageA`, `imageB`
* `prefered_images`, `non_prefered_images`
* `prefer_attributes`, `non_prefer_attributes`
* `user_id`
* `reference_prompt_list`

In particular:

* `conversations[0]["value"]` is used as the input question.
* `conversations[1]["value"]` is used to extract the target answer.
* Image file names are joined with `--image_folders` to form absolute image paths.

## 📘 Citation
If you find our work helpful for your own research or work, please cite our paper.
```
@article{xu2025preferthinker,
  title={PreferThinker: Reasoning-based Personalized Image Preference Assessment},
  author={Xu, Shengqi and Zhou, Xinpeng and Zhang, Yabo and Liu, Ming and Liang, Tao and Zhang, Tianyu and Bai, Yalong and Wu, Zuxuan and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2511.00609},
  year={2025}
}
* For historical preferred / non-preferred images, the current implementation uses up to the first 5 images in each list.

