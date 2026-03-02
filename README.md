<div align="center">

# [ICLR 2026 🔥] PreferThinker: Reasoning-based Personalized Image Preference Assessment

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/ZhouSimple/PreferImg-Bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## :black_nib: TODO List<a name="todo"></a>

- [x] Release [PreferThinker paper](https://arxiv.org/pdf/2511.00609).
- [x] Release evaluation code.
- [x] Release PreferThinker-Bench.
- [ ] Release inference checkpoints.
- [ ] Release CoT-based personalized preference assessment dataset.
- [ ] Release training code.

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

## 🛠️ Installation

1. Clone this repository and navigate to the folder:
```bash
git clone [https://github.com/your-username/PreferThinker.git](https://github.com/your-username/PreferThinker.git)
cd PreferThinker

```

2. Create a conda environment and install the dependencies:

```bash
conda create -n preferthinker python=3.10
conda activate preferthinker

pip install -r requirements.txt

```

## 🚀 Evaluation

You can use the provided `eval_PreferThinker.py` script to evaluate the model on our PreferImg-Bench. The script supports distributed evaluation using `torchrun`.

### Running the Evaluation

To evaluate the model on a specific test set (e.g., `PICKAPIC.json`), run the following command. Make sure to adjust `--nproc_per_node` according to the number of GPUs you have available:

```bash
torchrun --nproc_per_node=8 eval_PreferThinker.py \
    --model_path "/path/to/your/PreferThinker/checkpoint" \
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
