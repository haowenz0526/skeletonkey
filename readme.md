# Skeleton Key

This repository contains the implementation of the Skeleton Key method, a technique for improving transferability in adversarial examples.

## Overview

Skeleton Keys are carefully crafted adversarial examples that can maintain their effectiveness across different models. This repository provides tools for:

1. Generating Skeleton Keys using various techniques
2. Evaluating the transferability of adversarial examples across different models

## Requirements

```
pytorch
torchvision
numpy
matplotlib
tqdm
```

## Usage

### Generating Skeleton Keys

To generate skeleton keys, use the `skeleton_key.py` script:

```bash
python skeleton_key.py --model [MODEL_NAME] --dataset [DATASET] --attack [ATTACK_METHOD] --epsilon [EPSILON_VALUE] --output [OUTPUT_DIR]
```

#### Arguments:

- `--model`: Base model for generating the skeleton key (resnet18, vgg16, etc.)
- `--dataset`: Dataset to use (imagenet, cifar10, etc.)
- `--attack`: Attack method (pgd, fgsm, mifgsm, etc.)
- `--epsilon`: Perturbation budget (default: 0.03)
- `--output`: Directory to save the generated skeleton keys

#### Example:

```bash
python skeleton_key.py --model resnet50 --dataset imagenet --attack pgd --epsilon 0.05 --output ./keys
```

### Evaluating Transferability

To evaluate the transferability of the generated skeleton keys, use the `Transferability.py` script:

```bash
python Transferability.py --keys [KEYS_DIR] --target_models [MODEL1,MODEL2,...] --dataset [DATASET] --output [RESULTS_DIR]
```

#### Arguments:

- `--keys`: Directory containing the generated skeleton keys
- `--target_models`: Comma-separated list of target models to evaluate
- `--dataset`: Dataset to use for evaluation
- `--output`: Directory to save the evaluation results

#### Example:

```bash
python Transferability.py --keys ./keys --target_models resnet18,vgg16,densenet121 --dataset imagenet --output ./results
```

## Results Visualization

The evaluation results will be saved in the specified output directory. You can visualize the results using:

```bash
python visualize_results.py --results [RESULTS_DIR] --output [FIGURES_DIR]
```

## Additional Notes

- Ensure you have sufficient GPU memory for larger models and datasets
- For ImageNet, it's recommended to use a batch size of 16 or lower depending on available GPU memory
- The generation process can be time-consuming for large datasets

## Citation

If you use this code in your research, please cite:

```
@article{skeletonkey2023,
  title={Skeleton Key: Towards Robust Transferable Adversarial Examples},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## License

[MIT License](LICENSE)
