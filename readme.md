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
python skeleton_key.py
```


### Evaluating Transferability

To evaluate the transferability of the generated skeleton keys, use the `Transferability.py` script:

```bash
python Transferability.py
```

## Results Visualization

The evaluation results will be saved in the specified output directory. You can visualize the results using:

```bash
python visualize_results.py --results [RESULTS_DIR] --output [FIGURES_DIR]
```

## Additional Notes
reference code:
https://github.com/EthanRath/Game-Theoretic-Mixed-Experts

https://github.com/NVlabs/MambaVision/tree/main/mambavision

https://github.com/MzeroMiko/VMamba/tree/main

https://github.com/wzekai99/DM-Improves-AT
```

## License

[MIT License](LICENSE)
