# Chinese-Multimodal-Sentiment-Analysis

## Introduction

Chinese-Multimodal-Sentiment-Analysis is a comprehensive repository dedicated to advancing the field of sentiment analysis using multimodal data inputs, focusing on Chinese language data. This repository houses code, datasets, and models that integrate and analyze textual, audio, and visual information to understand and predict sentiments in Chinese multimedia content.

## Objective

The primary objective of this repository is to provide resources and tools for researchers and practitioners to perform sentiment analysis in Chinese, leveraging the power of multimodal learning techniques. This includes handling complex linguistic phenomena and capturing nuanced emotional expressions.

## Installation

To get started with Chinese-Multimodal-Sentiment-Analysis, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Chinese-Multimodal-Sentiment-Analysis.git
cd Chinese-Multimodal-Sentiment-Analysis
pip install -r requirements.txt
```

## Usage

```python
python3 main.py \
  --batch_size 64 \
  --lr 5e-5 \
  --eposhs 100 \
  --early_stop 20 \
  --model_save_to ./model  
```

## Results

- CH-SIMS

| Model  |Acc_3 |F1_score_3 |
| :---: | :---: | :---: |
| ef_lstm  |54.27 |38.18 |
| lf_dnn |70.20 |65.29 |
| tfn |65.95 |62.04 |
| lmf |66.87 |62.46 |
| mfn |54.14 |67.57 |
| graph_mfn  |68.44 |63.44 |
| mult |68.27 |64.23 |
| misa |67.05 |60.98 |
| mlf_dnn |70.37 |65.94 |
| mtfn  |70.28 |66.44 |
| mlmf  |71.60 |70.45 |
|**Ours** |**72.87** |**71.03** |
## Contributing

We welcome contributions to the Chinese-Multimodal-Sentiment-Analysis repository. If you have suggestions, bug reports, or want to contribute code or documentation, please submit a pull request or open an issue.

## Acknowledgments

This repository builds upon the work and findings of various research papers and datasets, including the CH-SIMS dataset and associated research. We thank all the contributors and researchers in the field for their valuable insights and contributions.

