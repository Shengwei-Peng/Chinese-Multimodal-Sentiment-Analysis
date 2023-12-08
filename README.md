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
python3 main.py
```

## Result

- SIMS

| Model |Has0_acc_2 |Has0_F1_score |Non0_acc_2 |Non0_F1_score |Acc_3 |F1_score_3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ef_lstm |69.37 |56.82 |63.92 |49.85 |54.27 |38.18 |
| lf_dnn |79.87 |79.97 |56.70 |55.27 |70.20 |65.29 |
| tfn |75.32 |75.66 |53.56 |52.79 |65.95 |62.04 |
| lmf |77.99 |77.59 |57.06 |53.83|66.87 |62.46 |
| mfn |78.25 |78.08 |56.96 |54.14 |67.57 |
| graph_mfn |79.21 |78.92 |57.99 |54.66 |68.44 |63.44 |
| mult |78.07 |78.07 |56.34 |54.26 |68.27 |64.23 |
| misa |78.07 |77.70 |57.27 |53.99 |67.05 |60.98 |
| mlf_dnn |80.79 |80.59 |58.19 |55.55 |70.37 |65.94 |
| mtfn |81.23 |81.24 |56.91 |55.29 |70.28 |66.44 |
| mlmf |81.45 |81.62 |56.60 |55.66 |71.60 |70.45 |
|**Ours** |  |  |  |  |**72.87** |**71.03** |
## Contributing

We welcome contributions to the Chinese-Multimodal-Sentiment-Analysis repository. If you have suggestions, bug reports, or want to contribute code or documentation, please submit a pull request or open an issue.

## Acknowledgments

This repository builds upon the work and findings of various research papers and datasets, including the CH-SIMS dataset and associated research. We thank all the contributors and researchers in the field for their valuable insights and contributions.
