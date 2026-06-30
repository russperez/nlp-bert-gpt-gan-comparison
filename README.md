# NLP Model Comparison: BERT vs GPT-2 vs Text-GAN

This project implements and compares three distinct NLP architectures on the same dataset: BERT (bidirectional encoder for classification), GPT-2 (autoregressive generation), and a custom Text-GAN (adversarial generation). The goal is to evaluate how differently these architectures handle language representation, generation, and adversarial synthesis using standardized metrics.

## Objectives

- Fine-tune `bert-base-uncased` for binary sentiment classification.
- Fine-tune `distilgpt2` for causal language modeling and controlled text generation.
- Build and train a lightweight LSTM generator and CNN discriminator Text-GAN using Gumbel-Softmax for differentiable discrete text generation.
- Compare all three using precision, recall, F1, BLEU, ROUGE, perplexity, and training time per epoch.

## Dataset

Sentiment Analysis Dataset, sourced from Kaggle: [https://drive.google.com/file/d/168Wiqn6zwQWwibrYS_dNs3QWFnI7KyQu/view?usp=drive_link]

The dataset contains short, informal comments labeled Positive, Neutral, or Negative. Due to severe class imbalance (only 79 Negative examples out of roughly 6,700 rows), labels were collapsed into a binary Positive vs Not-Positive classification task for the BERT variant.

Cleaning steps applied: dropped malformed rows during CSV parsing, removed rows with missing comments or sentiment labels, stripped URLs and extra whitespace from text.

Split: 70 percent train, 15 percent validation, 15 percent test, stratified by label. See `splits/` for the exact files used.

## Model Variants

### 1. BERT (Text Classification)
Fine-tuned `bert-base-uncased` with a sequence classification head on binary sentiment labels. Evaluated using precision, recall, and F1 on the held-out test set.

### 2. GPT-2 (Causal Language Modeling)
Fine-tuned `distilgpt2` on the same comment text. Evaluated using perplexity (derived from validation loss) and BLEU and ROUGE scores comparing generated continuations against real held-out text.

### 3. Text-GAN (Adversarial Text Generation)
Custom architecture: an LSTM generator conditioned on random noise, using Gumbel-Softmax to keep token sampling differentiable, paired with a 1D-CNN discriminator trained to distinguish real comments from generated ones. Evaluated using discriminator accuracy, precision, recall, and F1, plus BLEU and ROUGE on generator output.

## How to Run

1. Open `notebook.ipynb` in Google Colab.
2. Set the runtime to GPU: Runtime, Change runtime type, select T4 GPU.
3. Run all cells in order. Cell 1 installs dependencies and downloads the dataset from Drive.
4. Outputs are saved automatically to a `results` folder and copied to a Google Drive folder for persistence across sessions.

## Key Findings

### Tokenization Differences
BERT uses WordPiece subword tokenization with a fixed vocabulary of about 30,000 tokens and bidirectional context. GPT-2 uses byte-pair encoding, which handles informal text and rare words differently. The Text-GAN uses a simple word-level vocabulary built directly from the training set, with no pretrained embeddings, which limits its ability to generalize to words it has not seen during training.

### Why GANs Struggle With Discrete Text
Standard GANs cannot backpropagate gradients through a hard sampling step from a softmax distribution to a discrete token, since that step is non-differentiable. This project worked around that using Gumbel-Softmax, which approximates discrete sampling with a differentiable relaxation. This introduces additional training instability compared to GPT-2, which is trained with straightforward maximum likelihood estimation. During training, the discriminator and generator losses showed [insert your actual observation, for example oscillation, one network dominating, or slow convergence], illustrating this instability directly.

### Metric Tradeoffs
BLEU and ROUGE reward n-gram overlap rather than fluency or coherence, which can undersell GPT-2's actual output quality and oversell repetitive GAN output that happens to match common phrases. Perplexity only applies to models with a tractable likelihood, such as GPT-2, and does not apply to the GAN.
