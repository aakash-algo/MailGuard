# MailGuard: GPT-2 Spam Detection Engine

MailGuard is a spam detection system built by fine-tuning a pretrained GPT-2 (124M) model for binary spam classification. The project adapts the original GPT-2 architecture, which is designed for next-token prediction, into a spam vs ham classifier.

The model was trained using a custom PyTorch pipeline and achieved strong performance across train, validation, and test datasets.

---

## Results

* **Train Accuracy:** 97.2%
* **Validation Accuracy:** 97.3%
* **Test Accuracy:** 95.7%

---

## Features

* Fine-tuned a pretrained GPT-2 (124M) model for spam classification
* Implemented a GPT architecture with:

  * 12 Transformer blocks
  * Byte Pair Encoding (BPE) tokenization
  * Token embeddings and positional embeddings
  * Multi-head masked self-attention
  * Shortcut / residual connections
  * Layer Normalization
  * Dropout
  * Feed-forward layers with GELU activation
* Modified the GPT-2 architecture for classification fine-tuning
* Integrated Top-k sampling and Temperature scaling
* Built an end-to-end PyTorch training and evaluation pipeline

---

## Model Architecture

The spam classifier is based on GPT-2 Small (124M parameters).

### Architecture Components

1. **Tokenization**

   * Input text is tokenized using Byte Pair Encoding (BPE)
   * Sequences are converted into token IDs before being passed into the model

2. **Embedding Layer**

   * Token embeddings convert token IDs into dense vector representations
   * Positional embeddings preserve sequence order information

3. **Transformer Blocks**

   * 12 stacked Transformer decoder blocks
   * Each block contains:

     * Multi-head masked self-attention
     * Feed-forward network with GELU activation
     * Layer Normalization
     * Residual connections
     * Dropout

4. **Classification Layer**

   * GPT-2 hidden representations are passed through a classification head
   * Final output predicts:

     * Spam
     * Ham

---

## Training Pipeline

The project includes:

* Data preprocessing
* Tokenization using GPT-2 tokenizer
* Train, validation, and test split
* Fine-tuning pretrained GPT-2 weights
* Classification training using PyTorch
* Accuracy tracking on train, validation, and test datasets

---

## Sample Predictions

| Input Text                                | Predicted Label |
| ----------------------------------------- | --------------- |
| Win a free lottery ticket now             | Spam            |
| Let's meet for lunch tomorrow             | Ham             |
| Urgent! Your account has been compromised | Spam            |
| Please send the project files by evening  | Ham             |

---

## Tech Stack

* Python
* PyTorch
* GPT-2
* NumPy
* Pandas

---

## Key Learnings

* Learned how GPT-2 can be adapted for text classification tasks
* Understood masked self-attention in decoder-only Transformers
* Explored BPE tokenization and embedding layers
* Built a classification fine-tuning pipeline using PyTorch
* Gained experience with Transformer architectures and evaluation workflows

---

## Author

Aakash Nath
