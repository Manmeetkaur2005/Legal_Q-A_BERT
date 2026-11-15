# LegalQA: Privacy Policy Question Answering with BERT and Capsule Networks

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-required-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-BERT-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20in%20progress-yellow.svg)

An extractive question answering system for legal and privacy-policy text. Given a policy document and a natural-language question, the model predicts the answer span inside the document. The architecture combines a **BERT** encoder with a **Capsule Network**-style output head, implemented in **PyTorch**.

> **Status:** Ongoing research project. Results are preliminary and the manuscript is in preparation.

## Approach

1. **Data preparation:** load the PolicyQA dataset, normalize answers, validate every answer span against its context, remove duplicates and save the cleaned data.
2. **Tokenization:** encode question and context pairs with the `bert-base-uncased` tokenizer and map character offsets to token-level start and end labels.
3. **Model:** BERT encoder, then a capsule-style projection layer, then separate start and end token classifiers, with padding tokens masked in the logits.
4. **Training:** custom PyTorch `Dataset` and `DataLoader`, optimizer, training loop and checkpoint saving.
5. **Inference:** pick the best start and end positions from the logits and decode the answer text from the context.

### Model architecture

```
Question + Context
        |
BERT encoder (bert-base-uncased)
        |
Token representations
        |
Capsule-style projection layer
        |
   +----+----+
Start head  End head
   |         |
   +----+----+
        |
Answer span (start, end)
```

## Dataset

[`alzoubi36/policy_qa`](https://huggingface.co/datasets/alzoubi36/policy_qa) from Hugging Face. Each record contains `id`, `title`, `context`, `question` and `answers`.

Cleaning steps applied before training:

- Convert answer dictionaries into `primary_text`, `primary_start` and `primary_end`
- Verify that each span exactly matches the substring of `context`
- Remove exact duplicate records
- Strip leading and trailing whitespace
- Save the result to `policy_qa_cleaned.parquet`

## Getting Started

### Installation

```bash
git clone https://github.com/Manmeetkaur2005/Legal_Q-A_BERT.git
cd Legal_Q-A_BERT
pip install torch transformers datasets pandas pyarrow
```

### Usage

1. Open `main.ipynb` in Jupyter or Google Colab.
2. Run all cells in order. The notebook will:
   - load and clean the dataset
   - tokenize the question and context pairs
   - train the BERT + Capsule model (or reload a saved checkpoint)
   - save the trained weights as `bert_capsule_qa_weights.pt`
   - run a sample inference

### Example

```python
question = "What personal information does the website collect?"
context  = "<privacy policy text here>"

answer = predict_answer(question, context)
print(answer)
```
## Limitations

- Research prototype intended for learning and experimentation.
- Answer quality depends heavily on the dataset and training setup.
- Long legal documents may exceed BERT's 512-token limit and need chunking.
- This is **not** a legal compliance tool and is not a substitute for legal advice.



## Note on Model Weights

Trained weights (`.pt` files) are large and are not stored in this repository. Run the notebook to regenerate them.

## Author

**Manmeet Kaur**
B.Tech CSE, Guru Tegh Bahadur Institute of Technology
GitHub: [@Manmeetkaur2005](https://github.com/Manmeetkaur2005) | Email: meetka2005@gmail.com
