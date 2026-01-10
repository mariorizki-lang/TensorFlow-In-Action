# Chapter 13: Transformers

## State-of-the-Art Sequence Models

---

## Chapter Overview

Chapter 13 provides a deep dive into **Transformer architecture** and demonstrates practical applications through two use cases: spam classification with BERT and question answering with DistilBERT using Hugging Face Transformers library.

---

## Key Topics

### 1. Transformer Architecture Deep Dive

#### Evolution Context:
- **Chapter 11:** Basic Seq2Seq (information bottleneck)
- **Chapter 12:** Seq2Seq + Attention (dynamic focusing)
- **Chapter 13:** Transformers (fully attention-based, parallelizable)

#### Revolutionary Features:
- **Parallel Processing:** Can process entire sequence simultaneously (unlike RNNs)
- **Attention-Only:** Replaces RNN with pure attention mechanisms
- **Scalability:** Enables training on massive datasets
- **Transfer Learning:** Pre-trained models on billions of examples

---

### 2. Core Components of Transformer

#### Self-Attention Layer:
Input sequence: x = [x1, x2, ..., xn]

**Step 1: Project to Query, Key, Value**
```
Q = x * W_Q
K = x * W_K
V = x * W_V
```

**Step 2: Compute Attention**
```
scores = Q * K^T / sqrt(d_model)
```

**Step 3: Apply Softmax**
```
attention_weights = softmax(scores)
```

**Step 4: Weighted Sum**
```
output = attention_weights * V
```

Result: Each position attends to all other positions

#### Multi-Head Attention:
Different heads learn different relationships:
- **Head 1:** Short-range dependencies
- **Head 2:** Long-range dependencies
- **Head 3:** Syntax structure
- **Head 4:** Semantics
- **Typical:** 8 heads, each 64 dimensions

#### Feed-Forward Networks:
Position-wise MLPs:
- First dense layer: Expands (d_model → 4*d_model)
- ReLU activation: Non-linearity
- Second dense layer: Projects back (4*d_model → d_model)

#### Residual Connections and Layer Normalization:
```
Skip connection: x → Add Layer(x) → LayerNorm
```
- Enables deep networks
- Prevents training collapse

#### Positional Encoding:
- Attention is permutation-invariant (position-blind)
- Positional encoding adds position information
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
```
- Result: Each position has unique signature

---

### 3. BERT for Spam Classification

#### BERT Overview:
- **Name:** Bidirectional Encoder Representations from Transformers
- **Type:** Encoder-only Transformer
- **Pre-training:** Trained on Wikipedia + BookCorpus
- **Bidirectional:** Attends to left AND right context
- **Size:** Base model 110M parameters

#### Pre-training Objectives:

**Masked Language Model (MLM):**
- Original: "The quick brown fox jumps"
- Masked (15%): "The [MASK] brown fox jumps"
- Task: Predict masked words from context
- Result: Learns bidirectional context understanding

**Next Sentence Prediction (NSP):**
- Input: [CLS] sent_A [SEP] sent_B [SEP]
- Task: Predict if sent_B follows sent_A
- Training: 50% consecutive (positive), 50% random (negative)
- Result: Learns sentence relationships

#### Fine-tuning for Spam Classification:
1. Load pre-trained BERT
2. Add classification head: [CLS] token → Dense → Softmax
3. Fine-tune with small learning rate (2e-5)
4. Few epochs (3-5)
5. Update all parameters

#### Why BERT Works:
- Massive pre-training captures language understanding
- [CLS] token learns sentence-level representation
- Transfer learning: Minimal fine-tuning needed
- Bidirectional context critical for understanding

---

### 4. DistilBERT for Question Answering

#### DistilBERT Characteristics:
- **Size:** 40% smaller than BERT (66M parameters)
- **Speed:** 60% faster inference
- **Method:** Knowledge distillation (student mimicking teacher)
- **Quality:** Similar performance with smaller model

#### Question Answering Task:
Input Format: [CLS] question [SEP] context [SEP]

Example:
```
[CLS] What is AI? [SEP] AI is machine learning... [SEP]
```

Output:
```
Start position: Word 3
End position: Word 6
Answer: context[3:6] = "machine learning"
```

#### Model Architecture:
```
Token Embeddings + Position Embeddings
    ↓
DistilBERT Layers (6 layers, 768 hidden units)
├── Self-Attention
├── Feed-Forward
└── Residual + LayerNorm
    ↓
Output Layer (Splits into 2 heads):
├── Start position logits → softmax
└── End position logits → softmax
```

#### Data Processing:
1. Tokenize
2. Find answer span in context
3. Map answer to token positions
4. Handle sequences > 512 tokens with sliding window

#### Training Process:
```
For each batch:
  1. Pass through DistilBERT
  2. Get start and end logits
  3. Compare with true positions
  4. Compute loss
  5. Backpropagate and update
```

Metrics:
- **Exact Match (EM):** Exact string match
- **F1 Score:** Overlapping tokens
- **Dev set:** SQuAD dataset standard

---

### 5. Hugging Face Transformers Library

#### Overview:
- **Purpose:** Simplified access to pre-trained models
- **Models:** 50,000+ pre-trained models
- **Community:** Active, frequently updated
- **Integration:** TensorFlow, PyTorch, JAX

#### Model Categories:
- **Encoder-only:** BERT, RoBERTa (Classification, tagging)
- **Decoder-only:** GPT-2, GPT-3 (Text generation)
- **Encoder-Decoder:** BART, T5 (Translation, summarization)
- **Vision:** ViT, CLIP (Image classification, multimodal)

#### High-Level API (Pipelines):
```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification")
result = classifier("This movie is great!")
# Output: [{'label': 'POSITIVE', 'score': 0.99}]

# Question answering
qa = pipeline("question-answering")
result = qa(question="What is AI?", context="AI is...")
```

#### Low-Level API (Fine-tuning):
```python
from transformers import AutoTokenizer, TFAutoModel

# Load pre-trained
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModel.from_pretrained("distilbert-base-uncased")

# Tokenize and forward pass
inputs = tokenizer("Hello world", return_tensors="tf")
outputs = model(inputs)
```

---

### 6. Transformer Advantages

#### Over Seq2Seq + Attention:
- **Parallelization:** Process entire sequence simultaneously
- **Longer sequences:** No vanishing gradients
- **Pre-training:** Billions of examples enable transfer learning
- **Performance:** State-of-the-art on most NLP benchmarks

---

## Conclusion

Chapter 13 demonstrates:
- **Transformers revolutionized NLP:** Now state-of-the-art
- **Pre-training is crucial:** Transfer learning enables few-shot learning
- **Hugging Face democratizes:** Access to powerful models easily
- **Practical deployment:** Fine-tuning is feasible for practitioners

---

## Learning Outcomes

After studying Chapter 13, you should understand:
- Core Transformer architecture components
- Self-attention and multi-head attention mechanisms
- Positional encoding and why it's necessary
- BERT's pre-training objectives and fine-tuning approach
- DistilBERT's knowledge distillation
- Question answering as a span selection task
- Hugging Face library for practical implementations
- Transfer learning and its importance

---

## Key Architecture Evolution

| Model | Key Innovation | Processing | Pre-training |
|-------|--------------|-----------|--------------|
| Seq2Seq | Encoder-Decoder | Sequential | None |
| Seq2Seq + Attention | Attention mechanism | Sequential | None |
| Transformer | Self-attention only | Parallel | Massive (Billions) |
| BERT | MLM + NSP | Parallel | Wikipedia + Books |
| DistilBERT | Knowledge distillation | Parallel | Smaller but effective |

---

*TensorFlow in Action - Chapter 13*
*Last Updated: January 2026*