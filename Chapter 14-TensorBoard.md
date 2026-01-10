# Chapter 14: TensorBoard

## Visualization and Monitoring for Deep Learning

---

## Chapter Overview

Chapter 14 introduces **TensorBoard**, a powerful visualization and monitoring dashboard integral to TensorFlow's ecosystem. This chapter covers tools and techniques for understanding model behavior during training.

---

## Key Topics

### 1. Introduction to TensorBoard

#### Purpose:
- **Visualization Dashboard:** Web-based interface for model monitoring
- **Real-time Tracking:** Monitor metrics during training
- **Debugging:** Identify and troubleshoot training issues
- **Profiling:** Detect performance bottlenecks
- **Interpretation:** Understand model structure and behavior

#### Why TensorBoard Matters:
- Deep learning models are "black boxes" without visualization
- Training can take hours/days requiring monitoring
- Hyperparameter tuning requires insights into behavior
- Performance optimization requires profiling data
- Explainability is critical for production systems

---

### 2. Data Visualization with TensorBoard

#### Scalar Metrics:

**Purpose:** Track single numerical values over time

**Examples:**
- Training loss
- Validation accuracy
- Learning rate
- Custom metrics

**Visualization:** Line graphs showing trends

#### Histograms:

**Purpose:** Visualize distributions of values

**Examples:**
- Weight distributions per layer
- Activation distributions
- Gradient distributions

**Interpretation:**
- **Dying ReLUs:** Histogram mostly zeros
- **Exploding gradients:** Sudden spikes
- **Healthy training:** Gradual distribution shift

#### Images:

**Purpose:** Visualize image data

**Examples:**
- Input training images
- Generated predictions
- Attention heatmaps
- Feature maps
- Grad-CAM visualizations

#### Text:

**Purpose:** Log text-based information

**Examples:**
- Generated text samples
- Model predictions
- Confusion matrix representation

#### Embeddings:

**Purpose:** Visualize high-dimensional vectors

**Examples:**
- Word embeddings (language models)
- Feature vectors (computer vision)
- Context representations

**Visualization Methods:**
- **t-SNE:** Preserves local structure
- **PCA:** Linear projection
- **UMAP:** Preserves global structure

---

### 3. Model Monitoring and Tracking

#### Training Metrics to Track:

**Essential:**
1. Training Loss (should decrease mostly)
2. Validation Loss (should decrease then plateau)
3. Training Accuracy (should increase toward 1.0)
4. Validation Accuracy (plateau point = optimal training)
5. Learning Rate (monitor schedule changes)

#### Keras Callbacks for TensorBoard:

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch',
    profile_batch=(100, 110)
)

model.fit(train_data, epochs=10, 
          validation_data=val_data,
          callbacks=[tensorboard_callback])
```

#### Identifying Training Issues:

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Underfitting | Both train & val loss high | Larger model, more epochs |
| Overfitting | Train ↓, Val ↑ | Regularization, dropout |
| High LR | Loss unstable/NaN | Decrease learning rate |
| Low LR | Very slow improvement | Increase learning rate |
| Dying ReLU | Hidden zeros | Change activation function |

---

### 4. Custom Metrics with tf.summary

#### Writing Custom Metrics:
Track gradients, learning rate, per-layer statistics, custom domain metrics

Key Functions:
- `tf.summary.scalar()` - Single values
- `tf.summary.histogram()` - Distributions
- `tf.summary.image()` - Visual data
- `tf.summary.text()` - Text logs

---

### 5. Profiling for Performance Bottleneck Detection

#### Why Profile?
- Identify slow operations consuming most time
- Optimize data pipeline efficiency
- Detect GPU underutilization
- Reduce overall training time

#### TensorFlow Profiler:

Analyzes:
- **Op Statistics:** Which operations take longest
- **Input Pipeline:** Data loading efficiency
- **Device Utilization:** GPU/CPU usage percentage
- **Memory:** Peak consumption

#### Common Bottlenecks:
- Slow data loading (I/O bound)
- Model too large for GPU memory
- GPU underutilized (I/O bottleneck)
- Inefficient operations

---

### 6. Input Pipeline Optimization

#### tf.data Optimization:

**Key Techniques:**

1. **Caching:** Avoid re-reading from disk each epoch
2. **Prefetching:** Prepare next batch during current batch processing
3. **Parallel Operations:** Use num_parallel_calls for mapping
4. **Interleaving:** Mix multiple datasets

#### Optimized Pipeline Example:

```python
def optimized_dataset(dataset):
    return (dataset
        .cache()                    # Cache to memory
        .shuffle(buffer_size=1000)
        .batch(32)
        .map(preprocess,
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
```

#### Pipeline Performance Impact:
- **No optimization:** ~100ms per batch
- **With caching:** ~10ms per batch (first epoch)
- **With prefetching:** ~5ms per batch
- **With parallelization:** ~2ms per batch

---

### 7. Mixed Precision Training

#### Concept:
- Use float16 for forward/backward computation (faster)
- Keep variables in float32 for stability
- Loss scaling to prevent underflow
- Result: 2x speed, same accuracy

#### Benefits:
- ~2x faster computation
- ~50% memory reduction
- Same final accuracy
- Requires compute capability 7.0+ (V100, A100, etc.)

#### Implementation:

```python
from tensorflow.keras.mixed_precision import Policy

# Set mixed precision policy
policy = Policy('mixed_float16')
set_global_policy(policy)

# Train as normal
model.compile(...)
model.fit(...)
```

---

### 8. Embedding Visualization

#### Why Visualize Embeddings?
- Embedding spaces hard to understand (256-512 dimensions)
- Projection to 3D reveals structure
- Semantic relationships become visible

#### T-SNE Projection:
```
High-dimensional embeddings → T-SNE → 3D
```

Result: Similar items cluster together

#### Interpretation:
- **Clusters:** Words with similar meaning
- **Distances:** Semantic similarity
- **Relationships:** Linear algebra relationships
  - Vector(King) - Vector(Man) + Vector(Woman) ≈ Vector(Queen)

---

### 9. Best Practices for Logging

#### What to Log:

**Priority 1 (Essential):**
- Training loss
- Validation loss
- Validation accuracy

**Priority 2 (Important):**
- Learning rate
- Gradient norms
- Weight distributions

**Priority 3 (Optional):**
- Activation distributions
- Input batch statistics

#### Logging Frequency:

| Frequency | Content | Cost |
|-----------|---------|------|
| Every Step | Training loss, learning rate | Low |
| Every N Steps | Gradient stats, weight distributions | Medium |
| Every Epoch | Validation metrics, embeddings | High |

#### Organization:
Use naming convention: "category/metric"
```
Good:     "train/loss", "validation/loss", "learning_rate"
Bad:      "loss" (ambiguous), "acc" (abbreviation)
```

---

### 10. Launching TensorBoard

```bash
# Basic
tensorboard --logdir=./logs

# With port specification
tensorboard --logdir=./logs --port=6006

# Open in browser: http://localhost:6006
```

#### Features in Web Interface:
- **SCALARS:** Time-series metrics
- **IMAGES:** Visual inputs and outputs
- **DISTRIBUTIONS:** Weight/activation histograms
- **HISTOGRAMS:** Change over time
- **GRAPHS:** Computation graph visualization
- **EMBEDDINGS:** 3D embedding projections
- **PROFILE:** Performance profiling data

---

## Conclusion

Chapter 14 demonstrates that:
- **TensorBoard is essential:** Transforms model training from black box
- **Visualization enables understanding:** See what model is learning
- **Profiling guides optimization:** Data-driven performance improvements
- **Monitoring catches issues:** Early detection of problems

---

## Learning Outcomes

After studying Chapter 14, you should understand:
- TensorBoard's visualization capabilities
- How to log different metric types
- Identifying training issues from graphs
- Profiling to find performance bottlenecks
- Optimizing data pipelines effectively
- Mixed precision training advantages
- Embedding visualization techniques
- Best practices for monitoring production models

---

## Common Monitoring Checklist

When training a new model, monitor:
- [ ] Training loss decreasing
- [ ] Validation loss following similar trend
- [ ] No NaN or Inf values appearing
- [ ] Learning rate schedule working correctly
- [ ] Weight distributions not exploding/vanishing
- [ ] Gradient norms reasonable
- [ ] GPU utilization > 80%
- [ ] Data loading not bottleneck
- [ ] Training progressing at expected speed
- [ ] Model checkpoint saving working

---

*TensorFlow in Action - Chapter 14*
*Last Updated: January 2026*