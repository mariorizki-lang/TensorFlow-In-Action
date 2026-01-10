# Chapter 15: TFX and MLOps

## Production-Quality Machine Learning Pipelines

---

## Chapter Overview

Chapter 15, the final chapter, focuses on **TensorFlow Extended (TFX)** - a framework for building production-quality machine learning pipelines. This chapter covers the complete end-to-end workflow from data handling through model deployment and serving.

---

## Key Topics

### 1. Introduction to TFX

#### What is TFX?
- **Comprehensive Framework:** End-to-end ML workflow
- **Production-Ready:** Designed for large-scale systems
- **Standardized Components:** Reusable pipeline building blocks
- **Best Practices:** Incorporates ML engineering best practices
- **Orchestration:** Works with multiple orchestrators (Airflow, Kubeflow, Beam)

#### Why TFX?

**Manual Approach Problems:**
- Write custom scripts for each step
- Error-prone (data inconsistencies)
- Difficult to maintain
- No monitoring or validation
- Hard to reproduce

**TFX Advantages:**
- Standardized components
- Built-in validation and checks
- Production monitoring
- Reproducibility
- Scalability

#### MLOps Lifecycle:

```
Data Collection
    ↓
Data Validation (check quality, schema)
    ↓
Data Preprocessing (feature engineering)
    ↓
Model Training (hyperparameter tuning)
    ↓
Model Evaluation (compare with baseline)
    ↓
Model Deployment (load for serving)
    ↓
Model Serving (API endpoints)
    ↓
Monitoring (track performance)
    ↓
(Loop: Retrain when performance degrades)
```

---

### 2. TFX Components and Data Pipeline

#### ExampleGen Component:

**Purpose:** Load raw data from various sources

**Inputs:**
- CSV files
- Image directories
- TFRecord files
- Databases

**Output:**
- Standardized ExampleProto format
- Train and eval splits

#### StatisticsGen Component:

**Purpose:** Analyze data distributions

**Computes:**
- Column statistics (mean, std, min, max)
- Value frequencies
- Missing value counts
- Type inference

**Output:**
- Feature statistics for every column
- Used by schema inference
- Detects anomalies

#### SchemaGen Component:

**Purpose:** Define expected data schema

**Infers:**
- Data types (int, string, float)
- Value ranges
- Vocabulary for categorical
- Cardinality

**Output:**
- Schema defining valid data structure
- Foundation for validation
- Prevents training on invalid data

#### ExampleValidator Component:

**Purpose:** Quality control on data

**Checks:**
- Data matches schema
- No unexpected value distributions
- Detects data drift
- Identifies anomalies

**Output:**
- Validation report
- Flagged anomalies
- Warnings if data changed significantly

#### Transform Component:

**Purpose:** Feature engineering and preprocessing

**Functionality:**
- Scale/normalize features
- Encode categorical variables
- Create polynomial features
- Handle missing values
- Remove outliers

**Advantages:**
- TensorFlow Transform (tf.Transform)
- Graph-based: Same preprocessing train and serve
- Prevents training-serving skew
- Scalable to large datasets

#### Data Pipeline Diagram:

```
Raw Data
  ↓
ExampleGen (Load and standardize)
  ↓
StatisticsGen (Analyze distribution)
  ├─→ SchemaGen (Define schema)
  │     ↓
  └─→ ExampleValidator (Check quality)
        ↓
      Transform (Feature engineering)
        ↓
    Ready for Training
```

---

### 3. Model Training with TFX Trainer

#### Trainer Component:

**Purpose:** Train model with standardized interface

**Advantages:**
- Consistent training across projects
- Automatic evaluation
- Model versioning
- Loss and metrics logging

#### Trainer Implementation:

```python
def run_fn(fn_args):
    """Main training function"""
    
    # Load data
    train_dataset = fn_args.train_data_uri
    eval_dataset = fn_args.eval_data_uri
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    model.fit(train_dataset, epochs=10,
              validation_data=eval_dataset)
    
    # Save model
    model.save(fn_args.serving_model_dir)
```

#### Model Saving - SavedModel Format:

```
saved_model/
├── assets/                    (additional files)
├── variables/
│   ├── variables.data
│   └── variables.index
└── saved_model.pb            (computation graph)
```

**Advantages:**
- Language-independent
- Platform-independent
- Version-controllable
- Can serve via TensorFlow Serving

---

### 4. Model Evaluation

#### Evaluator Component:

**Purpose:** Assess model quality and compare with baseline

**Functions:**
- Compute metrics on eval set
- Compare with previous best model
- Blessing: Approve for serving if improved

#### Model Blessing Logic:

```python
if new_model.accuracy > baseline_model.accuracy:
    blessing = "approved"
    push_to_serving = True
else:
    blessing = "rejected"
    push_to_serving = False

Log results and decide whether to deploy
```

---

### 5. TensorFlow Serving and Deployment

#### SavedModel Format:

Standard format for serving:
- Contains computation graph
- Model weights/variables
- Input/output signatures
- Asset files

**Advantages:**
- TensorFlow Serving compatible
- Language/framework agnostic
- Versioning support
- Scalable

#### TensorFlow Serving Setup:

```
models/
└── model_name/
    ├── 1/              (version 1)
    │   ├── saved_model.pb
    │   └── variables/
    ├── 2/              (version 2)
    │   └── ...
    └── 3/              (version 3, latest)
        └── ...
```

#### REST API Example:

```bash
# Start server
tensorflow_model_server \
    --port=8500 \
    --rest_api_port=8501 \
    --model_name=spam_classifier \
    --model_base_path=/models/spam_classifier

# Query model
curl -X POST http://localhost:8501/v1/models/spam_classifier:predict \
    -H "Content-Type: application/json" \
    -d '{
        "instances": [
            {"text_input": "Buy cheap products now!"}
        ]
    }'

# Response
{
    "predictions": [
        [0.95]  # 95% probability of spam
    ]
}
```

#### gRPC API (High Performance):

```python
import grpc
from tensorflow_serving.apis import predict_pb2

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'spam_classifier'
request.model_spec.signature_name = 'serving_default'

result = stub.Predict(request, 10.0)
```

---

### 6. Docker Containerization

#### Why Docker?
- **Reproducibility:** Same environment everywhere
- **Scalability:** Easy to spin up replicas
- **Isolation:** No dependency conflicts
- **Cloud-Ready:** Deploy to Kubernetes, Cloud Run
- **Versioning:** Pin dependencies

#### Dockerfile:

```dockerfile
FROM tensorflow/serving:latest-gpu

COPY ./path/to/saved_model /models/model/1

ENV MODEL_NAME=model
ENV PORT=8500

EXPOSE 8500 8501

CMD ["tensorflow_model_server", \
     "--port=8500", \
     "--rest_api_port=8501", \
     "--model_name=${MODEL_NAME}", \
     "--model_base_path=/models/${MODEL_NAME}"]
```

#### Building and Running:

```bash
# Build image
docker build -t spam-classifier:1.0 .

# Run container
docker run \
    --gpus all \
    -p 8500:8500 \
    -p 8501:8501 \
    spam-classifier:1.0

# Test
curl http://localhost:8501/v1/models/spam_classifier:predict \
    -d '{"instances": [...]}'
```

---

### 7. Validation and Deployment Pipeline

#### Model Promotion Flow:

```
Train Model
  ↓
Evaluate (Compare with baseline)
  ├─ If not improved → Stop
  ├─ If improved → Continue
  │   ↓
  ├─ Validate (Shadow traffic test)
  │   ├─ Test on production-like data
  │   ├─ Monitor for errors
  │   ├─ Compare performance
  │   └─ If issues → Stop
  │   ↓
  ├─ Promote (Canary deployment)
  │   ├─ 1% traffic → 10% → 50% → 100%
  │   ├─ Monitor each stage
  │   ├─ Fast rollback if issues
  │   └─ If successful → Production
  │   ↓
  └─ Monitor (Production monitoring)
      ├─ Track predictions
      ├─ Monitor latency
      ├─ Check for errors
      └─ Schedule retraining
```

#### Validation Steps:

```python
# 1. Compare metrics
if (new_eval_metrics['accuracy'] > 
    baseline_eval_metrics['accuracy'] * 0.99):
    
    # 2. Test on edge cases
    if test_on_edge_cases(new_model):
        
        # 3. Shadow traffic (1% of real traffic)
        if shadow_traffic_test_passed(new_model):
            
            # 4. Canary (gradually increase traffic)
            deploy_canary(new_model, start_percentage=0.01)
```

---

### 8. Production Monitoring

#### Input Data Monitoring:

**What to Track:**
- Check distributions haven't changed
- Detect data drift (statistical tests)
- Monitor for missing values
- Track feature ranges

**Problem:** Data drift detected
→ Trigger retraining on new data

#### Model Predictions Monitoring:

**What to Track:**
- Monitor output distributions
- Track prediction latencies
- Count error rates
- Monitor for NaN/Inf outputs

**Problem:** Accuracy drops 5%
→ Investigate, plan retraining

#### System Performance Monitoring:

**What to Track:**
- Request latency (< target)
- Throughput (requests/sec)
- Error rates (< threshold)
- GPU/CPU utilization

**Problem:** Latency > target
→ Optimize or scale

#### Retraining Triggers:

**Scheduled:**
- Retrain daily/weekly regardless
- Ensures model stays fresh

**Data Drift:**
- Automatically detect changes
- Trigger retraining if threshold exceeded

**Performance Degradation:**
- Monitor evaluation metrics
- If accuracy drops → Retrain

**Manual:**
- Triggered by analysts
- For special events or issues

---

### 9. Complete TFX Pipeline

#### Pipeline Definition:

```python
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

def create_pipeline():
    example_gen = CsvExampleGen(input_base='data/')
    stats_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])
    validator = ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='preprocessing.py'
    )
    trainer = Trainer(
        module_file='trainer.py',
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema']
    )
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model']
    )
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=push_destination_pb2.PushDestination(
            filesystem=push_destination_pb2.PushDestination.Filesystem(
                base_directory='serving_model'
            )
        )
    )
    
    return pipeline.Pipeline(
        pipeline_name='spam_classifier_pipeline',
        pipeline_root='pipelines/',
        components=[
            example_gen, stats_gen, schema_gen, validator,
            transform, trainer, evaluator, pusher
        ],
        enable_cache=True
    )

# Run pipeline
BeamDagRunner().run(create_pipeline())
```

#### Pipeline Execution Flow:

```
Input: Raw data
    ↓
Step 1: ExampleGen (Load CSV, Split train/eval)
    ↓
Step 2: StatisticsGen (Analyze distributions)
    ↓
Step 3: SchemaGen (Infer schema from statistics)
    ↓
Step 4: ExampleValidator (Check data quality)
    ↓
Step 5: Transform (Feature engineering)
    ↓
Step 6: Trainer (Train model on transformed data)
    ↓
Step 7: Evaluator (Evaluate and compare baseline)
    ↓
Step 8: Pusher (Push to serving if blessed)
    ↓
Final: Model available at REST/gRPC API
```

---

## Key Concepts

#### Reproducibility:
- Same data → Same pipeline → Same model
- Version data, code, hyperparameters
- Enable rollback and comparison

#### Scalability:
- Designed for billions of examples
- Works with distributed data
- Parallelizable components
- Cloud-native architecture

#### Monitoring:
- Built-in validation and checks
- Real-time monitoring capabilities
- Alert on anomalies
- Track metrics over time

#### Automation:
- Orchestrated workflows
- Automatic retraining
- Continuous deployment
- Minimal manual intervention

---

## Production ML vs Research ML

| Aspect | Research | Production |
|--------|----------|-----------|
| Data | Static, fixed | Streaming, continuous |
| Model | Single version | Multiple versions |
| Deployment | One-time | Continuous updates |
| Monitoring | Manual checks | Automated monitoring |
| Testing | Ad-hoc | Comprehensive |
| Serving | Batch | Real-time |
| Latency | Not critical | < 100ms typical |
| Reliability | Best effort | 99.9%+ uptime |
| Explainability | Nice-to-have | Required |

---

## Conclusion

Chapter 15 demonstrates:
- **TFX provides complete solution:** Data to serving pipeline
- **MLOps is essential:** Not optional for production systems
- **Automation is key:** Reduces errors and manual work
- **Monitoring drives improvement:** Continuous feedback loops

From prototyping to large-scale production serving millions of users, TFX provides the tools and best practices for sustainable, reliable ML systems.

---

## Learning Outcomes

After studying Chapter 15, you should understand:
- TFX component architecture and data flow
- How to build end-to-end ML pipelines
- Data validation and schema management
- Feature engineering with tf.Transform
- Model training with TFX Trainer
- Model evaluation and blessing
- TensorFlow Serving setup and deployment
- Docker containerization for reproducibility
- Production monitoring and alerting
- Retraining triggers and automation
- Difference between research and production ML

---

## From Prototype to Production

| Stage | Focus | Tool | Output |
|-------|-------|------|--------|
| **Research** | Experimentation | Jupyter, TensorBoard | Proof of concept |
| **Prototype** | Implementation | TensorFlow, Keras | Working model |
| **Validation** | Testing | TFX Evaluator | Blessed model |
| **Deployment** | Serving | TensorFlow Serving | Production API |
| **Monitoring** | Health | TensorBoard, Logs | Metrics, Alerts |
| **Retraining** | Improvement | TFX Pipeline | Updated model |

---

## Final Thoughts on TensorFlow in Action

**The complete journey through all 15 chapters:**

- **Part 1:** Foundations of TensorFlow and deep learning
- **Part 2:** Real-world applications (vision, NLP)
- **Part 3:** Advanced models and production systems

**Transforms readers from:**
- "How do I implement?" → "How do I deploy and maintain?"
- Single models → Complete ML engineering systems
- Prototype code → Production pipelines

---

*TensorFlow in Action - Chapter 15 (Final Chapter)*
*Last Updated: January 2026*
*Complete Series: Chapters 1-15*