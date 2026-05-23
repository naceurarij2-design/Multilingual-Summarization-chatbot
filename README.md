# Multilingual-Summarization-chatbot
Phase 1: Balanced Data Gathering &amp; Tokenization Pipeline
## 🚀 Phase 2: Broad-Spectrum LoRA Fine-Tuning & SFT Optimization

### 📋 Overview
In this phase, we implemented **Parameter-Efficient Fine-Tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** on top of the pre-trained `facebook/mbart-large-50` architecture. The primary objective of this phase was to train the model to ingest our context-aware sequential datasets (built in Phase 2) and output fluid, interconnected multi-lingual summaries with human-like transitions.

By utilizing LoRA, we successfully froze the core 1.3B parameters of the base model and isolated training to lightweight, specialized matrix adapters, reducing computational overhead while maximizing stylistic adaptation.

---

### ⚙️ Core Technical Implementation & Parameter Roles

To achieve highly cohesive, ChatGPT-grade abstractive flow and prevent disjointed text outputs, the fine-tuning loop was heavily optimized across three core dimensions:

#### 1. PEFT/LoRA Architecture Layout
* **Rank Selection (`r=32`) & Scaling (`lora_alpha=64`):** Doubled the rank capacity from standard baselines to allow the adapter layers sufficient memory to track complex grammatical markers and cross-topic transition cues.
* **Broad-Spectrum Layer Targeting:** Injected adapters across the entire cross-attention mechanism (`q_proj`, `v_proj`, `k_proj`, `out_proj`) rather than just Query/Value layers. This allows the model to deeply map historical context summaries directly to incoming text chunks.
* **Regularization (`lora_dropout=0.05`):** Prevented the network from overfitting or memorizing the text verbatim by randomly deactivating 5% of adapter pathways during training.

#### 2. Advanced Supervised Fine-Tuning (SFT) Hyperparameters
* **Learning Rate (`3e-4`) with Cosine Schedule:** Implemented a `cosine` decay scheduler paired with a `warmup_ratio=0.1`. The model safely warms up its attention grids for the first 10% of training, scales to its maximum rate, and smoothly decelerates toward the final epochs to secure crisp, stable convergence.
* **Weight Decay (`0.05`):** Applied a strict mathematical penalty to complex weights, forcing the model to favor dense, high-value vocabulary and preventing sentence bloating or repetition loops.
* **Gradient Accumulation (`4`):** Configured to accumulate gradients over 4 micro-batches, successfully simulating a stable, large batch size of 16 while operating within limited hardware environments.

#### 3. Generation Safeguards
* Overwrote global configuration metrics (`no_repeat_ngram_size=3`, `early_stopping=True`) to structurally ban the model from getting caught in infinite phrase duplication loops during validation.

---

### 📊 Key Outcomes
* **Trainable Parameters:** ~9.4M weights (only **0.68%** of the global model size), keeping the final adapter export incredibly lightweight (a few megabytes) while preserving the foundational multilingual knowledge of mBART-50.
* **Artifact Generation:** The script successfully checkpoints tracking data to Google Drive and exports a highly-optimized, localized adapter directory (`final_multilingual_adapter`) ready to be dynamically hot-swapped into our Phase 4 production interface.
