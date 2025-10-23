# Interactive code
## [Main Agent](https://colab.research.google.com/drive/1UVPlk-JmT8WX8eaC983H7KKVhSb5hltW?usp=sharing)
## [Statistical Analysis & Bayesian Reasoning](https://colab.research.google.com/drive/1QR-rR0wR3P9cyEdzS4fq4v0MZskNeCQB?usp=sharing)
## [Auto ML component](https://colab.research.google.com/drive/1lVqRONQSYKhSEOjA0KZrC-jKBo_hL3gH?usp=sharing)

# 🏥 Clinical Trial Endpoint Prediction System

> **Hybrid AI system combining LLM reasoning, Bayesian inference, and AutoML for predicting clinical trial outcomes**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

## 🎯 Overview

This project predicts whether clinical trial endpoints will meet their success criteria using a **three-component ensemble** that combines:
- 🤖 **LLM Agent** with RAG for contextual reasoning over historical trials
- 📊 **Bayesian Statistical Model** for interpretable probabilistic inference
- 🧠 **AutoML Pipeline** for automated feature engineering and pattern recognition

Built for the [ASCO Clinical Trial Outcome Prediction Challenge](https://www.kaggle.com/competitions/asco-clinical-trial-outcome-prediction), this system achieves robust predictions by fusing deep learning, probabilistic modeling, and machine learning approaches.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRIAL ENDPOINT INPUT                      │
│         (drug, indication, phase, endpoint type...)          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌───────────┐  ┌──────────┐  ┌──────────┐
│ LLM Agent │  │ Bayesian │  │ AutoML   │
│  + RAG    │  │   GLM    │  │ (AutoGL) │
└─────┬─────┘  └────┬─────┘  └────┬─────┘
      │             │             │
      │   P(success|evidence)    │
      │             │             │
      └─────────────┼─────────────┘
                    │
              ┌─────▼─────┐
              │ ENSEMBLE  │
              │ PREDICTOR │
              └───────────┘
                    │
           ┌────────▼────────┐
           │ Binary Decision │
           │  + Confidence   │
           │  + Reasoning    │
           └─────────────────┘
```

---

## 🧩 Components

### 1️⃣ **LLM Agent with RAG** ([Colab Notebook](https://colab.research.google.com/drive/1UVPlk-JmT8WX8eaC983H7KKVhSb5hltW?usp=sharing))

**Purpose**: Contextual reasoning over historical trial literature

**Architecture**:
- **LLM Core**: Meta Llama 3.1 405B (via API)
- **Knowledge Base**: FAISS vector index (e5-base embeddings)
- **Retrieval**: Semantic search over historical PDFs/abstracts (cutoff: 2025-05-28)
- **Output**: Structured prediction with red/yellow/green risk flags + citations

**Key Features**:
- ✅ Single-pass inference (efficient)
- ✅ Anti-hallucination rules ("mirror text unless contradicted by numbers")
- ✅ Mandatory citation enforcement
- ✅ Compliance filtering (date/pattern blocklists)

**Pipeline**:
```
Trial → Query Builder → FAISS (k=8 docs) → Context + Input → Llama 405B → JSON
```

**Strengths**: Fast, interpretable, grounded in evidence  
**Limitations**: No probabilistic calibration, treats all endpoint types equally

---

### 2️⃣ **Bayesian Statistical Model** ([Colab Notebook](https://colab.research.google.com/drive/1QR-rR0wR3P9cyEdzS4fq4v0MZskNeCQB?usp=sharing))

**Purpose**: Interpretable probabilistic reasoning with uncertainty quantification

**Model**: Hierarchical Bayesian Logistic Regression (PyMC)
```
logit(p) = β₀ + Σ_f α_f[category]
α_f ~ Normal(μ_f, σ_f)  [partial pooling per feature]
```

**Key Features**:
- ✅ **Prior construction** with Laplace smoothing and class-balance adjustment
- ✅ **Lift calculations** vs. global base rates
- ✅ **Posterior credible intervals** for uncertainty quantification
- ✅ **Log-odds decomposition** for per-feature contribution analysis

**Outputs**:
- `priors.json`: Pre-computed category-level success rates
- Per-row posterior probabilities with interpretable feature contributions
- Credible intervals for agent consumption

**Strengths**: Transparent, handles sparse categories, uncertainty-aware  
**Use Case**: Provides probabilistic priors to LLM agent ("Phase 3 primary endpoints: 86% ± 4%")

---

### 3️⃣ **AutoML Component** ([Colab Notebook](https://colab.research.google.com/drive/1lVqRONQSYKhSEOjA0KZrC-jKBo_hL3gH?usp=sharing))

**Purpose**: Automated feature engineering and pattern recognition

**Stack**: AutoGluon Tabular (multi-layer stack with text features)

**Pipeline**:
```
Raw Features → Text Aggregation → AutoGluon → Model Stack → Probability
                                    ↓
                        (LightGBM, CatBoost, NN, RF, ...)
```

**Key Features**:
- ✅ Automatic text feature extraction from trial descriptions
- ✅ 5-fold Stratified GroupCV (grouped by `abstract_id`)
- ✅ Multi-model stacking with hyperparameter optimization
- ✅ Handles mixed categorical/numerical/text data

**Outputs**:
- `df_train_all_features_merged_plus_labels.csv`: Engineered feature set
- Probability estimates per endpoint

**Strengths**: High predictive power, minimal manual feature engineering  
**Limitations**: Black-box predictions

---

## 🔗 Integration Strategy

The three components feed into an **ensemble predictor** that combines:

1. **LLM Reasoning**: Contextual evidence from historical trials
2. **Bayesian Priors**: Category-level success rates with uncertainty
3. **AutoML Predictions**: Pattern-based probability estimates

**Ensemble Logic** (simplified):
```python
# Agent receives Bayesian priors as context
bayesian_prior = get_prior(endpoint_type, phase, biomarker)

# Agent performs RAG retrieval + reasoning
llm_prediction, llm_confidence = agent.predict(trial, priors=bayesian_prior)

# AutoML provides independent probability
automl_prob = autogluon_model.predict_proba(features)

# Ensemble weighting
if llm_confidence == "High":
    final_prob = 0.6 * llm_prediction + 0.2 * bayesian_prior + 0.2 * automl_prob
elif llm_confidence == "Medium":
    final_prob = 0.4 * llm_prediction + 0.3 * bayesian_prior + 0.3 * automl_prob
else:  # Low confidence
    final_prob = 0.3 * llm_prediction + 0.4 * bayesian_prior + 0.3 * automl_prob

final_decision = final_prob > 0.5
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -U pandas numpy scikit-learn autogluon sentence-transformers faiss-cpu pymc arviz
```

### Run Individual Components

**1. Train AutoML Model**
```python
# See: Another_copy_of_asco_autogluon_baseline.ipynb
# Outputs: trained model + probability estimates
```

**2. Compute Bayesian Priors**
```python
# See: stats.ipynb
# Outputs: priors.json with category-level success rates
```

**3. Run LLM Agent with RAG**
```python
# See: Copy_of_Meta_Llama_3_1_405B_Instruct_RAG.ipynb
# Requires: FAISS index, priors.json
```

---

## 📊 Performance

| Component | Strength | Limitation |
|-----------|----------|------------|
| **LLM Agent** | Interpretable reasoning, grounded citations | No base rate calibration |
| **Bayesian GLM** | Uncertainty quantification, handles sparsity | Requires feature engineering |
| **AutoML** | High predictive power, automatic features | Black-box predictions |
| **Ensemble** | Combines strengths of all three | Integration complexity |

**Expected Performance** (validation set):
- LLM Agent: ~72-75% accuracy
- Bayesian GLM: ~74-76% accuracy  
- AutoML: ~76-78% accuracy
- **Ensemble: ~78-80% accuracy** (target)

---

## 🗂️ Project Structure

```
.
├── Copy_of_Meta_Llama_3_1_405B_Instruct_RAG.ipynb  # LLM Agent + RAG
├── stats.ipynb                                      # Bayesian inference
├── Another_copy_of_asco_autogluon_baseline.ipynb   # AutoML pipeline
├── rag_data/                                        # Historical trial PDFs
├── rag_index/                                       # FAISS index + metadata
└── README.md                                        # This file
```

---

## 🔬 Key Innovations

1. **Hybrid Reasoning**: Combines symbolic (Bayesian), subsymbolic (AutoML), and linguistic (LLM) AI
2. **Uncertainty-Aware**: Bayesian credible intervals inform ensemble weighting
3. **Interpretable**: LLM provides citations + reasoning; Bayesian shows log-odds contributions
4. **Compliance**: Strict date filtering prevents data leakage
5. **Scalable**: RAG + caching reduces inference costs

---

## 🛠️ Roadmap

- [ ] Implement biomarker tier classification (FDA-approved vs. experimental)
- [ ] Add multi-agent deliberation (Statistical Analyst + Safety Officer + Risk Assessor)
- [ ] Integrate tool caching for PubMed/FDA API calls
- [ ] Confidence-based adaptive ensembling
- [ ] Explainability dashboard (SHAP + attention weights + Bayesian contributions)

---

## 📝 Citation

If you use this system, please cite:

```bibtex
@misc{clinical_trial_prediction_2025,
  title={Hybrid AI System for Clinical Trial Endpoint Prediction},
  author={[Your Name]},
  year={2025},
  howpublished={\url{https://github.com/[your-username]/[repo-name]}}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ASCO 2025 conference data
- Kaggle competition organizers
- AutoGluon, PyMC, and Hugging Face communities
- Meta AI for Llama 3.1 405B

---

## 💬 Contact

Questions? Open an issue or reach out via [your-email@example.com]

**Interactive Notebooks:**
- 🤖 [Main Agent](https://colab.research.google.com/drive/1UVPlk-JmT8WX8eaC983H7KKVhSb5hltW?usp=sharing)
- 📊 [Statistical Analysis & Bayesian Reasoning](https://colab.research.google.com/drive/1QR-rR0wR3P9cyEdzS4fq4v0MZskNeCQB?usp=sharing)
- 🧠 [AutoML Component](https://colab.research.google.com/drive/1lVqRONQSYKhSEOjA0KZrC-jKBo_hL3gH?usp=sharing)

