# Adaptive ETA Prediction for Inland Waterway Cargo

**Riyan Wankhede | 23BCE9287 | VIT-AP University**

An end-to-end machine learning pipeline for predicting and adaptively adjusting Estimated Time of Arrival (ETA) for inland waterway cargo vessels, combining deep learning, NLP, and reinforcement learning.

---

## Overview

Traditional ETA systems rely only on structured numerical data and produce static predictions. This system integrates three components to handle dynamic operational conditions reported through unstructured text (captain logs, port authority messages, maintenance reports).

---



## Architecture



### 1. Base ETA Prediction (Deep Learning)

- PyTorch MLP: `Input(5) → 128 → 64 → 32 → Output(1)`
- Batch normalisation + dropout at each hidden layer
- Trained on 5 structured features: distance, vessel speed, river current, weather severity, port congestion
- **Test MAE: 137.89 min | R²: 0.9773**



### 2. Delay Signal Extraction (NLP)

- Three-tier pipeline: phrase shortcuts → 35-keyword weighted average → DistilBERT fallback
- Outputs a delay severity score between 0 and 1
- Example keywords: `storm → 1.0`, `congestion → 0.9`, `fog → 0.8`, `smooth → 0.1`



### 3. Adaptive ETA Adjustment (Reinforcement Learning)

- Q-Learning agent with 210 states (10 ETA bins × 21 severity bins)
- 5 discrete actions: `[0%, 2%, 5%, 10%, 20%]` ETA increase
- Reward: `r = (|err_base| − |err_adj|) / base_eta`
- Trained over 5,000 episodes with ε-greedy exploration (ε: 1.0 → 0.05)
- **Adaptive MAE: 135.61 min | R²: 0.9801 | MAE Δ: -1.65%**

---



## Results


| Metric    | Base DNN | Adaptive (RL) |
| --------- | -------- | ------------- |
| MAE (min) | 137.89   | 135.61        |
| R²        | 0.9773   | 0.9801        |
| MAE Δ (%) | -        | -1.65%        |


---



## Project Structure

```
├── Adaptive_ETA_Prediction.ipynb   # Main notebook (end-to-end pipeline)
├── Research_Paper.pdf              # IEEE-format research paper
├── requirements.txt                # Python dependencies
├── ETA Prediction Architecture     # System Overview 
└── README.md
```

---



## Setup

```bash
pip install -r requirements.txt
```

Then open and run `Adaptive_ETA_Prediction.ipynb` top to bottom.

**Requirements:** Python 3.8+, PyTorch, Transformers (HuggingFace), scikit-learn, pandas, numpy, matplotlib

---



## Key Concepts

- **Synthetic dataset** of 1,000 inland waterway voyages
- **Latent severity** variable in dataset used only for RL reward evaluation (not fed to DNN)
- **NLP + RL integration**: NLP severity score forms part of the RL state representation
- Q-table persisted across episodes for stable convergence

---



## Citation

If you use this work, please cite:

> Wankhede, R. (2026). *Adaptive ETA Prediction for Inland Waterway Cargo Using Machine Learning, NLP, and Reinforcement Learning*. VIT-AP University.

---



## License

MIT