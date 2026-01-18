# Entity Linking for KBQA: Dense vs. Lexical Retrieval Replication

This repository contains the implementation and experimental evaluation for the technical report: **"Systematic Literature Review: Machine Learning Approaches for Entity Linking in Knowledge Base Question Answering."**

## Overview
This mini-project performs a replication study of the **Dense Retrieval** paradigm introduced in *Wu et al. (2020) "Scalable Zero-Shot Entity Linking with Dense Entity Retrieval"*[cite: 1, 9, 22].

The experiment quantitatively compares two candidate generation architectures for Entity Linking (EL) in Question Answering:
1.  [cite_start]**Dense Retrieval (Bi-Encoder):** Uses a pre-trained Transformer (`all-MiniLM-L6-v2`) to embed questions and entities into a shared vector space[cite: 9, 93].
2.  [cite_start]**Lexical Retrieval (BM25):** Uses the Okapi BM25 algorithm to rank entities based on sparse keyword overlap[cite: 68, 186].

The study validates the "Lexical Gap" hypothesis, demonstrating that Dense Retrieval significantly outperforms Lexical Retrieval on short, ambiguous user queries.

##  Repository Structure
```text
.
├── main.py                     # Main experiment script (Data ingestion, Model training, Evaluation)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── webqsp_experiment_results.csv # Generated raw results (CSV)
├── webqsp_accuracy_chart.png     # Generated visualization (PNG)
└── README.md                   # Documentation
