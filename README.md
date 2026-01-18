Here is the complete, compact content for your `README.md` file.

```markdown
# Entity Linking for KBQA: Dense vs. Lexical Retrieval Replication

This repository contains the implementation and experimental evaluation for the technical report: **"Systematic Literature Review: Machine Learning Approaches for Entity Linking in Knowledge Base Question Answering."**

## Overview
This project performs a replication study of the **Dense Retrieval** paradigm introduced in *Wu et al. (2020)*. The experiment quantitatively compares two candidate generation architectures for Entity Linking (EL) in Question Answering:

1. **Dense Retrieval (Bi-Encoder):** Uses a pre-trained Transformer (`all-MiniLM-L6-v2`) to embed questions and entities into a shared vector space.
2. **Lexical Retrieval (BM25):** Uses the Okapi BM25 algorithm to rank entities based on sparse keyword overlap.

The study validates the "Lexical Gap" hypothesis, demonstrating that Dense Retrieval significantly outperforms Lexical Retrieval on short, ambiguous user queries.

## Repository Structure
- `main.py`: Main experiment script (Data ingestion, Model training, Evaluation).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container configuration.
- `webqsp_experiment_results.csv`: Generated raw results.
- `webqsp_accuracy_chart.png`: Generated visualization.
- `README.md`: Documentation.

## Setup and Installation

### Option A: Local Installation (Python)
1. Clone the repository.
2. Create a file named `requirements.txt` with the following content:
   ```text
   pandas
   numpy
   torch
   matplotlib
   seaborn
   rank_bm25
   datasets
   wikipedia
   sentence-transformers
   scikit-learn

```

3. Install dependencies:
```bash
pip install -r requirements.txt

```


4. Run the experiment:
```bash
python main.py

```




The `main.py` script performs the following steps:

1. **Dynamic Data Ingestion:** Loads real user questions from the **WebQuestions** benchmark (via Hugging Face Datasets).
2. 
**KB Construction:** Dynamically builds a "Micro-Knowledge Base" by fetching real entity summaries from Wikipedia  for the ground truth answers.


3. **Distractor Generation:** Injects "hard negatives" (ambiguous entities, e.g., *Python (missile)* vs. *Python (programming)*) to rigorously test disambiguation.
4. **Evaluation:** Runs both the Bi-Encoder and BM25 on the query set and calculates Top-1 Accuracy.
5. **Artifact Generation:** Saves the results to CSV and plots a comparison chart.

## Results

The experiment confirms that Dense Retrieval is superior for handling the context sparsity of QA queries.

| Method | Top-1 Accuracy | Observation |
| --- | --- | --- |
| Lexical (BM25) | 40.7% | Failed on queries lacking exact keyword overlap. |
| Dense (Bi-Encoder) | 56.6% | Successfully linked entities based on semantic meaning. |

*(Results based on N=113 samples from WebQuestions)*

## References

[1] Wu, L., Petroni, F., Josifoski, M., Riedel, S., & Zettlemoyer, L. (2020). Scalable Zero-shot Entity Linking with Dense Entity Retrieval. EMNLP 2020. 
[2] Berant, J., et al. (2013). Semantic Parsing on Freebase from Question-Answer Pairs. EMNLP 2013.

```

```
