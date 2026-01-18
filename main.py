import pandas as pd
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import string
import wikipedia
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_SAMPLES = 150  # Keep low (10-20) to avoid Wikipedia API rate limits during testing
OUTPUT_CSV = 'webqsp_experiment_results.csv'
OUTPUT_CHART = 'webqsp_accuracy_chart.png'


# ---------------------------------------------------------
# 1. REAL DATA INGESTION (WebQuestions + Wikipedia)
# ---------------------------------------------------------
def fetch_real_data():
    """
    1. Loads the official WebQuestions dataset (precursor to WebQSP).
    2. Fetches real Wikipedia summaries for the answers (Gold Entities).
    3. Fetches 'Distractor' entities to simulate a noisy Knowledge Base.
    """
    logger.info("Loading 'stanfordnlp/web_questions' from Hugging Face...")
    try:
        # Load first N examples
        dataset = load_dataset("stanfordnlp/web_questions", split=f"train[:{NUM_SAMPLES}]")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}. Check internet connection.")
        return [], []

    kb_data = []
    test_queries = []

    logger.info("Building Dynamic Knowledge Base via Wikipedia API (this may take 30s)...")

    # 1. Add Gold Entities (Correct Answers)
    for row in dataset:
        question = row['question']
        answer_list = row['answers']  # List of strings

        if not answer_list: continue

        # Take the first answer as the primary target
        target_entity = answer_list[0]

        # Clean string for Wiki search
        search_query = target_entity.replace('"', '').strip()

        try:
            # Fetch real summary
            # We use the search result to find the most likely page
            # Note: In a full prod system, we would map to Freebase/Wikidata IDs.
            # Here we use string matching for the assignment simulation.
            summary = wikipedia.summary(search_query, sentences=2)

            ent_id = f"E-{len(kb_data)}"
            kb_data.append({"id": ent_id, "title": target_entity, "desc": summary, "type": "Gold"})

            test_queries.append({
                "query": question,
                "target_id": ent_id,
                "target_name": target_entity,
                "type": "Real_WebQSP"
            })

        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            # Skip if Wiki is confusing (common in automated scripts)
            continue
        except Exception as e:
            logger.warning(f"Wiki Error on {target_entity}: {e}")
            continue

    # 2. Add Hard Distractors (Ambiguous Terms)
    # These ensures BM25 gets confused by words like "Java", "Python", "Amazon"
    distractors = ["Java (programming language)", "Java (island)", "Amazon (company)",
                   "Amazon River", "Python (missile)", "Monty Python", "Washington D.C.",
                   "George Washington", "Jaguar (band)", "Apple Corps"]

    for term in distractors:
        try:
            summary = wikipedia.summary(term, sentences=2)
            kb_data.append({"id": f"D-{len(kb_data)}", "title": term, "desc": summary, "type": "Distractor"})
        except:
            pass

    logger.info(f"Final KB Size: {len(kb_data)} entities (Real + Distractors)")
    logger.info(f"Test Set Size: {len(test_queries)} questions")

    return kb_data, test_queries


# ---------------------------------------------------------
# 2. MODELS (Bi-Encoder vs BM25)
# ---------------------------------------------------------
class DenseEntityLinker:
    def __init__(self, kb_data):
        self.model = SentenceTransformer(MODEL_NAME)
        self.kb_df = pd.DataFrame(kb_data)
        # Encode KB once
        self.embeddings = self.model.encode(
            [f"{r['title']} {r['desc']}" for _, r in self.kb_df.iterrows()],
            convert_to_tensor=True
        )

    def predict(self, query):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]
        best_idx = torch.argmax(scores).item()
        return self.kb_df.iloc[best_idx]['id']


class LexicalEntityLinker:
    def __init__(self, kb_data):
        self.kb_df = pd.DataFrame(kb_data)
        # Tokenize KB
        corpus = [self._tokenize(f"{r['title']} {r['desc']}") for _, r in self.kb_df.iterrows()]
        self.bm25 = BM25Okapi(corpus)

    def _tokenize(self, text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

    def predict(self, query):
        scores = self.bm25.get_scores(self._tokenize(query))
        best_idx = np.argmax(scores)
        return self.kb_df.iloc[best_idx]['id']


# ---------------------------------------------------------
# 3. RUN EXPERIMENT
# ---------------------------------------------------------
def run():
    # 1. Load Data
    kb_data, queries = fetch_real_data()
    if not queries:
        logger.error("No data loaded. Exiting.")
        return

    # 2. Initialize Models
    dense_model = DenseEntityLinker(kb_data)
    lexical_model = LexicalEntityLinker(kb_data)

    results = []

    # 3. Loop
    for q in queries:
        # Dense Prediction
        dense_pred = dense_model.predict(q['query'])
        # Lexical Prediction
        lex_pred = lexical_model.predict(q['query'])

        results.append({
            "Question": q['query'],
            "Target": q['target_name'],
            "Dense_Correct": dense_pred == q['target_id'],
            "Lexical_Correct": lex_pred == q['target_id']
        })

    # 4. Results & Plotting
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    acc_dense = df['Dense_Correct'].mean() * 100
    acc_lex = df['Lexical_Correct'].mean() * 100

    print("\n" + "=" * 40)
    print(f"FINAL ACCURACY (N={len(df)})")
    print("=" * 40)
    print(f"Dense Retrieval (Bi-Encoder): {acc_dense:.1f}%")
    print(f"Lexical Retrieval (BM25):     {acc_lex:.1f}%")
    print("=" * 40)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['BM25 (Lexical)', 'Bi-Encoder (Dense)'], y=[acc_lex, acc_dense], palette='viridis')
    plt.title(f"Accuracy on WebQuestions (Real Data)", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.savefig(OUTPUT_CHART)
    logger.info(f"Chart saved to {OUTPUT_CHART}")


if __name__ == "__main__":
    run()