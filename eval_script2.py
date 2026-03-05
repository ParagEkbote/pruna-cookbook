"""
knowledge_collapse_risk.py

Compute Knowledge Collapse Risk for QA datasets.

Usage:
python knowledge_collapse_risk.py --dataset AINovice2005/EldenRing_Small
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from datasets import load_dataset
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# Load dataset
# ------------------------------

def load_hf_dataset(name):

    dataset = load_dataset(name)

    if "train" in dataset:
        ds = dataset["train"]
    else:
        ds = dataset[list(dataset.keys())[0]]

    df = pd.DataFrame(ds)

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain 'question' and 'answer' columns")

    return df


# ------------------------------
# Entity extraction
# ------------------------------

def extract_entities(df):

    nlp = spacy.load("en_core_web_lg")

    entities = []

    for ans in df["answer"]:
        doc = nlp(ans)

        for ent in doc.ents:
            entities.append(ent.text.lower())

    return Counter(entities)


# ------------------------------
# Entropy calculation
# ------------------------------

def compute_entropy(counter):

    counts = np.array(list(counter.values()))

    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return entropy


# ------------------------------
# Frequency skew
# ------------------------------

def compute_frequency_skew(counter):

    counts = np.array(list(counter.values()))

    top10 = np.sort(counts)[-10:]

    skew = np.sum(top10) / np.sum(counts)

    return skew


# ------------------------------
# Template similarity
# ------------------------------

def compute_template_similarity(df, sample_size=5000):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sample = df.sample(min(sample_size, len(df)))

    answers = sample["answer"].tolist()

    embeddings = model.encode(answers, show_progress_bar=True)

    sim = cosine_similarity(embeddings)

    upper = sim[np.triu_indices_from(sim, k=1)]

    similarity_score = np.mean(upper > 0.9)

    return similarity_score


# ------------------------------
# Visualizations
# ------------------------------

def plot_entity_distribution(counter):

    top = counter.most_common(20)

    labels = [x[0] for x in top]
    values = [x[1] for x in top]

    plt.figure()

    plt.bar(labels, values)

    plt.xticks(rotation=45)

    plt.title("Top Entity Frequency")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("entity_frequency.png")


def plot_entity_histogram(counter):

    counts = list(counter.values())

    plt.figure()

    plt.hist(counts, bins=50)

    plt.title("Entity Frequency Distribution")

    plt.xlabel("Frequency")

    plt.ylabel("Entities")

    plt.savefig("entity_distribution.png")


# ------------------------------
# Knowledge Collapse Risk Score
# ------------------------------

def compute_collapse_risk(entropy, skew, template_similarity):

    risk = template_similarity + skew - (entropy / 10)

    return risk


# ------------------------------
# Main
# ------------------------------

def run_analysis(dataset_name):

    print("Loading dataset:", dataset_name)

    df = load_hf_dataset(dataset_name)

    print("Dataset size:", len(df))

    print("\nExtracting entities...")

    entity_counter = extract_entities(df)

    unique_entities = len(entity_counter)

    entropy = compute_entropy(entity_counter)

    skew = compute_frequency_skew(entity_counter)

    print("\nComputing answer template similarity...")

    template_similarity = compute_template_similarity(df)

    collapse_risk = compute_collapse_risk(entropy, skew, template_similarity)

    plot_entity_distribution(entity_counter)

    plot_entity_histogram(entity_counter)

    print("\n==============================")
    print("KNOWLEDGE COLLAPSE ANALYSIS")
    print("==============================")

    print("Unique entities:", unique_entities)

    print("Entity entropy:", round(entropy, 3))

    print("Entity frequency skew:", round(skew, 3))

    print("Template similarity:", round(template_similarity, 3))

    print("\nCollapse Risk Score:", round(collapse_risk, 3))

    if collapse_risk > 0.8:
        print("⚠ High collapse risk — dataset dominated by templates")

    elif collapse_risk > 0.4:
        print("⚠ Moderate collapse risk — consider adding entity diversity")

    else:
        print("✓ Low collapse risk — dataset suitable for SFT training")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    run_analysis(args.dataset)