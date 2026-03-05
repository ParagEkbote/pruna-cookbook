"""
Dataset Quality Audit Tool

Analyzes Hugging Face datasets for:
- Token length statistics
- Semantic variance and rarity
- Duplicate detection (exact, near-duplicate)
- Topic clustering
- Answer determinism
- Contradiction detection
- Training regime recommendations
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

sns.set(style="whitegrid")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_hf_dataset(name):
    """Load dataset from Hugging Face and extract train/first split."""
    ds = load_dataset(name)
    
    if "train" in ds:
        ds = ds["train"]
    else:
        ds = ds[list(ds.keys())[0]]
    
    return ds


def dataset_to_df(ds):
    """Convert dataset to pandas DataFrame with validation."""
    df = pd.DataFrame(ds)
    
    if "question" not in df.columns:
        raise ValueError("Dataset must contain 'question' column")
    if "answer" not in df.columns:
        raise ValueError("Dataset must contain 'answer' column")
    
    return df


# ============================================================================
# TOKEN LENGTH ANALYSIS
# ============================================================================

def compute_token_lengths(df):
    """Compute question and answer token lengths using BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    df["q_len"] = df["question"].apply(lambda x: len(tokenizer.tokenize(x)))
    df["a_len"] = df["answer"].apply(lambda x: len(tokenizer.tokenize(x)))
    
    return df


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================

def detect_exact_duplicates(df):
    """Detect exact duplicates in question-answer pairs."""
    exact_dup = df.duplicated(subset=["question", "answer"]).sum()
    return exact_dup


def detect_question_duplicates(df):
    """Detect duplicate questions (multiple answers for same question)."""
    question_dup = df.duplicated(subset=["question"]).sum()
    return question_dup


def detect_near_duplicates(df, sample_size=5000):
    """Detect near-duplicate questions using semantic similarity (>0.95)."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    embeddings = model.encode(sample["question"].tolist(), show_progress_bar=True)
    
    sim_matrix = cosine_similarity(embeddings)
    near_dup = np.sum(sim_matrix > 0.95) - len(sim_matrix)
    near_dup_rate = near_dup / len(sample)
    
    return near_dup_rate


# ============================================================================
# SEMANTIC ANALYSIS
# ============================================================================

def compute_token_rarity(df):
    """Compute rarity score for answers based on token frequency."""
    tokens = []
    
    for ans in df["answer"]:
        tokens.extend(ans.lower().split())
    
    freq = Counter(tokens)
    rarity_scores = []
    
    for ans in df["answer"]:
        score = np.mean([1 / (freq[t] + 1) for t in ans.lower().split()])
        rarity_scores.append(score)
    
    df["rarity_score"] = rarity_scores
    return df


def compute_semantic_variance(df):
    """Compute semantic variance of answers relative to all answers."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df["answer"].tolist(), show_progress_bar=True)
    
    sim = cosine_similarity(embeddings)
    variance_scores = []
    
    for i in range(len(df)):
        variance = np.var(sim[i])
        variance_scores.append(variance)
    
    df["semantic_variance"] = variance_scores
    return df


# ============================================================================
# DIFFICULTY ESTIMATION
# ============================================================================

def compute_recall_difficulty(df):
    """Combine rarity, semantic variance, and answer length into difficulty score."""
    df["recall_difficulty"] = (
        df["rarity_score"] +
        df["semantic_variance"] +
        (df["a_len"] / df["a_len"].max())
    )
    return df


def compute_answer_determinism(df):
    """Compute determinism score: 1 / (number of unique answers per question)."""
    determinism = []
    
    for q, group in df.groupby("question"):
        unique_answers = group["answer"].nunique()
        determinism_score = 1 / unique_answers
        determinism.extend([determinism_score] * len(group))
    
    df["determinism_score"] = determinism
    return df


# ============================================================================
# CONTRADICTION DETECTION
# ============================================================================

def detect_contradictions(df):
    """Detect contradictory answers to semantically similar questions."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode(df["question"].tolist())
    
    sim = cosine_similarity(q_emb)
    contradictions = []
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sim[i][j] > 0.92:
                if df["answer"].iloc[i] != df["answer"].iloc[j]:
                    contradictions.append((i, j))
    
    return contradictions


# ============================================================================
# TOPIC CLUSTERING
# ============================================================================

def compute_topic_clusters(df, n_clusters=10):
    """Cluster questions into topics using K-means on embeddings."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(df["question"].tolist(), show_progress_bar=True)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    df["cluster"] = labels
    cluster_counts = df["cluster"].value_counts()
    
    return df, cluster_counts


# ============================================================================
# PREFERENCE PAIR MINING
# ============================================================================

def mine_preference_pairs(df):
    """Mine preference pairs (chosen vs rejected) for preference training."""
    pairs = []
    
    for q, group in df.groupby("question"):
        if len(group) < 2:
            continue
        
        sorted_group = group.sort_values("recall_difficulty")
        best = sorted_group.iloc[0]
        worst = sorted_group.iloc[-1]
        
        pairs.append({
            "prompt": q,
            "chosen": best["answer"],
            "rejected": worst["answer"]
        })
    
    return pd.DataFrame(pairs)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_token_distributions(df):
    """Plot question and answer token length distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(df["q_len"], bins=50, color="skyblue", edgecolor="black")
    axes[0].set_title("Question Length Distribution")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(df["a_len"], bins=50, color="lightcoral", edgecolor="black")
    axes[1].set_title("Answer Length Distribution")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")
    
    difficulty = df["q_len"] + df["a_len"]
    axes[2].hist(difficulty, bins=50, color="lightgreen", edgecolor="black")
    axes[2].set_title("Total Length Distribution")
    axes[2].set_xlabel("Tokens (Q+A)")
    axes[2].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("token_distributions.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_difficulty_metrics(df):
    """Plot recall difficulty and determinism distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(df["recall_difficulty"], bins=50, color="orange", edgecolor="black")
    axes[0].set_title("Recall Difficulty Distribution")
    axes[0].set_xlabel("Difficulty Score")
    axes[0].set_ylabel("Samples")
    
    axes[1].hist(df["determinism_score"], bins=20, color="purple", edgecolor="black")
    axes[1].set_title("Answer Determinism Distribution")
    axes[1].set_xlabel("Determinism Score")
    axes[1].set_ylabel("Samples")
    
    plt.tight_layout()
    plt.savefig("difficulty_metrics.png", dpi=100, bbox_inches="tight")
    plt.close()


def plot_cluster_distribution(cluster_counts):
    """Plot topic cluster distribution."""
    plt.figure(figsize=(12, 5))
    cluster_counts.sort_index().plot(kind="bar", color="teal", edgecolor="black")
    plt.title("Topic Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cluster_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================================
# TRAINING RECOMMENDATION
# ============================================================================

def recommend_training_regime(df, contradictions, exact_dup, question_dup, near_dup_rate):
    """Generate training regime recommendations based on dataset quality metrics."""
    avg_difficulty = df["recall_difficulty"].mean()
    avg_determinism = df["determinism_score"].mean()
    avg_answer_len = df["a_len"].mean()
    duplicate_ratio = question_dup / len(df)
    
    recommendations = []
    
    # Determinism-based recommendation
    if avg_determinism > 0.8:
        recommendations.append("✓ SFTTrainer (high answer determinism, standard supervised fine-tuning)")
    elif avg_determinism > 0.5:
        recommendations.append("✓ Weighted SFT (moderate determinism, use importance weighting)")
    else:
        recommendations.append("✓ Preference Training (DPO/ORPO - low determinism, multiple valid answers)")
    
    # Answer length feedback
    if avg_answer_len > 100:
        recommendations.append("• Consider weighted loss for long-form answers (>100 tokens)")
    elif avg_answer_len < 10:
        recommendations.append("⚠ Warning: Answers too short (<10 tokens) may harm recall performance")
    
    # Duplicate feedback
    if duplicate_ratio > 0.3:
        recommendations.append("• High question duplication - dataset may support preference training")
    if near_dup_rate > 0.2:
        recommendations.append("⚠ High near-duplicate rate (>20%) - consider deduplication")
    
    # Contradiction feedback
    if len(contradictions) > 0:
        contradiction_ratio = len(contradictions) / len(df)
        recommendations.append(f"⚠ Found {len(contradictions)} contradictory pairs ({contradiction_ratio:.1%})")
    
    return {
        "avg_recall_difficulty": round(avg_difficulty, 4),
        "avg_answer_determinism": round(avg_determinism, 4),
        "avg_answer_length": round(avg_answer_len, 2),
        "question_duplication_ratio": round(duplicate_ratio, 4),
        "near_duplicate_rate": round(near_dup_rate, 4),
        "exact_duplicates": exact_dup,
        "question_duplicates": question_dup,
        "contradiction_pairs": len(contradictions),
        "recommendations": recommendations
    }


# ============================================================================
# MAIN AUDIT FUNCTION
# ============================================================================

def run_audit(dataset_name):
    """Run complete dataset audit and generate report."""
    print("\n" + "=" * 70)
    print("DATASET QUALITY AUDIT")
    print("=" * 70)
    
    # Load and prepare data
    print(f"\n[1/8] Loading dataset: {dataset_name}")
    ds = load_hf_dataset(dataset_name)
    df = dataset_to_df(ds)
    print(f"      Dataset size: {len(df)} samples")
    
    # Token analysis
    print("[2/8] Computing token lengths...")
    df = compute_token_lengths(df)
    
    # Duplicate detection
    print("[3/8] Detecting duplicates...")
    exact_dup = detect_exact_duplicates(df)
    question_dup = detect_question_duplicates(df)
    near_dup_rate = detect_near_duplicates(df)
    
    # Semantic analysis
    print("[4/8] Computing token rarity...")
    df = compute_token_rarity(df)
    
    print("[5/8] Computing semantic variance...")
    df = compute_semantic_variance(df)
    
    # Difficulty metrics
    print("[6/8] Computing recall difficulty...")
    df = compute_recall_difficulty(df)
    
    print("[7/8] Computing answer determinism...")
    df = compute_answer_determinism(df)
    
    # Contradiction detection
    print("[8/8] Detecting contradictions...")
    contradictions = detect_contradictions(df)
    
    # Topic clustering (parallel to above)
    print("     Computing topic clusters...")
    df, cluster_counts = compute_topic_clusters(df)
    
    # Generate outputs
    print("\n" + "-" * 70)
    print("GENERATING OUTPUTS")
    print("-" * 70)
    
    print("Creating visualizations...")
    plot_token_distributions(df)
    plot_difficulty_metrics(df)
    plot_cluster_distribution(cluster_counts)
    
    print("Mining preference pairs...")
    pref_pairs = mine_preference_pairs(df)
    
    # Generate recommendation
    result = recommend_training_regime(df, contradictions, exact_dup, question_dup, near_dup_rate)
    
    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset Size:                  {len(df)}")
    print(f"Average Question Length:       {df['q_len'].mean():.1f} tokens")
    print(f"Average Answer Length:         {df['a_len'].mean():.1f} tokens")
    print(f"\nExact Duplicates:              {result['exact_duplicates']}")
    print(f"Question Duplicates:           {result['question_duplicates']}")
    print(f"Near-Duplicate Rate:           {result['near_duplicate_rate']:.2%}")
    print(f"Contradiction Pairs:           {result['contradiction_pairs']}")
    
    print(f"\nAvg Recall Difficulty:         {result['avg_recall_difficulty']:.4f}")
    print(f"Avg Answer Determinism:        {result['avg_answer_determinism']:.4f}")
    print(f"Unique Questions:              {df['question'].nunique()}")
    print(f"Topic Clusters:                {df['cluster'].nunique()}")
    print(f"Preference Pairs Mined:        {len(pref_pairs)}")
    
    print("\n" + "-" * 70)
    print("TRAINING RECOMMENDATIONS")
    print("-" * 70)
    for rec in result["recommendations"]:
        print(rec)
    
    print("\n" + "=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)
    
    # Save CSV outputs
    df.to_csv("dataset_audit_scores.csv", index=False)
    print("✓ Saved: dataset_audit_scores.csv")
    
    if len(pref_pairs) > 0:
        pref_pairs.to_csv("preference_pairs.csv", index=False)
        print("✓ Saved: preference_pairs.csv")
    
    # Save audit report
    with open("audit_report.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET AUDIT REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset Size: {len(df)}\n\n")
        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Question Length: {df['q_len'].mean():.1f} tokens\n")
        f.write(f"Average Answer Length: {df['a_len'].mean():.1f} tokens\n")
        f.write(f"Exact Duplicates: {result['exact_duplicates']}\n")
        f.write(f"Question Duplicates: {result['question_duplicates']}\n")
        f.write(f"Near-Duplicate Rate: {result['near_duplicate_rate']:.2%}\n")
        f.write(f"Contradiction Pairs: {result['contradiction_pairs']}\n")
        f.write(f"Unique Questions: {df['question'].nunique()}\n")
        f.write(f"Topic Clusters: {df['cluster'].nunique()}\n\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        for rec in result["recommendations"]:
            f.write(rec + "\n")
    
    print("✓ Saved: audit_report.txt")
    print("\nVisualizations:")
    print("✓ token_distributions.png")
    print("✓ difficulty_metrics.png")
    print("✓ cluster_distribution.png")
    
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit Hugging Face dataset quality and generate training recommendations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of Hugging Face dataset to audit (e.g., 'squad', 'wmt14')"
    )
    
    args = parser.parse_args()
    run_audit(args.dataset)