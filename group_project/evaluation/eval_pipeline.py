"""
RAG Evaluation Pipeline using RAGAS.
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from litellm import completion

from src.task9_retrieval_pipeline import retrieve
from src.task5_semantic_search import semantic_search
from src.task10_generation import format_context

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_ragas_llm():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.0-flash-lite-preview-02-05:free",
    )
    return LangchainLLMWrapper(llm)

def build_ragas_embeddings():
    embed = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    return LangchainEmbeddingsWrapper(embed)

def evaluate_with_ragas(config_name: str, config_type: str, golden_dataset: list[dict], ragas_llm, ragas_embeddings) -> pd.DataFrame:
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    
    # Slice to 5 to avoid rate limits
    subset = golden_dataset[:5]
    
    for i, item in enumerate(subset):
        print(f"[{config_name}] Q{i+1}/{len(subset)}: {item['question']}")
        
        if config_type == "hybrid":
            sources = retrieve(item["question"], top_k=3, use_reranking=True)
        else:
            sources = semantic_search(item["question"], top_k=3)
            
        context_str = format_context(sources)
        prompt = f"Context:\n{context_str}\n\nQuestion:\n{item['question']}\n\nAnswer the question concisely based on the context."
        
        response = completion(
            model="openrouter/google/gemini-2.0-flash-lite-preview-02-05:free",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append([c.get("content", "") for c in sources])
        eval_data["ground_truth"].append(item["expected_answer"])
        
    dataset = Dataset.from_dict(eval_data)
    
    print(f"Running RAGAS metrics for {config_name}...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )
    
    return result.to_pandas()

def compare_configs(golden_dataset: list[dict]):
    ragas_llm = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()
    
    print("Evaluating Config A (Hybrid + Reranking)...")
    df_a = evaluate_with_ragas("Config A", "hybrid", golden_dataset, ragas_llm, ragas_embeddings)
    
    print("\nEvaluating Config B (Dense Only)...")
    df_b = evaluate_with_ragas("Config B", "dense", golden_dataset, ragas_llm, ragas_embeddings)
    
    return {"Config A (Hybrid)": df_a, "Config B (Dense)": df_b}

def export_results(results: dict):
    content = "# RAG Evaluation Results\n\n"
    content += "## A/B Comparison\n\n"
    
    for config_name, df in results.items():
        content += f"### {config_name}\n"
        avg_scores = df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean()
        
        content += "| Metric | Score |\n|--------|-------|\n"
        for metric, score in avg_scores.items():
            content += f"| {metric} | {score:.4f} |\n"
        content += "\n"
        
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")

if __name__ == "__main__":
    dataset = load_golden_dataset()
    print(f"Loaded {len(dataset)} test cases from golden_dataset.json")
    results = compare_configs(dataset)
    export_results(results)
