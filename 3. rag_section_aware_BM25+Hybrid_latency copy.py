import re
import json
import pickle
import faiss
import os
import fitz
import statistics
from datetime import datetime
import time
import numpy as np
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
# 
# ----------------------------------
# 1. Configuration
# ----------------------------------

PDF_FOLDER = "data/pdfs"
INDEX_FOLDER = "index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 5

os.makedirs(INDEX_FOLDER, exist_ok=True)

model = SentenceTransformer(EMBEDDING_MODEL)


import time
import numpy as np
from collections import defaultdict


class RetrievalProfiler:

    def __init__(self):
        self.stage_times = defaultdict(list)
        self.total_times = []

    def reset(self):
        self.stage_times.clear()
        self.total_times.clear()

    def start(self):
        return time.perf_counter()

    def record(self, stage, start_time):
        elapsed = time.perf_counter() - start_time
        self.stage_times[stage].append(elapsed)

    def record_total(self, start_time):
        elapsed = time.perf_counter() - start_time
        self.total_times.append(elapsed)

    def summary(self):

        if not self.total_times:
            print("No profiling data available.")
            return

        avg_total = np.mean(self.total_times)

        print("\n" + "=" * 75)
        print("RETRIEVAL PERFORMANCE SUMMARY")
        print("=" * 75)

        print(
            f"{'Component':<25}"
            f"{'Avg (ms)':>12}"
            f"{'P95 (ms)':>12}"
            f"{'Max (ms)':>12}"
            f"{'% Total':>12}"
        )

        print("-" * 75)

        for stage, timings in self.stage_times.items():

            avg_time = np.mean(timings)
            p95_time = np.percentile(timings, 95)
            max_time = np.max(timings)

            percentage = (
                (avg_time / avg_total) * 100
                if avg_total > 0
                else 0
            )

            print(
                f"{stage:<25}"
                f"{avg_time * 1000:>12.2f}"
                f"{p95_time * 1000:>12.2f}"
                f"{max_time * 1000:>12.2f}"
                f"{percentage:>11.1f}%"
            )

        print("-" * 75)

        print(f"Total Questions       : {len(self.total_times)}")
        print(f"Average Total Latency : {avg_total * 1000:.2f} ms")
        print(f"P95 Total Latency     : {np.percentile(self.total_times, 95) * 1000:.2f} ms")
        print(f"Maximum Total Latency : {np.max(self.total_times) * 1000:.2f} ms")

        slowest_stage = max(
            self.stage_times,
            key=lambda stage: np.mean(self.stage_times[stage])
        )

        print(f"Slowest Component     : {slowest_stage}")

        print("=" * 75)
profiler = RetrievalProfiler()
# ----------------------------------
# 2. Extract PDF Text
# ----------------------------------

def extract_pdf_text(pdf_path):

    pages = []

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "text": text,
                "source": os.path.basename(pdf_path),
                "page": page_number + 1
            })

    return pages


# ----------------------------------
# 3. Create Chunks
# ----------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def is_heading(line: str) -> bool:
    """
    Detect likely section headings and numbered clauses.
    """

    line = line.strip()

    if not line:
        return False

    patterns = [
        r"^\d+\.?\s+[A-Z].*",             # 1. INTRODUCTION
        r"^\d+\.\d+\.?\s+.*",              # 1.1 Eligibility
        r"^\d+\.\d+\.\d+\.?\s+.*",         # 1.1.1 Details
        r"^[A-Z][A-Z\s\-:,()]{5,}$",       # ALL CAPS HEADING
        r"^Annex\b.*",
        r"^Appendix\b.*",
        r"^Chapter\b.*",
        r"^Section\b.*",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def split_into_sections(page_text: str):
    """
    Split page text using detected headings.
    """

    lines = page_text.split("\n")

    sections = []
    current_heading = "General"
    current_text = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if is_heading(line):

            if current_text:
                sections.append({
                    "section": current_heading,
                    "text": "\n".join(current_text)
                })

            current_heading = line
            current_text = []

        else:
            current_text.append(line)

    if current_text:
        sections.append({
            "section": current_heading,
            "text": "\n".join(current_text)
        })

    return sections


def create_chunks(pages):
    """
    Section-aware chunk creation.

    Expected input:
        pages = [
            {
                "text": "...",
                "source": "circular.pdf",
                "page": 1
            }
        ]
    """

    chunks = []

    for page in pages:

        page_text = page["text"]
        source = page["source"]
        page_number = page["page"]

        sections = split_into_sections(page_text)

        for section_data in sections:

            section_text = section_data["text"]
            section_name = section_data["section"]

            words = section_text.split()

            start = 0

            while start < len(words):

                end = start + CHUNK_SIZE

                chunk_text = " ".join(words[start:end])

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page_number,
                    "section": section_name
                })

                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
# ----------------------------------
# 4. Build Vector Index
# ----------------------------------

DATA_DIR = Path("index")

FAISS_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
BM25_PATH = DATA_DIR / "bm25.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
## Build dense FAISS index + BM25 index

def build_index(pages):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Section-aware chunking
    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError("No chunks generated.")

    # 2. Dense embeddings
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # 3. FAISS index
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, str(FAISS_PATH))

    # 4. Save chunks and metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 5. BM25 index
    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Total chunks: {len(chunks)}")
    print("FAISS index built.")
    print("BM25 index built.")

# ----------------------------------
# 5. Retrieve Documents
# ----------------------------------

TOP_K = 5
CANDIDATE_K = 40

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4

RRF_K = 60

### New: Hybrid Dense + BM25 retrieval

def load_retrieval_index():

    faiss_index = faiss.read_index(str(FAISS_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    return faiss_index, chunks, bm25

faiss_index, chunks, bm25 = load_retrieval_index()

def retrieve(query):

    total_start = profiler.start()

    # --------------------------------
    # 1. Query Embedding
    # --------------------------------

    stage_start = profiler.start()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    profiler.record("Query Embedding", stage_start)

    # --------------------------------
    # 2. Dense Retrieval
    # --------------------------------

    stage_start = profiler.start()

    dense_scores, dense_indices = faiss_index.search(
        query_embedding,
        min(CANDIDATE_K, len(chunks))
    )

    dense_rank = {}

    for rank, idx in enumerate(dense_indices[0], start=1):

        if idx >= 0:
            dense_rank[int(idx)] = rank

    profiler.record("Dense Retrieval", stage_start)

    # --------------------------------
    # 3. BM25 Retrieval
    # --------------------------------

    stage_start = profiler.start()

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = np.argsort(bm25_scores)[::-1][
        :min(CANDIDATE_K, len(chunks))
    ]

    bm25_rank = {}

    for rank, idx in enumerate(bm25_top_indices, start=1):

        bm25_rank[int(idx)] = rank

    profiler.record("BM25 Retrieval", stage_start)

    # --------------------------------
    # 4. Hybrid RRF Fusion
    # --------------------------------

    stage_start = profiler.start()

    candidate_indices = (
        set(dense_rank.keys()) |
        set(bm25_rank.keys())
    )

    hybrid_scores = {}

    for idx in candidate_indices:

        dense_score = 0.0
        bm25_score = 0.0

        if idx in dense_rank:
            dense_score = (
                DENSE_WEIGHT /
                (RRF_K + dense_rank[idx])
            )

        if idx in bm25_rank:
            bm25_score = (
                BM25_WEIGHT /
                (RRF_K + bm25_rank[idx])
            )

        hybrid_scores[idx] = dense_score + bm25_score

    profiler.record("RRF Fusion", stage_start)

    # --------------------------------
    # 5. Final Ranking
    # --------------------------------

    stage_start = profiler.start()

    ranked_indices = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )[:TOP_K]

    results = []

    for idx in ranked_indices:

        chunk = chunks[idx].copy()

        chunk["score"] = hybrid_scores[idx]
        chunk["hybrid_score"] = hybrid_scores[idx]

        results.append(chunk)

    profiler.record("Final Ranking", stage_start)

    # --------------------------------
    # 6. Total Retrieval Latency
    # --------------------------------

    profiler.record_total(total_start)

    return results
# ----------------------------------
# 6. Generate Answer
# ----------------------------------

def generate_answer(query):

    results = retrieve(query)

    context = ""

    for result in results:

        context += (
            f"\nSource: {result['source']}"
            f"\nPage: {result['page']}"
            f"\nContent: {result['text']}\n"
        )

    prompt = f"""
You are a Banking Regulatory Assistant.

Answer ONLY using the provided context.

Rules:
1. Do not invent facts.
2. If information is missing, say:
   "The requested information is not available
   in the indexed banking circulars."
3. Give a concise and clear answer.
4. Include source document and page number.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={
            "temperature": 0.1
        }
    )

    print("\nANSWER:")
    print(response["response"])

    print("\nRETRIEVED DOCUMENTS:")

    for result in results:

        print(
            result["source"],
            "| Page:",
            result["page"],
            "| Score:",
            round(result["score"], 3)
        )


# ----------------------------------
# 8. Evaluate Retrieval
# ----------------------------------


# ---------------------------------------------
# RAG RETRIEVAL EVALUATION FRAMEWORK
# ---------------------------------------------

def evaluate_retrieval():

    # 1. Load evaluation dataset
    profiler.reset()
    with open(
        "evaluation_questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        test_data = json.load(f)

    os.makedirs("evaluation", exist_ok=True)

    # Unique evaluation run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    latencies = []

    hit_count = 0
    reciprocal_ranks = []

    print("\nStarting RAG Evaluation...")
    print("Total Questions:", len(test_data))

    # 2. Evaluate each question

    for item in test_data:

        query = item["question"]

        expected_source = item.get(
            "expected_source", ""
        )

        expected_pages = item.get(
            "expected_page", []
        )

        if isinstance(expected_pages, int):
            expected_pages = [expected_pages]

        start_time = time.perf_counter()

        results = retrieve(query)

        latency = time.perf_counter() - start_time

        latencies.append(latency)

        # 3. Check source and page matching

        found_rank = 0
        page_found = False

        retrieved_chunks = []

        for rank, result in enumerate(results, start=1):

            source = result["source"]
            page = result["page"]

            source_match = (
                str(source).lower() == str(expected_source).lower()
            )

            page_match = (
                not expected_pages
                or page in expected_pages
            )

            if source_match and found_rank == 0:

                found_rank = rank

            if source_match and page_match:

                page_found = True

            retrieved_chunks.append({

                "rank": rank,
                "source": source,
                "page": page,
                "score": round(result.get("hybrid_score", result.get("score", 0.0)), 4),
                "text": result["text"]

            })

        # 4. Calculate retrieval metrics

        source_hit = found_rank > 0

        if source_hit:

            hit_count += 1

            reciprocal_ranks.append(
                1 / found_rank
            )

        else:

            reciprocal_ranks.append(0)

        # 5. Save individual question results

        evaluation_result = {

            "run_id": run_id,

            "question_id": item.get(
                "question_id", ""
            ),

            "question": query,

            "category": item.get(
                "question_category", ""
            ),

            "difficulty": item.get(
                "difficulty", ""
            ),

            "expected_source": expected_source,

            "expected_page": expected_pages,

            "expected_answer": item.get(
                "expected_answer", ""
            ),

            "retrieved_chunks": retrieved_chunks,

            "correct_source_rank": found_rank,

            "source_hit_at_5": source_hit,

            "correct_page_found": page_found,

            "retrieval_latency_sec": round(
                latency, 4
            )

        }

        all_results.append(evaluation_result)

        # 6. Print individual results

        print("\n" + "=" * 60)

        print("Question:", query)

        print("Expected Source:", expected_source)

        print("Expected Pages:", expected_pages)

        print("Correct Source Rank:", found_rank)

        print("Correct Page Found:", page_found)

        print("Latency:", round(latency, 4), "sec")

        print("\nRetrieved Results:")

        for chunk in retrieved_chunks:

            print(
                f"Rank {chunk['rank']} | "
                f"{chunk['source']} | "
                f"Page {chunk['page']} | "
                f"Score {chunk['score']}"
            )

            print(
                "Text:",
                chunk["text"][:250].replace("\n", " "),
                "..."
            )

    # 7. Calculate overall metrics

    total = len(test_data)

    hit_rate = hit_count / total if total else 0

    mrr = (
        sum(reciprocal_ranks) / total
        if total else 0
    )

    sorted_latencies = sorted(latencies)

    if total:

        p95_index = max(
            0,
            int(0.95 * total + 0.999999) - 1
        )

        p95_latency = sorted_latencies[p95_index]

        avg_latency = statistics.mean(latencies)

        max_latency = max(latencies)

    else:

        avg_latency = 0
        p95_latency = 0
        max_latency = 0

    page_hit_count = sum(
        r["correct_page_found"]
        for r in all_results
    )

    # 8. Create summary

    summary = {

        "run_id": run_id,

        "timestamp": datetime.now().isoformat(),

        "total_questions": total,

        "hit_at_5": round(hit_rate, 4),

        "mrr_at_5": round(mrr, 4),

        "correct_page_rate": round(
            page_hit_count / total, 4
        ) if total else 0,

        "average_latency_sec": round(
            avg_latency, 4
        ),

        "p95_latency_sec": round(
            p95_latency, 4
        ),

        "max_latency_sec": round(
            max_latency, 4
        )

    }

    # 9. Save JSON results

    json_path = (
        f"evaluation/results_{run_id}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # 10. Save CSV results

    csv_path = (
        f"evaluation/results_{run_id}.csv"
    )

    csv_columns = [

        "question_id",
        "question",
        "category",
        "difficulty",
        "expected_source",
        "expected_page",
        "correct_source_rank",
        "source_hit_at_5",
        "correct_page_found",
        "retrieval_latency_sec",
        "retrieved_chunks"

    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_columns
        )

        writer.writeheader()

        for result in all_results:

            row = result.copy()

            row["retrieved_chunks"] = json.dumps(
                row["retrieved_chunks"],
                ensure_ascii=False
            )

            writer.writerow({

                key: row.get(key, "")
                for key in csv_columns

            })

    # 11. Save summary

    summary_path = (
        f"evaluation/summary_{run_id}.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # 12. Display final report

    print("\n" + "=" * 60)

    print("          RAG EVALUATION SUMMARY")

    print("=" * 60)

    for key, value in summary.items():

        print(f"{key}: {value}")

    print("\nFiles saved successfully:")

    print(json_path)
    print(csv_path)
    print(summary_path)

    print("\nEvaluation completed!")
    profiler.summary()

# ----------------------------------
# 7. Main
# ----------------------------------

if __name__ == "__main__":

    choice = input(
        "Enter 1 to build index, 2 to ask question:, 3 for evaluation "
    )

    if choice == "1":

        
        pdf_folder = Path("data/pdfs")

        pages = []

        for pdf_path in pdf_folder.glob("*.pdf"):
            extracted_pages = extract_pdf_text(pdf_path)
            pages.extend(extracted_pages)

        build_index(pages)


    elif choice == "2":

        query = input("Enter your question: ")

        generate_answer(query)
    elif choice == "3":

        evaluate_retrieval()