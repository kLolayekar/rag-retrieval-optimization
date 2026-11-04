
import os
import json
import fitz
import faiss
import ollama
import time
import csv
import statistics
from datetime import datetime
from sentence_transformers import SentenceTransformer


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

def create_chunks(pages):

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page in pages:

        words = page["text"].split()

        for start in range(0, len(words), step):

            chunk_words = words[start:start + CHUNK_SIZE]

            if not chunk_words:
                continue

            chunk = {
                "text": " ".join(chunk_words),
                "source": page["source"],
                "page": page["page"]
            }

            chunks.append(chunk)

    return chunks


# ----------------------------------
# 4. Build Vector Index
# ----------------------------------

def build_index():

    all_chunks = []

    for filename in os.listdir(PDF_FOLDER):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(
                PDF_FOLDER, filename
            )

            print("Processing:", filename)

            pages = extract_pdf_text(pdf_path)

            chunks = create_chunks(pages)

            all_chunks.extend(chunks)

    print("Total chunks:", len(all_chunks))

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    # Exact cosine similarity using normalized vectors
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        f"{INDEX_FOLDER}/faiss.index"
    )

    with open(
        f"{INDEX_FOLDER}/chunks.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False
        )

    print("Index created successfully!")


# ----------------------------------
# 5. Retrieve Documents
# ----------------------------------

def retrieve(query):

    index = faiss.read_index(
        f"{INDEX_FOLDER}/faiss.index"
    )

    with open(
        f"{INDEX_FOLDER}/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    results = []

    for score, idx in zip(
        scores[0], indices[0]
    ):

        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score)
        })

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
                "score": round(result["score"], 4),
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

# ----------------------------------
# 7. Main
# ----------------------------------

if __name__ == "__main__":

    choice = input(
        "Enter 1 to build index, 2 to ask question: "
    )

    if choice == "1":

        build_index()


    elif choice == "2":

        query = input("Enter your question: ")

        generate_answer(query)
    elif choice == "3":

        evaluate_retrieval()