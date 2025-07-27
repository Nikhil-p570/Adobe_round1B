# 🎯 📄 Adobe India Hackathon 2025 – Round 1B Submission

### 🔍 Challenge: “Connect What Matters – For the User Who Matters”
 **Team Name:** NAN

---

## 🧠 Project Summary

An intelligent, persona-driven system that analyzes a collection of PDFs and extracts the most relevant sections tailored to a specific **persona** and their **job-to-be-done**. Designed to automate content filtering, contextual information retrieval, and personalized document summarization.

---

## 🚀 Overview

This system reads multiple PDF documents and intelligently identifies sections that best align with a user-defined **persona** and **task**. Applications include:

- Personalized document summarization  
- Smart travel itinerary planning  
- Contextual extraction for digital assistants  
- Content filtering for recommendation systems  

---

## 📌 Features

- ✨ **Context-aware extraction** of document sections  
- 🤖 **Semantic understanding** using transformer models  
- 🧾 **Heading + body parsing** using PDF layout detection  
- 🔑 **Dynamic keyword extraction** (via YAKE)  
- 🎯 **Cross-encoder based re-ranking** for precision  
- 📊 **Multi-factor scoring** (intent, specificity, and content density)  
- 📂 **Clean JSON output** with ranked and refined content  

---

## 🧰 Technologies & Libraries

| Category                | Library / Tool                                 |
|------------------------|-------------------------------------------------|
| 📄 PDF Parsing         | `PyMuPDF (fitz)`                                |
| 🔍 Semantic Scoring    | `sentence-transformers (bi-encoder + cross-encoder)` |
| 🧠 Relevance Ranking   | `cross-encoder/ms-marco-MiniLM-L-6-v2`          |
| 🧵 Keyword Extraction  | `YAKE`                                          |
| 🧪 Text Processing     | `regex`, `collections`, `json`                  |
| 🔢 Computation         | `numpy`                                         |
| 📁 Utilities           | `os`, `datetime`, `tqdm`                        |

---

## ⚙️ System Workflow

### 1. 📥 Input Configuration  

Users provide a config JSON file with:

```json
{
  "persona": "travel planner",
  "task": "plan a 5-day trip for 6 college students",
  "documents": [
    "South of France - Cuisine.pdf",
    "South of France - Things to Do.pdf"
  ]
}
```

---

### 2. 🧠 Process Pipeline

- **Query Expansion** — Enrich persona + task into multi-angle prompts.
- **PDF Parsing** — Headings and content blocks are extracted using layout-based rules.
- **Keyword Scoring** — Dynamic keywords are pulled to amplify context.
- **Bi-Encoder Ranking** — Fast filtering using MiniLM-based similarity.
- **Cross-Encoder Re-ranking** — Precision scoring with deeper interaction.
- **Heuristics & Boosting** — Priority for price info, group tips, location mentions, etc.
- **Deduplication + Output** — Top-ranked refined outputs are saved with metadata.

---

## 🧾 Sample Output Format

```json
{
  "metadata": {
    "input_documents": [
      "South of France - Cities.pdf",
      "South of France - Cuisine.pdf",
      "South of France - History.pdf",
      // more pdfs...
    ],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-10T15:31:22.632389"
  },
  "extracted_sections": [
    {
      "document": "South of France - Cities.pdf",
      "section_title": "Comprehensive Guide to Major Cities in the South of France",
      "importance_rank": 1,
      "page_number": 1
    },
    {
      "document": "South of France - Things to Do.pdf",
      "section_title": "Coastal Adventures",
      "importance_rank": 2,
      "page_number": 2
    }
    // more sections...
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Things to Do.pdf",
      "refined_text": "The South of France is renowned for its beautiful coastline along the Mediterranean Sea...",
      "page_number": 2
    },
    {
      "document": "South of France - Cuisine.pdf",
      "refined_text": "In addition to dining at top restaurants, there are several culinary experiences you should consider...",
      "page_number": 6
    }
    // more subsections...
  ]
}
```

---

## ✅ Constraints Satisfied

| Constraint                                | Status       |
|------------------------------------------|--------------|
| 💻 Runs on **CPU only**                  | ✅ Supported |
| 📦 Model size **≤ 1GB**                  | ✅ Compliant |
| ⏱️ Processes **3–5 PDFs in ≤ 60s**       | ✅ Achieved  |
| 🌐 Requires **No internet access**       | ✅ Offline   |

---

## 🐳 Docker Instructions

### Build the Image

```bash
docker build -t adobe-doc-intel .
```

### Run the Container

```bash
docker run --rm \
  -v ${pwd}/input:/app/input \
  -v ${pwd}/output:/app/output \
  adobe-doc-intel
```

- `input/`: Folder with `config.json` and PDFs  
- `output/`: Output `result.json` with extracted content

---

## 📂 Project Structure

```
.
├── input/
│   ├── config.json
│   └── *.pdf
├── output/
│   └── result.json
├── extract.py
├── model/
│   └── MiniLM model weights
├── utils/
│   └── helpers, scorers, preprocessors
├── Dockerfile
└── README.md
```

---

## 👥 Contributors

- Team NAN – Adobe Document Intelligence Hackathon 2025
