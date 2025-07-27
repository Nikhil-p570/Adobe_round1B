# Adobe_round1B
 🧠 Persona-Driven Document Intelligence System
Team Name : NAN
A system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

---

## 🚀 Overview

This tool reads multiple PDF documents and intelligently identifies sections that best match a user-defined **persona** and **task**. It's ideal for automating document analysis, content recommendation, or contextual section extraction based on user intent.

---

## 📌 Features

- ✨ Context-aware section extraction
- 🤖 Semantic understanding using transformer-based models
- 🧾 PDF layout parsing with heading-body detection
- 🔑 Dynamic keyword extraction (YAKE)
- 🎯 Fine-grained cross-encoder re-ranking
- 📊 Scoring with heuristics: intent match, specificity, density
- 📂 Outputs JSON with ranked sections and cleaned text

---

## 🧰 Models and Libraries Used

- **PyMuPDF (fitz):** PDF layout parsing  
- **sentence-transformers (all-MiniLM-L6-v2):** Semantic similarity scoring (bi-encoder)  
- **sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2):** Relevance re-ranking (cross-encoder)  
- **yake:** Dynamic keyword extraction  
- **regex (re):** Pattern-based filtering, text analysis, and intent detection  
- **numpy:** Vector manipulation and numerical computations  
- **collections (defaultdict):** Grouping and organizing sections by document  
- **json, os, datetime:** File I/O, path management, and timestamping  
- **tqdm:** Progress bars during long-running operations  

---

## ⚙️ How It Works

1. **Input Configuration**  
   Provide a JSON file with:
   - The **persona** (e.g., `"travel planner"`)
   - A **task** (e.g., `"plan a 5-day trip for 6 college students"`)
   - A list of **PDF filenames**

2. **Context Query Generation**  
   Expands the persona and task into multiple rich search queries to guide document understanding.

3. **PDF Parsing**  
   Extracts headings and body sections using font size, style, and layout via PyMuPDF.

4. **Keyword Extraction**  
   Uses YAKE to generate task-specific keywords that help prioritize relevant content.

5. **Semantic Ranking**  
   - Bi-encoder (`all-MiniLM-L6-v2`) generates similarity scores for all sections.
   - Cross-encoder (`ms-marco-MiniLM-L-6-v2`) refines ranking based on exact query-section match.

6. **Heuristic Scoring**  
   Boosts content that includes prices, locations, tips, group info, or dining keywords.

7. **Top-K Selection**  
   Picks top-ranked, deduplicated sections and saves them with metadata in a JSON file.

---


