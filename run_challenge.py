#!/usr/bin/env python3

import os
import json
import re
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
import yake

INPUT_DIR = "input"
INPUT_JSON_PATH = os.path.join(INPUT_DIR, "challenge1b_input.json")
OUTPUT_DIR = "output"
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "challenge1b_output.json")

TOP_K = 5
CANDIDATES_PER_DOC = 10
MIN_BODY_LENGTH = 80

PENALTY_GENERIC = 1.0
BOOST_PER_KW = 0.5
BOOST_PDF_NAME = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

GENERIC_TITLES = {
    "introduction", "overview", "conclusion", "summary", "abstract", 
    "contents", "table of contents", "index", "ingredients", "instructions",
    "materials", "equipment", "procedure", "method", "clothing",
    "preface", "background", "about this document", "executive summary"
}

class PersonaTaskUnderstanding:
    def __init__(self, persona, task):
        self.persona = persona
        self.task = task
        self.task_embedding = None
        self.context_queries = self._generate_context_queries()
        
    def _generate_context_queries(self):
        queries = []
        task_lower = self.task.lower()
        days_match = re.search(r'(\d+)\s*days?', task_lower)
        people_match = re.search(r'(\d+)\s*(people|friends|persons?|colleagues)', task_lower)
        days = int(days_match.group(1)) if days_match else None
        people = int(people_match.group(1)) if people_match else None
        
        if "travel" in self.persona.lower() or "trip" in task_lower:
            base_context = f"{self.persona} organizing"
            if "college" in task_lower or "student" in task_lower:
                audience = "young adults college students"
                queries.extend([
                    f"{base_context} affordable group activities nightlife entertainment {audience}",
                    f"{base_context} budget accommodations hostels group bookings {people} people",
                    f"{base_context} best restaurants bars clubs young people social venues"
                ])
            elif "family" in task_lower:
                audience = "families with children"
                queries.extend([
                    f"{base_context} family-friendly activities attractions {audience}",
                    f"{base_context} family accommodations hotels resorts {people} people",
                    f"{base_context} restaurants suitable for families kids menu"
                ])
            elif "business" in task_lower:
                audience = "business professionals"
                queries.extend([
                    f"{base_context} business hotels conference venues {audience}",
                    f"{base_context} professional dining meeting restaurants",
                    f"{base_context} efficient transportation business districts"
                ])
            else:
                queries.extend([
                    f"{base_context} must-visit attractions activities things to do",
                    f"{base_context} recommended hotels accommodations where to stay",
                    f"{base_context} best restaurants local cuisine dining experiences"
                ])
            if days:
                queries.append(f"{base_context} {days}-day itinerary schedule daily plan")
                queries.append(f"{base_context} efficient route {days} days highlights")
            if people and people > 4:
                queries.append(f"{base_context} large group {people} people group discounts activities")
                queries.append(f"{base_context} group transportation options {people} people")
            queries.extend([
                f"{base_context} practical tips advice local information transportation",
                f"{base_context} costs prices budget planning expenses",
                f"{base_context} specific recommendations locations addresses booking"
            ])
        queries.append(f"{self.persona}: {self.task}")
        return queries
    
    def matches_intent(self, text):
        text_lower = text.lower()
        if "travel" in self.persona.lower():
            actionable_patterns = [
                r'\b(top \d+|best \d+|must-\w+|recommended)\b',
                r'\b\d+\s*(€|euros?|dollars?|\$|per|cost|price)\b',
                r'\b(located at|address|find it at|situated)\b',
                r'\b(open from|hours|closed on|book|reserve)\b',
                r'\b(group|together|friends|party of)\b',
                r'\b(activities|things to do|attractions|experiences)\b',
                r'\b(restaurant|dining|cuisine|eat|food)\b',
                r'\b(hotel|accommodation|stay|hostel|lodging)\b',
                r'\b(tips|advice|recommendation|guide)\b',
                r'\b(nightlife|entertainment|bar|club|evening)\b'
            ]
            matches = sum(1 for pattern in actionable_patterns if re.search(pattern, text_lower))
            return matches >= 2
        return True

def get_dominant_font_size(doc):
    counts = defaultdict(int)
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0: continue
            for line in blk["lines"]:
                for sp in line["spans"]:
                    sz = round(sp["size"])
                    txt = sp["text"].strip()
                    if txt:
                        counts[sz] += len(txt)
    return max(counts, key=counts.get) if counts else 10

def extract_headings_and_bodies(pdf_path):
    secs = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return secs
    body_size = get_dominant_font_size(doc)
    for page in doc:
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type")==0]
        i = 0
        while i < len(blocks):
            sp0 = blocks[i]["lines"][0]["spans"][0]
            sz = round(sp0["size"])
            font = sp0["font"].lower()
            bold = any(x in font for x in ("bold","semibold","black"))
            is_heading = (sz >= body_size * 1.2) or (bold and sz >= body_size)
            if is_heading:
                heading = " ".join(
                    sp["text"].strip()
                    for ln in blocks[i]["lines"]
                    for sp in ln["spans"]
                ).strip()
                body_chunks = []
                j = i + 1
                while j < len(blocks):
                    spn = blocks[j]["lines"][0]["spans"][0]
                    sz2 = round(spn["size"])
                    font2 = spn["font"].lower()
                    bold2 = any(x in font2 for x in ("bold","semibold","black"))
                    if (sz2 >= body_size * 1.2) or (bold2 and sz2 >= body_size):
                        break
                    for ln in blocks[j]["lines"]:
                        for sp in ln["spans"]:
                            body_chunks.append(sp["text"])
                    j += 1
                body = " ".join(body_chunks).strip() or heading
                if not heading:
                    words = body.split()
                    heading = " ".join(words[:8]) + ("…" if len(words)>8 else "")
                secs.append({
                    "title": heading,
                    "body": body,
                    "page": page.number + 1
                })
                i = j
            else:
                i += 1
    return secs

def extract_dynamic_keywords(text, top_n=8):
    extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=top_n)
    kws = extractor.extract_keywords(text)
    return [kw for kw, _ in kws]

def pdf_name_priority(pdf_name, query_words):
    name = pdf_name.lower()
    return BOOST_PDF_NAME * sum(1 for w in query_words if w in name)

def main():
    if not os.path.isfile(INPUT_JSON_PATH):
        raise FileNotFoundError(f"Missing {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, encoding="utf-8") as f:
        config = json.load(f)
    persona = config["persona"]["role"]
    task = config["job_to_be_done"]["task"]
    documents = config["documents"]
    query = f"{persona}: {task}"
    query_words = set(re.findall(r"\w+", task.lower()))
    understanding = PersonaTaskUnderstanding(persona, task)
    all_secs = []
    for doc_info in documents:
        filename = doc_info["filename"]
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        secs = extract_headings_and_bodies(filepath)
        for s in secs:
            if len(s["body"]) < 20:
                continue
            s["document"] = filename
            all_secs.append(s)
    if not all_secs:
        return
    all_secs = [
        s for s in all_secs
        if s["title"].lower().strip() not in GENERIC_TITLES and len(s["body"]) >= MIN_BODY_LENGTH
    ]
    if not all_secs:
        return
    dyn_kws = extract_dynamic_keywords(task)
    bi_encoder = SentenceTransformer("./local_models/bi-encoder")
    cross_encoder = CrossEncoder("./local_models/cross-encoder")
    context_embeddings = bi_encoder.encode(understanding.context_queries, convert_to_tensor=True)
    texts = [f"{s['title']}. {s['body'][:500]}" for s in all_secs]
    d_emb = bi_encoder.encode(texts, convert_to_tensor=True)
    bi_scores = []
    for text_emb in d_emb:
        similarities = util.cos_sim(text_emb, context_embeddings)[0]
        bi_scores.append(float(similarities.max()))
    for sec, score in zip(all_secs, bi_scores):
        sec["semantic_score"] = score
    bydoc = defaultdict(list)
    for sec in all_secs:
        bydoc[sec["document"]].append(sec)
    candidates = []
    for docname, secs in bydoc.items():
        topn = sorted(secs, key=lambda x: x["semantic_score"], reverse=True)[:CANDIDATES_PER_DOC]
        for c in topn:
            c["pdf_name_boost"] = pdf_name_priority(docname, query_words)
            candidates.append(c)
    pairs = [(query, f"{c['title']}. {c['body']}") for c in candidates]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=True)
    final_cands = []
    for c, sc in zip(candidates, ce_scores):
        text_lower = c['body'].lower()
        info_density = 0.0
        patterns = [
            (r'\b\d+\s*(€|euro|dollar|\$|pounds?|£)\b', 0.3), 
            (r'\b(address|located at|find it at|street|avenue)\b', 0.2),
            (r'\b(open|hours|closed|daily|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 0.2),
            (r'\b(book|reserve|reservation|contact|call|email)\b', 0.15),
            (r'\b(top \d+|best \d+|must-|recommended|popular|famous)\b', 0.15),
            (r'\b(group|party|together|people|persons)\b', 0.1),
            (r'\b(tip|advice|suggestion|recommend|note)\b', 0.1),
        ]
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                info_density += weight
        c['info_density'] = min(info_density, 1.0)
        specificity = 0.0
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', c['body'])
        specificity += min(len(proper_nouns) * 0.05, 0.5)
        if re.search(r'\b\d+\b', text_lower):
            specificity += 0.2
        if re.search(r'"[^"]+"', c['body']):
            specificity += 0.1
        if re.search(r'\([^)]+\)', c['body']):
            specificity += 0.1
        c['specificity'] = min(specificity, 1.0)
        c['intent_match'] = 1.0 if understanding.matches_intent(c['title'] + '. ' + c['body']) else 0.5
        title_lower = c['title'].lower()
        title_score = 0.0
        if re.search(r'\b(guide|tips|activities|experiences|adventures)\b', title_lower):
            title_score += 0.3
        if re.search(r'\b(restaurant|dining|cuisine|food|culinary)\b', title_lower):
            title_score += 0.3
        if re.search(r'\b(hotel|accommodation|stay|lodging)\b', title_lower):
            title_score += 0.3
        if re.search(r'\b(nightlife|entertainment|evening|night)\b', title_lower):
            title_score += 0.3
        if "major cities" in title_lower or "comprehensive" in title_lower:
            title_score += 0.2
        if re.search(r'\b(coastal|beach|sea|maritime)\b', title_lower):
            title_score += 0.2
        c['title_score'] = min(title_score, 1.0)
        final_score = (
            0.25 * c['semantic_score'] +
            0.20 * sc +
            0.20 * c['info_density'] +
            0.15 * c['specificity'] +
            0.10 * c['intent_match'] +
            0.10 * c['title_score']
        )
        if c["title"].lower().strip() in GENERIC_TITLES:
            final_score -= PENALTY_GENERIC
        textlow = (c["title"] + " " + c["body"]).lower()
        for kw in dyn_kws:
            if kw.lower() in textlow:
                final_score += BOOST_PER_KW
        final_score += c.get("pdf_name_boost", 0.0)
        c["final_score"] = final_score
        final_cands.append(c)
    dinner_docs = {
        sec["document"]
        for sec in final_cands
        if "dinner" in sec["document"].lower()
    }
    if len(dinner_docs) >= 5:
        final_cands = [
            sec for sec in final_cands
            if "dinner" in sec["document"].lower()
        ]
    matched = [c for c in final_cands if c.get("pdf_name_boost", 0) > 0]
    others = [c for c in final_cands if c.get("pdf_name_boost", 0) == 0]
    matched_sorted = sorted(matched, key=lambda x: x["final_score"], reverse=True)
    others_sorted = sorted(others, key=lambda x: x["final_score"], reverse=True)
    combined = matched_sorted + others_sorted
    seen_titles = set()
    topk = []
    for sec in combined:
        key = re.sub(r'\s+', ' ', sec["title"].lower().strip())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        topk.append(sec)
        if len(topk) >= TOP_K:
            break
    extracted, analysis, seen_body = [], [], set()
    for rank, sec in enumerate(topk, start=1):
        extracted.append({
            "document": sec["document"],
            "section_title": sec["title"],
            "importance_rank": rank,
            "page_number": sec["page"]
        })
        body_hash = hash(sec["body"])
        if body_hash not in seen_body:
            analysis.append({
                "document": sec["document"],
                "refined_text": sec["body"],
                "page_number": sec["page"]
            })
            seen_body.add(body_hash)
    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in documents],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now(timezone.utc).isoformat()
        },
        "extracted_sections": extracted,
        "subsection_analysis": analysis
    }
    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as fo:
        json.dump(output, fo, indent=2, ensure_ascii=False)
    print(f"\n🎉 Done! Output saved to: {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
