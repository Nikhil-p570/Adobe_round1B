# Persona-Driven Document Intelligence System

## Overview

This project is a sophisticated AI-powered system designed for the Adobe Hackathon Challenge 1B. It intelligently analyzes a collection of PDF documents to extract and rank the most relevant sections based on a specific user persona and their job-to-be-done. The system uses a multi-layered scoring architecture and advanced NLP models to understand user intent and content relevance with high accuracy.

## Features

* **Dynamic Persona Understanding**: Intelligently parses the persona and task to generate multiple contextual search queries.
* **Robust Section Extraction**: A hybrid heading extraction engine analyzes both font sizes and font weights (boldness) to reliably parse sections from PDFs with varying layouts.
* **Multi-Faceted Scoring**: Each section is evaluated across multiple dimensions:
    * **Semantic Similarity**: Uses a Bi-Encoder for fast relevance scoring.
    * **Cross-Encoder Reranking**: Employs a more powerful Cross-Encoder for nuanced, high-accuracy reranking of top candidates.
    * **Information Density**: Scores sections based on the presence of actionable data like prices, locations, and contact information.
    * **Specificity**: Ranks sections higher if they contain specific, concrete details like proper nouns and numbers.
* **Dynamic Keyword Boosts**: Uses YAKE (Yet Another Keyword Extractor) to identify key terms in the user's task and boosts sections that contain them.
* **Filename Prioritization**: Gives a slight score advantage to sections from documents whose filenames match keywords in the task.
* **Generic Title Filtering**: Automatically identifies and filters out low-value, generic sections like "Introduction" or "Conclusion."

## Repository Structure

Your final repository should be organized as follows:

```
/your-project-folder
├── input/
│   ├── challenge1b_input.json
│   └── ... (sample pdfs)
├── Dockerfile
├── requirements.txt
├── run_challenge.py
└── README.md
```

## How to Run

1.  **Build the Docker Image**: Navigate to the project's root directory in your terminal and run the `docker build` command.
    ```bash
    docker build -t adobe-challenge .
    ```

2.  **Run the Container**: After building the image, use the `docker run` command to execute the script. This command links your local `input` and `output` folders to the container.

    ```bash
    # Create an empty 'output' folder first
    mkdir output

    # Run the container
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-challenge
    ```
    The final `challenge1b_output.json` will be generated in your local `output` folder.

## Methodology

The script operates through a sophisticated pipeline:

1.  **Persona Analysis**: The `PersonaTaskUnderstanding` class first deconstructs the `persona` and `job_to_be_done` to create a set of diverse, context-rich search queries.
2.  **Section Extraction**: The `extract_headings_and_bodies` function processes each PDF specified in the input file, using a hybrid font-based heuristic to accurately extract all potential sections. Low-quality sections (e.g., those with very short body text or generic titles) are filtered out early.
3.  **Candidate Selection**: A fast Bi-Encoder (`all-MiniLM-L6-v2`) calculates an initial semantic score for all sections against the contextual queries. To ensure diversity, the top candidates from each document are selected for the next stage.
4.  **Advanced Scoring & Reranking**: The selected candidates are then passed through a comprehensive scoring function (`compute_section_scores`) which evaluates them on multiple criteria. This includes a high-accuracy relevance score from a Cross-Encoder model, information density, title quality, and dynamic keyword matches.
5.  **Final Selection**: The sections are ranked based on their final, multi-faceted score. The system then prioritizes sections from documents whose filenames match the query, before selecting the top 5 unique sections for the final output.