# Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives

A research implementation for cross-lingual Retrieval-Augmented Generation (RAG) over historical legislative archives, with a focus on OCR noise, multilingual queries, document retrieval, and evidence-grounded answer generation.

**Paper:** *Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives*  
**Venue:** CVIP 2026  
**Repository:** [GitHub](https://github.com/radhakrishnadeshpande/Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Key Features](#key-features)
- [System Pipeline](#system-pipeline)
- [Research and Evaluation](#research-and-evaluation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset and Data Preparation](#dataset-and-data-preparation)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

Historical legislative archives contain valuable information about government decisions, parliamentary proceedings, public expenditure, administrative policies, and legislative activities. However, extracting this information reliably is challenging because archival documents frequently contain degraded scans, complex layouts, tables, inconsistent typography, and OCR errors.

This project investigates a cross-lingual RAG pipeline designed to retrieve relevant evidence from legislative archives and generate answers to questions expressed in different languages.

The research focuses on the **Punjab Vidhan Sabha archives**, including historical documents containing Punjabi text in Gurmukhi script and English content. It explores how OCR quality, multilingual semantic retrieval, iterative query refinement, and evidence-grounded generation affect the reliability of question answering over archival documents.

The central objective is to make historical legislative information more accessible without losing important details during OCR, retrieval, or answer generation.

## Motivation

Conventional RAG systems can struggle with historical documents for several reasons:

- **OCR noise:** Incorrectly recognized characters and words can distort important names, dates, figures, and administrative terminology.
- **Cross-lingual queries:** The language of a user's question may differ from the language or script of the retrieved document.
- **Complex document layouts:** Tables, columns, and mixed narrative content can make it difficult to preserve relationships between extracted values.
- **Historical terminology:** Older administrative language, spelling variations, and historical currency conventions can complicate semantic matching.
- **Evidence consistency:** A fluent generated answer is not necessarily supported by the correct archival source.

This project studies these challenges in the context of legislative information retrieval and question answering.

## Key Features

- **Cross-lingual retrieval:** Explore retrieval across language boundaries, including Punjabi queries and English archival content.
- **OCR-aware processing:** Investigate how OCR quality affects downstream retrieval and answer generation.
- **Evidence-grounded question answering:** Generate answers using information retrieved from relevant archival documents.
- **Iterative retrieval and query refinement:** Evaluate an agentic retrieval loop that can refine a query over multiple iterations.
- **Multiple OCR systems:** Compare OCR outputs and their downstream impact using the systems evaluated in the research.
- **Quantitative evaluation:** Assess retrieval effectiveness and OCR quality using defined metrics.
- **Historical document analysis:** Address archival material containing tables, administrative records, dates, names, and numerical values.

## System Pipeline

The research workflow can be summarized as follows:

```text
Historical Legislative Documents
              |
              v
     Document Images / Scans
              |
              v
       OCR and Text Extraction
              |
              v
   Text Processing and Indexing
              |
              v
       User Query (Multilingual)
              |
              v
 Cross-Lingual Retrieval and Ranking
              |
              v
 Iterative Retrieval / Query Refinement
              |
              v
      Evidence-Grounded Generation
              |
              v
       Answer with Source Evidence
              |
              v
       Retrieval and OCR Evaluation
```

The exact execution order and implementation details depend on the individual Python modules and their configurations.

## Research and Evaluation

### 1. OCR quality

OCR systems are evaluated on archival document text to investigate recognition errors and their potential effect on downstream retrieval.

The research includes comparisons involving the OCR systems used in the experiments, including Gemini, Sarvam, Surya2, and Ourocr.

### 2. Cross-lingual retrieval

The retrieval component is evaluated on questions that require matching a query with relevant information in legislative documents, including scenarios where the query and source content use different languages.

### 3. Agentic retrieval

An iterative retrieval-and-rewriting strategy is investigated to examine whether refining a query over multiple retrieval steps can improve evidence selection and answer quality.

### 4. Answer generation

Generated answers are assessed against reference information to study whether the system can preserve the factual details contained in the retrieved evidence.

### Evaluation results

The repository accompanies the research paper, which documents the experimental setup, evaluation methodology, and reported results.

Refer to the paper and the corresponding experiment scripts for the precise dataset splits, model configurations, evaluation settings, and numerical results. Results should be interpreted within the scope of the evaluated dataset and experimental conditions.

## Repository Structure

The repository contains Python source code for the research implementation. The exact file organization may evolve as the code is updated.

A suggested organization for maintaining the project is:

```text
Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives/
│
├── README.md
├── requirements.txt                 # If provided
├── .gitignore
│
├── data/                             # Local data; not committed by default
├── preprocessing/                    # Document/OCR processing, if separated
├── retrieval/                        # Retrieval components, if separated
├── generation/                       # Answer generation, if separated
├── evaluation/                       # Evaluation scripts, if separated
└── *.py                              # Research implementation scripts
```

This is an illustrative organizational layout, not a claim that all these directories already exist. Consult the actual Python files to identify the implemented modules and entry points.

## Getting Started

### Prerequisites

Before running the implementation, ensure that you have:

- Python compatible with the project's dependencies.
- Git.
- Access to the required archival data or preprocessed documents.
- The model checkpoints and services required by the selected pipeline components.
- API credentials for any external model services used by your chosen configuration.

GPU availability depends on the models and processing stages you intend to run.

### 1. Clone the repository

```bash
git clone https://github.com/radhakrishnadeshpande/Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives.git

cd Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n legislative-rag python=3.10 -y
conda activate legislative-rag
```

Alternatively, use Python's built-in virtual environment support:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Choose a Python version compatible with the dependencies used by the actual implementation.

### 3. Install dependencies

If the repository provides a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If no dependency file is provided, install the packages required by the specific scripts you intend to run. The required packages may differ depending on the OCR provider, embedding model, vector store, and language model used.

### 4. Prepare the data

Obtain the authorized document images, OCR outputs, or processed text required by the experiment. Configure the relevant paths in the implementation.

Large datasets, proprietary documents, model weights, and generated indexes may need to be downloaded separately. They are not necessarily included in this GitHub repository.

## Configuration

Before execution, check the relevant Python scripts for:

- Input document or image directories.
- OCR provider and model selection.
- OCR language and modality settings.
- Embedding model configuration.
- Vector database or index paths.
- Language model and generation settings.
- Retrieval top-k and ranking parameters.
- Maximum retrieval-refinement iterations.
- Evaluation dataset and output directories.

Use environment variables or a local configuration file where supported.

**Security:** Never commit API keys, access tokens, passwords, private document collections, or other confidential information. Store credentials in environment variables or an appropriate secret manager.

## Running the Code

The Python files in this repository may support different stages of the research workflow, such as OCR processing, indexing, retrieval, generation, and evaluation.

To identify the available scripts:

```bash
find . -type f -name "*.py" -not -path "./.git/*"
```

To inspect a script's command-line arguments, where supported:

```bash
python path/to/script.py --help
```

Replace `path/to/script.py` with the actual script path.

Run the appropriate entry point only after checking its required inputs, configuration, and dependencies. Some scripts may be intended as experiment-specific utilities rather than standalone applications.

## Evaluation Metrics

The research evaluates retrieval effectiveness and OCR quality.

### Retrieval metrics

- **MRR@1 (Mean Reciprocal Rank at 1):** Measures whether the first retrieved result is relevant under the benchmark's relevance definition.
- **Hit@k:** Measures whether at least one relevant document appears in the top-k retrieved results.
- **Precision@3:** Measures the fraction of the top three retrieved documents that are relevant, averaged according to the evaluation protocol.

### OCR quality metrics

OCR evaluation examines the quality of recognized text and its potential downstream effects. The precise metrics, normalization rules, reference text, and scoring procedures should be taken from the evaluation scripts and research paper.

### Reproducing the evaluation

For meaningful comparisons:

1. Use the same evaluation dataset and query set.
2. Keep preprocessing and text normalization consistent.
3. Use the same retrieval settings and ranking configuration.
4. Record the OCR and language model versions.
5. Report the metrics using the same definitions as the paper.
6. Record the configuration and random seeds wherever applicable.

Do not compare results across different evaluation conditions without documenting the differences.

## Dataset and Data Preparation

The research focuses on historical Punjab Vidhan Sabha documents.

Depending on the experiment, inputs may include:

- Scanned legislative pages.
- OCR-generated text.
- Processed or indexed document representations.
- Evaluation queries and relevance annotations.
- Reference answers used for evaluation.

The dataset, annotations, and preprocessing outputs should be handled according to their applicable access and usage permissions.

If the underlying archival documents or benchmark annotations are not publicly distributed, users must obtain access independently. The availability of this source code does not imply that the complete dataset or external model services are included.

## Reproducibility

To make experimental results easier to reproduce, record:

- OCR model and version.
- Embedding model and version.
- Generation model and version.
- Dataset version and preprocessing procedure.
- Retrieval and reranking parameters.
- Maximum number of iterative retrieval steps.
- Evaluation script and metric definitions.
- Hardware, software dependencies, and relevant configuration.

Exact numerical reproduction may depend on model versions, external services, computational resources, and dataset availability.

## Limitations

This research addresses important challenges in OCR-based cross-lingual retrieval, but several limitations remain:

- Recognition errors may still affect retrieval and numerical accuracy.
- Complex tables and historical layouts can be difficult to interpret reliably.
- Retrieval quality depends on document coverage, indexing, and query formulation.
- Generated answers require verification against the original archival sources.
- Performance on the evaluated corpus may not generalize to every language, archive, or document type.

For historical, legal, or administrative research, users should verify important claims and numerical values against the cited source documents.

## Citation

If you use this implementation in academic work, please cite the associated paper.

**Paper title:** *Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives*  
**Venue:** CVIP 2026

Add the final bibliographic citation here once the official proceedings metadata, author list, and publication details are available.

## Acknowledgements

This project investigates the intersection of OCR, multilingual information retrieval, and generative question answering for historical legislative archives. It aims to support research into more accessible and reliable retrieval of information from archival government documents.

## License

Specify the repository's applicable license here. If no license has been added, users should not assume that the source code is licensed for unrestricted reuse or redistribution.

---

For questions about the implementation, experimental setup, or research methodology, please open an issue in this repository with the relevant script, configuration, and error details.
# Reading-Beyond-OCR-Noise-Robust-Cross-Lingual-RAG-for-Legislative-Archives
