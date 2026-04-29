# Vanilla Transformer + RAG Pipeline From Scratch

This repository contains the implementation for NLP Assignment 3. The goal of this project was to build a complete Transformer architecture and a Retrieval-Augmented Generation (RAG) pipeline from scratch using standard PyTorch linear layers, without relying on high-level transformer libraries.

The model is trained on a balanced subset of 30,000 Amazon Reviews to predict sentiment, estimate review length, and generate context-aware explanations.

## System Architecture

The pipeline is split into three main components:

* **Part A: Multi-Task Encoder** A custom Transformer Encoder that processes sequences and outputs a fixed-dimensional <CLS> embedding. It uses shared representations to simultaneously predict:
  1. Sentiment (Negative, Neutral, Positive)
  2. Word Count (Regression)

* **Part B: RAG Retrieval Module**
  A vector search engine that calculates cosine similarity between a query's <CLS> embedding and the training set to retrieve the top-k most semantically similar reviews.

* **Part C: Autoregressive Decoder**
  A causal (lower-triangular masked) Decoder model that takes the original review, the predicted sentiment, and the retrieved RAG context to generate a token-by-token explanation.

## Repository Structure

* `data_sampler.py`: Memory-efficient script to extract and balance 30,000 reviews from raw JSON files.
* `i211672_NLP_Assignment3.ipynb`: Main Jupyter Notebook containing the data loaders, architecture, training loop, and evaluation.
* `models/`: Directory containing the saved `encoder_weights.pth` after training.
* `results/`: Directory containing the extracted <CLS> embeddings (`train_embeddings.pt`) used for the RAG search.
* `Report.pdf`: A detailed 3-page write-up covering design justifications, hyperparameters, and an ablation study on RAG perplexity.

## Tech Stack
* Python 3
* PyTorch
* Scikit-Learn
* NumPy

## How to Run
1. Ensure the raw Amazon review JSON files are in the working directory.
2. Run `python data_sampler.py` to generate the `sampled_dataset.json`.
3. Execute all cells in `i211672_NLP_Assignment3.ipynb`.
4. The notebook will train the encoder, save weights, extract RAG embeddings, and generate text explanations.

---
*Developed by Waleed Saeed (21i-1672) for CS-4063: Natural Language Processing.*