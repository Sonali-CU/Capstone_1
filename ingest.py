"""Ingest PDFs from data/ into a simple vector store (store/).
Run: python ingest.py
"""
import os
from utils import read_pdfs, chunk_texts, get_embeddings, build_store

PDF_FOLDER = 'data'
OUT_DIR = 'store'

def main():
    print('Reading PDFs from', PDF_FOLDER)
    docs = read_pdfs(PDF_FOLDER)
    print(f'Found {len(docs)} text chunks (pre-split).')
    docs = chunk_texts(docs, max_chars=1000)
    print(f'After chunking: {len(docs)} chunks.')
    texts = [d['text'] for d in docs]
    print('Requesting embeddings from OpenRouter (make sure OPENROUTER_API_KEY is set)...')
    embeddings = get_embeddings(texts)
    print('Embeddings created. Saving store...')
    build_store(docs, embeddings, out_dir=OUT_DIR)
    print('Done.')

if __name__ == '__main__':
    main()
