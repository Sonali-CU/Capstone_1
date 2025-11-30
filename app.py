"""Streamlit RAG Chat UI (OpenRouter)
Run: streamlit run app.py
"""
import streamlit as st
import openai, os
from utils import SimpleRetriever, EMBED_MODEL
st.set_page_config(page_title='RAG Chatbot - NCERT For All Science Lovers', layout='wide')

openai.api_key = os.getenv('OPENROUTER_API_KEY')

openai.api_base = "https://openrouter.ai/api/v1"


st.title('RAG Chatbot — NCERT Class 9,10,11,12 Math,Science(Physics,Chemistry,Biology) Tutor')
st.write('Ask questions about NCERT Class 9,10,11,12 Maths and Science(Physics,Chemistry,Biology) PDFs placed in /data and ingested.')

if 'retriever' not in st.session_state:
    try:
        st.session_state.retriever = SimpleRetriever(store_dir='store')
    except Exception as e:
        st.warning('Vector store not ready. Run `python ingest.py` first. Error: ' + str(e))

query = st.text_input('Your question:', '')
k = st.slider('Number of retrieved chunks to use in context', 1, 6, 3)

if st.button('Ask') and query.strip():
    if openai.api_key is None:
        st.error('OPENROUTER_API_KEY not set in environment.')
    else:
        retriever = st.session_state.get('retriever', None)
        if retriever is None:
            st.error('Retriever not available. Run ingest first.')
        else:
            # embed query via OpenRouter
            emb_resp = openai.Embedding.create(model=EMBED_MODEL, input=[query])
            q_emb = emb_resp['data'][0]['embedding']
            results = retriever.retrieve(q_emb, k=k)
            context_texts = []
            for r in results:
                md = r['doc']
                context_texts.append(f"(Source: {md['source']} page:{md['page']})\n{md['text']}")
            context = '\n\n---\n\n'.join(context_texts)
            prompt = f"You are a helpful tutor for Class 10 Mathematics. Use only the provided context to answer the question. If the context doesn't contain the answer, say you can't find it and give a short hint.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            # Chat completion via OpenRouter - specify a model available on OpenRouter
            completion = openai.ChatCompletion.create(
                model='openai/gpt-3.5-turbo',
                messages=[{'role':'system','content':'You are a helpful assistant.'},
                          {'role':'user','content':prompt}],
                max_tokens=400
            )
            answer = completion['choices'][0]['message']['content']
            st.subheader('Answer')
            st.write(answer)
            st.subheader('Retrieved passages (for transparency)')
            for r in results:
                md = r['doc']
                st.markdown(f"**Source:** {md['source']} — page {md['page']} — score {r['score']:.3f}")
                st.write(md['text'])
