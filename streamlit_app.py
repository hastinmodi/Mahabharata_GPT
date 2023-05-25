# reference 1 - https://www.youtube.com/watch?v=Dhc_fq5iCnU&list=PLpdmBGJ6ELULEfPWvvks0HtwzCvQo1zu0&index=5
# reference 2 - https://www.youtube.com/watch?v=9TxEQQyv9cE&list=PLpdmBGJ6ELULEfPWvvks0HtwzCvQo1zu0&index=8
# reference 3 - https://www.youtube.com/watch?v=MlK6SIjcjE8&t=322s

from llama_index import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext, load_index_from_storage, StorageContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.storage_context import SimpleVectorStore, SimpleIndexStore
import torch
from langchain.llms.base import LLM
from transformers import pipeline
from typing import Mapping, Any
import streamlit as st

st.title('Mahabharata-GPT')
prompt = st.text_input('Ask your question here')

@st.cache
class CustomLLM(LLM):
    model_name = "google/flan-t5-xl"
    device = "cpu"
    model_kwargs = {"device": device}

    pipeline = pipeline("text2text-generation", model=model_name, **model_kwargs)

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"

@st.cache
def return_index():

    llm_predictor = LLMPredictor(llm=CustomLLM())
    hfemb = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(hfemb)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="index_saved"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="index_saved"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="index_saved"),
    )

    index = load_index_from_storage(storage_context, service_context=service_context)

    return index

index = return_index()

query_engine = index.as_query_engine()

if prompt:
    response = query_engine.query(prompt)
    st.write(response)

#llm_predictor = LLMPredictor(llm=CustomLLM())

#hfemb = HuggingFaceEmbeddings()
#embed_model = LangchainEmbedding(hfemb)

#service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

#storage_context = StorageContext.from_defaults(
#    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="index_saved"),
#    vector_store=SimpleVectorStore.from_persist_dir(persist_dir="index_saved"),
#    index_store=SimpleIndexStore.from_persist_dir(persist_dir="index_saved"),
#)

#index = load_index_from_storage(storage_context, service_context=service_context)

#index = return_index()

#query_engine = index.as_query_engine()

#if prompt:
#    response = query_engine.query(prompt)
#    st.write(response)
