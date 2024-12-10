import numpy as np
import pandas as pd
import os, json
from openai import AzureOpenAI

client: AzureOpenAI = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        api_version = "2024-02-01",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-3-small"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_docs(df, user_query, top_n=4, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-3-small"
    )
    df["similarities"] = df.readmeVector.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        pd.set_option('display.max_colwidth', None)
        print(res[['title', 'source']])
    return res

df = pd.read_json('workloads/workloads_with_vectors.json')

res = search_docs(df, "I want to build a chatbot using OpenAI", top_n=4)