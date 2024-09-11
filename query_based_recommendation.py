from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import LabelEncoder
from annoy import AnnoyIndex

# Load the dataset
data = pd.read_csv('/path/to/your/amazon.csv')

# Initialize FastAPI
app = FastAPI()

# NLP Preprocessing
def preprocess_text(text):
    text = text.lower()
    return text

data['product_name'] = data['product_name'].apply(preprocess_text)
data['category'] = data['category'].apply(preprocess_text)
data['about_product'] = data['about_product'].apply(preprocess_text)

# Convert text to sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data['product_name'].tolist() + data['category'].tolist() + data['about_product'].tolist())

data['product_name_seq'] = tokenizer.texts_to_sequences(data['product_name'])
data['category_seq'] = tokenizer.texts_to_sequences(data['category'])
data['about_product_seq'] = tokenizer.texts_to_sequences(data['about_product'])

# Define Embedding layer
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50  # Adjust as needed
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=None)

# Function to get embeddings
def get_embeddings(sequence):
    sequence = tf.constant(sequence, dtype=tf.int32)
    return tf.reduce_mean(embedding_layer(sequence), axis=0).numpy()

# Apply embedding to each column
data['product_name_emb'] = data['product_name_seq'].apply(get_embeddings)
data['category_emb'] = data['category_seq'].apply(get_embeddings)
data['about_product_emb'] = data['about_product_seq'].apply(get_embeddings)

# Combine embeddings into a single vector for each product
data['combined_emb'] = data.apply(lambda row: np.mean([
    row['product_name_emb'],
    row['category_emb'],
    row['about_product_emb']
], axis=0), axis=1)

# Build the Annoy Index
dimension = embedding_dim
annoy_index = AnnoyIndex(dimension, 'angular')

for i, embedding in enumerate(data['combined_emb']):
    annoy_index.add_item(i, embedding)

annoy_index.build(n_trees=10)

# FastAPI models
class QueryModel(BaseModel):
    query: str
    k: int = 20

# Endpoint to get similar products
@app.post("/recommend/")
def recommend_products(query_model: QueryModel):
    try:
        query = preprocess_text(query_model.query)
        query_seq = tokenizer.texts_to_sequences([query])
        query_emb = get_embeddings(query_seq[0])
        similar_indices = annoy_index.get_nns_by_vector(query_emb, query_model.k)
        similar_products = data.iloc[similar_indices]
        return similar_products[['product_id', 'product_name', 'category', 'about_product']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
