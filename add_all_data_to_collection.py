import pyarrow.parquet as pq
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np
import time

parquet_file_path = r'C:\Users\Öykü Güner\OneDrive\Masaüstü\python.py\task\imdb_single_sentence.parquet'
collection_name = 'my_collection'
embedding_model_name = 'all-MiniLM-L6-v2'
batch_size = 128

model = SentenceTransformer(embedding_model_name)

client = QdrantClient(host='localhost', port=6333)

def read_parquet_in_batches(file_path, batch_size):

    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield pd.DataFrame(batch.to_pandas())

def add_all_data_to_collection():

    total_vectors = 0

    for df in read_parquet_in_batches(parquet_file_path, batch_size):
        try:
            sentences = df['text'].tolist()
            vectors = model.encode(
                sentences=sentences,
                batch_size=batch_size,
                normalize_embeddings=True
            ).tolist()

            print(f"Veriler vektörleştirildi. Toplam: {len(vectors)} vektör.")

            
            points = [
                {
                    'id': int(i + total_vectors),
                    'vector': vectors[i],
                    'payload': {
                        'label': int(df.iloc[i]['label'])
                    }
                }
                for i in range(len(sentences))
            ]

            client.upsert(
                collection_name=collection_name,
                points=points
            )

            total_vectors += len(sentences) 
            print(f"{len(sentences)} adet veri eklendi. Toplam eklenen veri: {total_vectors}")

        except Exception as e:
            print(f"Hata oluştu: {e}")
            break  

        time.sleep(1)

if __name__ == "__main__":
    add_all_data_to_collection()

