from qdrant_client import QdrantClient

def create_collection():
    client = QdrantClient(host="localhost", port=6333)

    collection_name = "my_collection"
    vector_size = 384 
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": vector_size,
                "distance": "Cosine"
            }
        )
        print(f"Koleksiyon '{collection_name}' başarıyla oluşturuldu.")
    except Exception as e:
        print("Koleksiyon oluşturma hatası:", e)

if __name__ == "__main__":
    create_collection()
