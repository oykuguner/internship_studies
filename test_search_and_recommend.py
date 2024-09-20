from qdrant_client import QdrantClient

class QdrantTest:
    def __init__(self, collection_name, qdrant_url="http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    def perform_search(self, query_sentence, limit=5, score_threshold=0.6, label_filter=None):
        #...
        pass

    def perform_recommend(self, query_sentence, limit=5, score_threshold=0.6):
        #...
        pass

if __name__ == "__main__":
    qdrant_test = QdrantTest(collection_name="my_collection")
    qdrant_test.perform_search("Positive sentiment", limit=5, score_threshold=0.6, label_filter=1)
    qdrant_test.perform_recommend("Positive sentiment", limit=5, score_threshold=0.6)
