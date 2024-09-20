from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

class QdrantClientManager:
    def __init__(self, collection_name: str, qdrant_url: str = "http://localhost:6333"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    def encode_query(self, query: str):
        # Encode the query sentence into a vector
        return self.model.encode(query).tolist()  # Convert numpy array to list

    def search(self, query_sentence: str, limit: int = 10, score_threshold: float = 0.7, label_filter: int = None):
        # Encode the query sentence
        query_vector = self.encode_query(query_sentence)

        # Prepare the query filter if a label filter is provided
        query_filter = None
        if label_filter is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="label",
                        match=MatchValue(value=label_filter)  # Use MatchValue
                    )
                ]
            )

        # Perform the search query
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            print(f"Search Error: {e}")
            return []

    def recommend(self, positive_sentences: list, limit: int = 10, score_threshold: float = 0.7):
        # Encode positive sentences into vectors
        positive_vectors = [self.encode_query(sentence) for sentence in positive_sentences]

        # Perform the recommendation query
        try:
            recommendations = self.client.recommend(
                collection_name=self.collection_name,
                positive=positive_vectors,  # Directly use the vectors
                limit=limit,
                score_threshold=score_threshold
            )
            return recommendations
        except Exception as e:
            print(f"Recommendation Error: {e}")
            return []

if __name__ == "__main__":
    qdrant_manager = QdrantClientManager(collection_name="my_collection")

    # Perform a search
    search_results = qdrant_manager.search("Positive sentiment", limit=5, score_threshold=0.6, label_filter=1)
    print("Search Results:")
    if search_results:
        for result in search_results:
            print(f"ID: {result.id}, Distance: {result.score}, Payload: {result.payload}")
    else:
        print("No results found or an error occurred.")

    # Example positive sentences for recommendations
    positive_sentences = ["perfect", "glory", "modern"]

    # Perform a recommendation
    recommendations = qdrant_manager.recommend(positive_sentences, limit=5, score_threshold=0.6)
    print("\nRecommendations:")
    if recommendations:
        for recommendation in recommendations:
            print(f"ID: {recommendation.id}, Score: {recommendation.score}")
    else:
        print("No recommendations found or an error occurred.")
