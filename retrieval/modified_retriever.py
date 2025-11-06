class ContextAwareRetriever:
    def __init__(self, index, embeddings, expander):
        self.index = index
        self.embeddings = embeddings
        self.expander = expander

    def get_relevant_documents(self, query):
        expanded_queries = self.expander.expand_query(query)
        results = []
        for q in expanded_queries:
            q_vec = self.embeddings.embed_query(q)
            matches = self.index.query(vector=q_vec, top_k=2, include_metadata=True)
            results.extend([m["metadata"]["text"] for m in matches["matches"]])
        return results
