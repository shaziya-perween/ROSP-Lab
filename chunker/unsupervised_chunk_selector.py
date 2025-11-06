class UnsupervisedChunkSelector:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters

    def select_chunks(self, chunks, vectors):
        # Just pick first few chunks for now
        return [(chunks[i], vectors[i]) for i in range(min(self.num_clusters, len(chunks)))]
