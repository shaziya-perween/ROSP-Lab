class ContextualQueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query):
        return [query, f"More about {query}"]
