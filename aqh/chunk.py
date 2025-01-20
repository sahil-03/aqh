import tiktoken 
from openai import OpenAI

class Chunker: 
    def __init__(
        self, 
        filepath: str, 
        model_name: str = "text-embedding-ada-002",
        chunk_size: int = 1024,
        chunk_overlap: int = 128
    ):
        self.filepath = filepath
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = OpenAI()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _read_file(self) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    
    def _tokenize_text(self, text: str) -> list:
        return self.tokenizer.encode(text)
    
    def _detokenize_text(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)
    
    def chunk_text(self, text: str) -> list:
        tokens = self._tokenize_text(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._detokenize_text(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap
            if start < 0:
                break
        
        return chunks
    
    def embed_chunk(self, chunk: str) -> list:
        response = self.client.embeddings.create(
            input=chunk,
            model=self.model_name
        )
        embedding_vector = response.data[0].embedding
        return embedding_vector
    
    def process(self) -> list:
        text = self._read_file()
        chunks = self.chunk_text(text)

        embedded_chunks = []
        for idx, chunk in enumerate(chunks):
            embedding_vector = self.embed_chunk(chunk)
            embedded_chunks.append({
                "chunk_index": idx,
                "chunk": chunk,
                "embedding": embedding_vector
            })
        
        return embedded_chunks

