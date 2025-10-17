"""
Vector Memory Module for Ultravox WebSocket API

This module provides semantic memory storage and retrieval capabilities using vector databases.
It's designed for real-time voice AI applications with minimal latency impact.

Features:
- Text embedding generation using Hugging Face transformers
- Vector database operations (ChromaDB for local, Pinecone for cloud)
- Semantic search and memory recall
- Session-based memory management
- Caching for ultra-low latency
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
import hashlib

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class MemoryFact:
    """Represents a semantic memory fact"""
    id: str
    session_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SearchResult:
    """Represents a semantic search result"""
    fact: MemoryFact
    similarity_score: float
    distance: float

class VectorMemoryManager:
    """Manages semantic memory using vector databases"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 db_type: str = "chroma",  # "chroma" or "pinecone"
                 cache_size: int = 1000,
                 similarity_threshold: float = 0.7,
                 max_memories_per_session: int = 100):
        """
        Initialize the vector memory manager
        
        Args:
            model_name: Hugging Face model for embeddings
            db_type: Type of vector database ("chroma" or "pinecone")
            cache_size: Maximum number of memories to cache in memory
            similarity_threshold: Minimum similarity score for relevant memories
            max_memories_per_session: Maximum memories per session before cleanup
        """
        self.model_name = model_name
        self.db_type = db_type
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.max_memories_per_session = max_memories_per_session
        
        # Initialize embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Initialize vector database
        self._init_vector_db()
        
        # In-memory cache for ultra-fast access
        self.memory_cache: Dict[str, List[MemoryFact]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Performance metrics
        self.metrics = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0,
            "total_embeddings": 0,
            "avg_embedding_time": 0.0
        }
        
        logger.info(f"VectorMemoryManager initialized with {db_type} database")
    
    def _init_vector_db(self):
        """Initialize the vector database"""
        if self.db_type == "chroma":
            # Use ChromaDB for local storage
            self.chroma_client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection("ultravox_memories")
                logger.info("Connected to existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="ultravox_memories",
                    metadata={"description": "Ultravox semantic memories"}
                )
                logger.info("Created new ChromaDB collection")
                
        elif self.db_type == "pinecone":
            # Pinecone setup (requires API key)
            try:
                import pinecone
                pinecone.init(api_key="your-pinecone-api-key", environment="your-environment")
                self.index = pinecone.Index("ultravox-memories")
                logger.info("Connected to Pinecone index")
            except ImportError:
                logger.error("Pinecone not installed. Install with: pip install pinecone-client")
                raise
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the transformer model"""
        start_time = time.time()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            self.metrics["cache_hits"] += 1
            return self.embedding_cache[text_hash]
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings)
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embeddings
            if len(self.embedding_cache) > self.cache_size:
                # Remove oldest entries
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            # Update metrics
            embedding_time = time.time() - start_time
            self.metrics["total_embeddings"] += 1
            self.metrics["avg_embedding_time"] = (
                (self.metrics["avg_embedding_time"] * (self.metrics["total_embeddings"] - 1) + embedding_time) 
                / self.metrics["total_embeddings"]
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default size for all-MiniLM-L6-v2
    
    async def store_memory(self, session_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store a semantic memory for a session"""
        try:
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Create memory fact
            memory_id = str(uuid.uuid4())
            fact = MemoryFact(
                id=memory_id,
                session_id=session_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )
            
            # Store in vector database
            if self.db_type == "chroma":
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[{
                        "session_id": session_id,
                        "timestamp": fact.timestamp.isoformat(),
                        **fact.metadata
                    }]
                )
            elif self.db_type == "pinecone":
                self.index.upsert([(
                    memory_id,
                    embedding.tolist(),
                    {
                        "session_id": session_id,
                        "content": content,
                        "timestamp": fact.timestamp.isoformat(),
                        **fact.metadata
                    }
                )])
            
            # Add to cache
            if session_id not in self.memory_cache:
                self.memory_cache[session_id] = []
            self.memory_cache[session_id].append(fact)
            
            # Cleanup old memories for this session
            await self._cleanup_session_memories(session_id)
            
            logger.info(f"Stored memory for session {session_id}: {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return None
    
    async def search_memories(self, 
                            session_id: str, 
                            query: str, 
                            top_k: int = 5,
                            include_other_sessions: bool = False) -> List[SearchResult]:
        """Search for semantically similar memories"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search in vector database
            if self.db_type == "chroma":
                # Build filter for session-specific search
                where_filter = None if include_other_sessions else {"session_id": session_id}
                
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    where=where_filter
                )
                
                # Convert results to SearchResult objects
                search_results = []
                if results['ids'] and results['ids'][0]:
                    for i, (memory_id, distance, document, metadata) in enumerate(zip(
                        results['ids'][0],
                        results['distances'][0],
                        results['documents'][0],
                        results['metadatas'][0]
                    )):
                        # Convert distance to similarity score (ChromaDB uses cosine distance)
                        similarity = 1 - distance
                        
                        if similarity >= self.similarity_threshold:
                            fact = MemoryFact(
                                id=memory_id,
                                session_id=metadata.get('session_id', 'unknown'),
                                content=document,
                                metadata=metadata
                            )
                            search_results.append(SearchResult(
                                fact=fact,
                                similarity_score=similarity,
                                distance=distance
                            ))
            
            elif self.db_type == "pinecone":
                # Pinecone search
                search_filter = None if include_other_sessions else {"session_id": session_id}
                
                results = self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                    filter=search_filter
                )
                
                search_results = []
                for match in results.matches:
                    similarity = match.score
                    if similarity >= self.similarity_threshold:
                        metadata = match.metadata
                        fact = MemoryFact(
                            id=match.id,
                            session_id=metadata.get('session_id', 'unknown'),
                            content=metadata.get('content', ''),
                            metadata=metadata
                        )
                        search_results.append(SearchResult(
                            fact=fact,
                            similarity_score=similarity,
                            distance=1 - similarity
                        ))
            
            # Update metrics
            search_time = time.time() - start_time
            self.metrics["total_searches"] += 1
            self.metrics["avg_search_time"] = (
                (self.metrics["avg_search_time"] * (self.metrics["total_searches"] - 1) + search_time) 
                / self.metrics["total_searches"]
            )
            
            logger.info(f"Found {len(search_results)} relevant memories for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def get_session_context(self, session_id: str, max_memories: int = 10) -> List[MemoryFact]:
        """Get recent memories for a session as context"""
        try:
            if session_id in self.memory_cache:
                # Return most recent memories from cache
                recent_memories = sorted(
                    self.memory_cache[session_id], 
                    key=lambda x: x.timestamp, 
                    reverse=True
                )[:max_memories]
                return recent_memories
            
            # Fallback to database query
            if self.db_type == "chroma":
                results = self.collection.query(
                    query_embeddings=[[0.0] * 384],  # Dummy embedding
                    n_results=max_memories,
                    where={"session_id": session_id}
                )
                
                memories = []
                if results['ids'] and results['ids'][0]:
                    for memory_id, document, metadata in zip(
                        results['ids'][0],
                        results['documents'][0],
                        results['metadatas'][0]
                    ):
                        memories.append(MemoryFact(
                            id=memory_id,
                            session_id=session_id,
                            content=document,
                            metadata=metadata
                        ))
                return memories
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
            return []
    
    async def _cleanup_session_memories(self, session_id: str):
        """Clean up old memories for a session to prevent memory bloat"""
        try:
            if session_id not in self.memory_cache:
                return
            
            memories = self.memory_cache[session_id]
            if len(memories) > self.max_memories_per_session:
                # Keep only the most recent memories
                memories.sort(key=lambda x: x.timestamp, reverse=True)
                memories_to_keep = memories[:self.max_memories_per_session]
                memories_to_remove = memories[self.max_memories_per_session:]
                
                # Remove from database
                if memories_to_remove:
                    ids_to_remove = [m.id for m in memories_to_remove]
                    if self.db_type == "chroma":
                        self.collection.delete(ids=ids_to_remove)
                    elif self.db_type == "pinecone":
                        self.index.delete(ids=ids_to_remove)
                
                # Update cache
                self.memory_cache[session_id] = memories_to_keep
                
                logger.info(f"Cleaned up {len(memories_to_remove)} old memories for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up session memories: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "cache_size": len(self.embedding_cache),
            "sessions_cached": len(self.memory_cache),
            "total_memories_cached": sum(len(memories) for memories in self.memory_cache.values())
        }
    
    def clear_session_memories(self, session_id: str):
        """Clear all memories for a session"""
        try:
            # Remove from cache
            if session_id in self.memory_cache:
                del self.memory_cache[session_id]
            
            # Remove from database
            if self.db_type == "chroma":
                # ChromaDB doesn't have a direct delete by filter, so we need to query first
                results = self.collection.query(
                    query_embeddings=[[0.0] * 384],
                    n_results=1000,  # Large number to get all
                    where={"session_id": session_id}
                )
                if results['ids'] and results['ids'][0]:
                    self.collection.delete(ids=results['ids'][0])
            elif self.db_type == "pinecone":
                self.index.delete(filter={"session_id": session_id})
            
            logger.info(f"Cleared all memories for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing session memories: {e}")

# Global instance
vector_memory = None

def initialize_vector_memory(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                           db_type: str = "chroma",
                           cache_size: int = 1000) -> VectorMemoryManager:
    """Initialize the global vector memory instance"""
    global vector_memory
    if vector_memory is None:
        vector_memory = VectorMemoryManager(
            model_name=model_name,
            db_type=db_type,
            cache_size=cache_size
        )
    return vector_memory

def get_vector_memory() -> Optional[VectorMemoryManager]:
    """Get the global vector memory instance"""
    return vector_memory

# Convenience functions for easy integration
async def store_conversation_fact(session_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
    """Store a conversation fact in vector memory"""
    if vector_memory is None:
        logger.warning("Vector memory not initialized")
        return None
    return await vector_memory.store_memory(session_id, content, metadata)

async def search_conversation_memories(session_id: str, query: str, top_k: int = 5) -> List[SearchResult]:
    """Search for relevant conversation memories"""
    if vector_memory is None:
        logger.warning("Vector memory not initialized")
        return []
    return await vector_memory.search_memories(session_id, query, top_k)

async def get_conversation_context(session_id: str, max_memories: int = 10) -> List[MemoryFact]:
    """Get conversation context for a session"""
    if vector_memory is None:
        logger.warning("Vector memory not initialized")
        return []
    return await vector_memory.get_session_context(session_id, max_memories)

def format_memories_for_prompt(memories: List[MemoryFact], max_length: int = 500) -> str:
    """Format memories for inclusion in Ultravox prompts"""
    if not memories:
        return ""
    
    formatted_memories = []
    current_length = 0
    
    for memory in memories:
        memory_text = f"- {memory.content}"
        if current_length + len(memory_text) > max_length:
            break
        formatted_memories.append(memory_text)
        current_length += len(memory_text)
    
    if formatted_memories:
        return "Relevant context from previous conversation:\n" + "\n".join(formatted_memories)
    return ""
