#!/usr/bin/env python3
"""
Sentence Transformer Embeddings Manager - M3 Max Optimized
High-performance embedding generation for document similarity and clustering
"""

import asyncio
import logging
import numpy as np
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import semantic_search, cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ SentenceTransformers not available")

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import faiss
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import umap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel(Enum):
    """Available sentence transformer models"""
    # General purpose models
    ALL_MPNET_BASE = "all-mpnet-base-v2"              # Best quality, slower
    ALL_MINILM_L12 = "all-MiniLM-L12-v2"             # Good balance
    ALL_MINILM_L6 = "all-MiniLM-L6-v2"               # Fast, smaller
    
    # Specialized models
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large"   # Multilingual
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"          # High performance
    BGE_BASE_EN = "BAAI/bge-base-en-v1.5"            # Balanced
    BGE_SMALL_EN = "BAAI/bge-small-en-v1.5"          # Fast
    
    # Technical/Scientific models  
    SCIBERT = "allenai/scibert_scivocab_uncased"      # Scientific text
    SPECTER = "allenai/specter2_base"                 # Scientific papers

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: EmbeddingModel
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    cache_dir: Optional[str] = None

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embeddings: np.ndarray
    texts: List[str]
    model_name: str
    generation_time: float
    metadata: Dict[str, Any]

class EmbeddingCache:
    """High-performance embedding cache with persistence"""
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model"""
        combined = f"{model_name}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        cache_key = self._get_cache_key(text, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file)
                self.memory_cache[cache_key] = embedding  # Add to memory cache
                self.cache_stats["hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text, model_name)
        
        # Store in memory cache
        self.memory_cache[cache_key] = embedding
        
        # Store in disk cache asynchronously
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
            self.cache_stats["saves"] += 1
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "memory_cached_items": len(self.memory_cache),
            "disk_cached_files": len(list(self.cache_dir.glob("*.npy"))),
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            **self.cache_stats
        }
    
    def clear_memory_cache(self):
        """Clear memory cache to free RAM"""
        self.memory_cache.clear()
        logger.info("🗑️ Cleared embedding memory cache")

class SentenceTransformerManager:
    """High-performance sentence transformer manager for M3 Max"""
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.loaded_models = {}
        self.cache = EmbeddingCache(cache_dir)
        self.device = self._get_optimal_device()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.stats = {
            "embeddings_generated": 0,
            "total_generation_time": 0.0,
            "cache_hits": 0,
            "models_loaded": 0
        }
        
        logger.info(f"🚀 SentenceTransformerManager initialized on device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return "cpu"
        
        # For M3 Max, check MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    async def load_model(self, config: EmbeddingConfig) -> bool:
        """Load a sentence transformer model"""
        model_name = config.model.value
        
        if model_name in self.loaded_models:
            logger.info(f"✅ Model {model_name} already loaded")
            return True
        
        try:
            start_time = time.time()
            
            # Load model in thread pool to avoid blocking
            model = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._load_model_sync,
                config
            )
            
            if model is None:
                return False
            
            load_time = time.time() - start_time
            
            self.loaded_models[model_name] = {
                'model': model,
                'config': config,
                'load_time': load_time,
                'last_used': time.time()
            }
            
            self.stats["models_loaded"] += 1
            
            logger.info(f"✅ Loaded {model_name} in {load_time:.2f}s on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            return False
    
    def _load_model_sync(self, config: EmbeddingConfig) -> Optional[SentenceTransformer]:
        """Synchronous model loading"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ SentenceTransformers not available")
            return None
        
        try:
            model_name = config.model.value
            
            # Configure device
            device = config.device if config.device != "auto" else self.device
            
            # Load model with configuration
            model = SentenceTransformer(
                model_name,
                cache_folder=config.cache_dir,
                device=device
            )
            
            # Set max sequence length
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = config.max_seq_length
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {str(e)}")
            return None
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model_name: str,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Optional[EmbeddingResult]:
        """Generate embeddings for a list of texts"""
        
        if model_name not in self.loaded_models:
            logger.error(f"❌ Model {model_name} not loaded")
            return None
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        config = model_info['config']
        
        start_time = time.time()
        embeddings_list = []
        cached_count = 0
        
        # Determine batch size
        effective_batch_size = batch_size or config.batch_size
        
        logger.info(f"🎯 Generating embeddings for {len(texts)} texts using {model_name}")
        
        # Process in batches
        for i in range(0, len(texts), effective_batch_size):
            batch_texts = texts[i:i + effective_batch_size]
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text
            for j, text in enumerate(batch_texts):
                if use_cache:
                    cached_embedding = self.cache.get(text, model_name)
                    if cached_embedding is not None:
                        batch_embeddings.append(cached_embedding)
                        cached_count += 1
                        continue
                
                # Add to uncached list
                uncached_texts.append(text)
                uncached_indices.append(j)
                batch_embeddings.append(None)  # Placeholder
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    new_embeddings = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._generate_embeddings_sync,
                        model,
                        uncached_texts,
                        config.normalize_embeddings
                    )
                    
                    # Fill in the uncached embeddings and cache them
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings[idx] = embedding
                        if use_cache:
                            self.cache.set(batch_texts[idx], model_name, embedding)
                            
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed for batch {i//effective_batch_size + 1}: {str(e)}")
                    return None
            
            embeddings_list.extend(batch_embeddings)
            
            if show_progress and len(texts) > effective_batch_size:
                progress = min(i + effective_batch_size, len(texts))
                logger.info(f"📈 Progress: {progress}/{len(texts)} texts processed")
        
        # Convert to numpy array
        embeddings = np.vstack(embeddings_list)
        
        generation_time = time.time() - start_time
        
        # Update statistics
        self.stats["embeddings_generated"] += len(texts)
        self.stats["total_generation_time"] += generation_time
        self.stats["cache_hits"] += cached_count
        model_info['last_used'] = time.time()
        
        logger.info(f"✅ Generated {len(texts)} embeddings in {generation_time:.2f}s (Cache hits: {cached_count})")
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model_name=model_name,
            generation_time=generation_time,
            metadata={
                'cache_hits': cached_count,
                'device': self.device,
                'batch_size': effective_batch_size,
                'embedding_dim': embeddings.shape[1]
            }
        )
    
    def _generate_embeddings_sync(
        self,
        model: SentenceTransformer,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """Synchronous embedding generation"""
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings
    
    async def find_similar(
        self,
        query_texts: List[str],
        corpus_texts: List[str],
        model_name: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find similar texts using semantic search"""
        
        # Generate embeddings for query and corpus
        query_result = await self.generate_embeddings(query_texts, model_name)
        corpus_result = await self.generate_embeddings(corpus_texts, model_name)
        
        if not query_result or not corpus_result:
            logger.error("❌ Failed to generate embeddings for similarity search")
            return {}
        
        # Perform semantic search
        results = {}
        
        for i, query_text in enumerate(query_texts):
            query_embedding = query_result.embeddings[i:i+1]
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_embedding, corpus_result.embeddings)[0]
            
            # Get top-k results above threshold
            similar_indices = np.argsort(similarities)[::-1]
            
            similar_results = []
            for idx in similar_indices[:top_k]:
                score = float(similarities[idx])
                if score >= threshold:
                    similar_results.append({
                        'text': corpus_texts[idx],
                        'score': score,
                        'index': int(idx)
                    })
            
            results[query_text] = similar_results
        
        return results
    
    async def cluster_texts(
        self,
        texts: List[str],
        model_name: str,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Cluster texts using embeddings"""
        
        # Generate embeddings
        result = await self.generate_embeddings(texts, model_name)
        if not result:
            logger.error("❌ Failed to generate embeddings for clustering")
            return {}
        
        embeddings = result.embeddings
        
        # Perform clustering
        if method == "kmeans":
            n_clusters = n_clusters or min(10, len(texts) // 2)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 2))
        else:
            logger.error(f"❌ Unknown clustering method: {method}")
            return {}
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Organize results by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append({
                'text': texts[i],
                'index': i,
                'embedding': embeddings[i].tolist()
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, items in clusters.items():
            if len(items) > 1:
                cluster_embeddings = np.array([item['embedding'] for item in items])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate average distance to centroid
                distances = [
                    float(np.linalg.norm(np.array(item['embedding']) - centroid))
                    for item in items
                ]
                
                cluster_stats[cluster_id] = {
                    'size': len(items),
                    'centroid': centroid.tolist(),
                    'avg_distance': float(np.mean(distances)),
                    'coherence': float(1.0 / (1.0 + np.mean(distances)))  # Higher = more coherent
                }
        
        return {
            'clusters': clusters,
            'statistics': cluster_stats,
            'method': method,
            'total_clusters': len(clusters),
            'total_texts': len(texts),
            'clustering_time': result.generation_time
        }
    
    async def reduce_dimensionality(
        self,
        texts: List[str],
        model_name: str,
        method: str = "umap",
        n_components: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """Reduce dimensionality of embeddings for visualization"""
        
        # Generate embeddings
        result = await self.generate_embeddings(texts, model_name)
        if not result:
            logger.error("❌ Failed to generate embeddings for dimensionality reduction")
            return {}
        
        embeddings = result.embeddings
        
        # Perform dimensionality reduction
        if method == "umap":
            reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
        else:
            logger.error(f"❌ Unknown dimensionality reduction method: {method}")
            return {}
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        return {
            'reduced_embeddings': reduced_embeddings.tolist(),
            'texts': texts,
            'method': method,
            'n_components': n_components,
            'original_dim': embeddings.shape[1],
            'generation_time': result.generation_time
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        memory = psutil.virtual_memory()
        
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'device': self.device,
            'cache_stats': self.cache.get_stats(),
            'performance_stats': {
                'embeddings_generated': self.stats["embeddings_generated"],
                'avg_generation_time': (
                    self.stats["total_generation_time"] / max(1, self.stats["embeddings_generated"])
                ),
                'cache_hit_rate': (
                    self.stats["cache_hits"] / max(1, self.stats["embeddings_generated"])
                ),
                'models_loaded': self.stats["models_loaded"]
            },
            'memory_status': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent
            },
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'mlx_available': MLX_AVAILABLE
        }
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        if model_name not in self.loaded_models:
            logger.warning(f"⚠️ Model {model_name} not loaded")
            return False
        
        try:
            del self.loaded_models[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if using CUDA/MPS
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            logger.info(f"🗑️ Unloaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unload {model_name}: {str(e)}")
            return False
    
    async def cleanup_cache(self, max_memory_items: int = 1000):
        """Cleanup cache to free memory"""
        if len(self.cache.memory_cache) > max_memory_items:
            self.cache.clear_memory_cache()
            logger.info(f"🧹 Cleared embedding cache (was {len(self.cache.memory_cache)} items)")

# Utility functions for common embedding tasks
async def generate_document_embeddings(
    documents: List[str],
    model: EmbeddingModel = EmbeddingModel.BGE_BASE_EN,
    batch_size: int = 32
) -> Optional[EmbeddingResult]:
    """Quick utility to generate document embeddings"""
    manager = SentenceTransformerManager()
    
    config = EmbeddingConfig(
        model=model,
        batch_size=batch_size,
        max_seq_length=512,
        normalize_embeddings=True
    )
    
    success = await manager.load_model(config)
    if not success:
        return None
    
    return await manager.generate_embeddings(documents, model.value)

async def find_duplicate_documents(
    documents: List[str],
    similarity_threshold: float = 0.9,
    model: EmbeddingModel = EmbeddingModel.BGE_BASE_EN
) -> List[Tuple[int, int, float]]:
    """Find potential duplicate documents using embeddings"""
    manager = SentenceTransformerManager()
    
    config = EmbeddingConfig(model=model)
    await manager.load_model(config)
    
    result = await manager.generate_embeddings(documents, model.value)
    if not result:
        return []
    
    embeddings = result.embeddings
    similarities = cosine_similarity(embeddings)
    
    duplicates = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            similarity = similarities[i][j]
            if similarity >= similarity_threshold:
                duplicates.append((i, j, float(similarity)))
    
    return sorted(duplicates, key=lambda x: x[2], reverse=True)

# Example usage
async def main():
    """Example usage of SentenceTransformerManager"""
    manager = SentenceTransformerManager()
    
    # Print system status
    status = manager.get_system_status()
    print("🖥️ Embedding System Status:")
    print(json.dumps(status, indent=2))
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        # Load a model
        config = EmbeddingConfig(
            model=EmbeddingModel.BGE_BASE_EN,
            batch_size=32,
            max_seq_length=512
        )
        
        success = await manager.load_model(config)
        if success:
            # Test embeddings
            texts = [
                "5G NR technology provides enhanced mobile broadband",
                "LTE networks support high-speed data transmission",
                "Network slicing enables customized service delivery",
                "Machine learning improves network optimization"
            ]
            
            result = await manager.generate_embeddings(texts, config.model.value)
            if result:
                print(f"📊 Generated {result.embeddings.shape[0]} embeddings")
                print(f"   Embedding dimension: {result.embeddings.shape[1]}")
                print(f"   Generation time: {result.generation_time:.2f}s")
                
                # Test similarity search
                similar = await manager.find_similar(
                    query_texts=["What is 5G technology?"],
                    corpus_texts=texts,
                    model_name=config.model.value,
                    top_k=2
                )
                
                print("🔍 Similarity Search Results:")
                for query, results in similar.items():
                    print(f"Query: {query}")
                    for result in results:
                        print(f"  - {result['text']} (score: {result['score']:.3f})")

if __name__ == "__main__":
    asyncio.run(main())