import asyncio
import aiohttp
import json
import time
from typing import List, Optional, Tuple
import numpy as np
from semantic_text_splitter import MarkdownSplitter
from tqdm.asyncio import tqdm
import logging
from helper import AsyncRateLimiter  
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_embeddings_async(
    session: aiohttp.ClientSession,
    text: str,
    api_key: str,
    rate_limiter: AsyncRateLimiter,
    model: str = "gemini-embedding-exp-03-07",
    max_retries: int = 3
) -> Optional[List[float]]:
    """Async version of embedding generation with improved error handling"""
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.wait()
            
            # Using Google Gemini API (adapt URL as needed)
            url = "https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            
            payload = {
                "model": f"models/{model}",
                "content": {
                    "parts": [{"text": text}]
                }
            }
            
            async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embedding"]["values"]
                
                elif response.status == 429:  # Rate limit
                    wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait_time)
                    
                elif response.status in [500, 502, 503, 504]:  # Server errors
                    wait_time = (2 ** attempt)
                    logger.warning(f"Server error {response.status}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    if attempt == max_retries - 1:
                        return None
                    await asyncio.sleep(1)
                        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(2 ** attempt)
    
    return None

def get_chunks_optimized(text: str, chunk_size: int = 8000, chunk_overlap: int = 100, max_embedding_chars: int = 8000) -> List[str]:
    """Optimized chunking with better memory management"""
    if not text or len(text.strip()) == 0:
        return []
    
    # Your existing chunking logic but with early returns for efficiency
    effective_chunk_size = min(chunk_size, max_embedding_chars)
    
    if len(text) <= effective_chunk_size:
        return [text.strip()]
    
    # Use generator for memory efficiency with large texts
    chunks = []
    # Clean up whitespace and newlines, preserving meaningful paragraph breaks
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
    text = text.strip()
    
    # Apply embedding limit constraint - ensure chunk_size doesn't exceed embedding limit
    effective_chunk_size = min(chunk_size, max_embedding_chars)
    
    # If text is very short, return it as a single chunk
    if len(text) <= effective_chunk_size:
        return [text]
    
    # Split text by paragraphs for more meaningful chunks
    paragraphs = text.split('\n')
    current_chunk = ""
    
    for i, para in enumerate(paragraphs):
        # If this paragraph alone exceeds chunk size, we need to split it further
        if len(para) > effective_chunk_size:
            # If we have content in the current chunk, store it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = ""
            
            for sentence in sentences:
                # If single sentence exceeds chunk size, split by chunks with overlap
                if len(sentence) > effective_chunk_size:
                    # If we have content in the sentence chunk, store it first
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                        sentence_chunk = ""
                    
                    # Process the long sentence in chunks with embedding limit consideration
                    for j in range(0, len(sentence), effective_chunk_size - chunk_overlap):
                        sentence_part = sentence[j:j + effective_chunk_size]
                        if sentence_part:
                            chunks.append(sentence_part.strip())
                
                # If adding this sentence would exceed chunk size, save current and start new
                elif len(sentence_chunk) + len(sentence) > effective_chunk_size and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                else:
                    # Add to current sentence chunk
                    if sentence_chunk:
                        sentence_chunk += " " + sentence
                    else:
                        sentence_chunk = sentence
            
            # Add any remaining sentence chunk
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
            
        # Normal paragraph handling - if adding would exceed chunk size
        elif len(current_chunk) + len(para) > effective_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with this paragraph
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Verify we have chunks and apply overlap between chunks
    if chunks:
        # Create new chunks list with proper overlap
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # If the previous chunk ends with a partial sentence, find where it begins
            if len(prev_chunk) > chunk_overlap:
                # Find a good breaking point for overlap
                overlap_start = max(0, len(prev_chunk) - chunk_overlap)
                # Try to find sentence boundary
                sentence_break = prev_chunk.rfind('. ', overlap_start)
                if sentence_break != -1 and sentence_break > overlap_start:
                    overlap = prev_chunk[sentence_break+2:]
                    if overlap and not current_chunk.startswith(overlap):
                        # Ensure the overlapped chunk doesn't exceed embedding limit
                        proposed_chunk = overlap + " " + current_chunk
                        if len(proposed_chunk) <= max_embedding_chars:
                            current_chunk = proposed_chunk
                        # If it would exceed limit, truncate the overlap
                        else:
                            available_space = max_embedding_chars - len(current_chunk) - 1
                            if available_space > 0:
                                truncated_overlap = overlap[:available_space]
                                current_chunk = truncated_overlap + " " + current_chunk
                
            overlapped_chunks.append(current_chunk)
        
        # Final validation: ensure no chunk exceeds embedding limit
        validated_chunks = []
        for chunk in overlapped_chunks:
            if len(chunk) <= max_embedding_chars:
                validated_chunks.append(chunk)
            else:
                # If a chunk still exceeds limit, split it further
                for j in range(0, len(chunk), max_embedding_chars - chunk_overlap):
                    subchunk = chunk[j:j + max_embedding_chars]
                    if subchunk:
                        validated_chunks.append(subchunk.strip())
        
        return validated_chunks
    
    return chunks
