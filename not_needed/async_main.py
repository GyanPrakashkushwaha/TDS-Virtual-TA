import asyncio
import aiohttp
import json
import time
from typing import List, Optional, Tuple
import numpy as np
from semantic_text_splitter import MarkdownSplitter
from tqdm.asyncio import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int = 60, requests_per_second: int = 2):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0
        self._lock = asyncio.Lock()
    
    async def wait(self):
        async with self._lock:
            current_time = time.time()
            
            # Per-second rate limiting
            time_since_last = current_time - self.last_request_time
            if time_since_last < (1 / self.requests_per_second):
                sleep_time = (1 / self.requests_per_second) - time_since_last
                await asyncio.sleep(sleep_time)
            
            # Per-minute rate limiting
            current_time = time.time()
            self.request_times = [t for t in self.request_times if t > current_time - 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    current_time = time.time()
                    self.request_times = [t for t in self.request_times if t > current_time - 60]
            
            self.request_times.append(current_time)
            self.last_request_time = current_time

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

def get_chunks_optimized(text: str, chunk_size: int = 1000, chunk_overlap: int = 100, max_embedding_chars: int = 8000) -> List[str]:
    """Optimized chunking with better memory management"""
    if not text or len(text.strip()) == 0:
        return []
    
    # Your existing chunking logic but with early returns for efficiency
    effective_chunk_size = min(chunk_size, max_embedding_chars)
    
    if len(text) <= effective_chunk_size:
        return [text.strip()]
    
    # Use generator for memory efficiency with large texts
    chunks = []
    # ... (your existing chunking logic from embed.txt)
    
    return chunks
