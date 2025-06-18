from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import base64

# Your existing imports
from embed import (get_embeddings, generate_answer, describe_base64_image)
from config import  OPEN_API_KEY
from helper import (load_embeddings, extract_europe1_urls, image_url_to_base64)
from get_answer import find_similar_content, parse_llm_response

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class QueryResponse(BaseModel):
    answer: str
    links: list

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="Simple API for querying with optional images")

# Load embeddings once at startup
discourse_embeddings = load_embeddings('discourse_embeddings.npz')
markdown_embeddings = load_embeddings('markdown_embeddings.npz')

async def process_query(question: str, image_base64: Optional[str] = None):
    """
    Modified version of your result_response function
    """
    try:
        CONTEXT = 10
        
        # Handle image processing
        if image_base64:
            # If image is provided directly as base64
            img_description = await describe_base64_image(image_base64, OPEN_API_KEY, 3, question=question)
            question = question + " " + img_description
        else:
            # Try to extract URLs from question if no direct image provided
            try:
                url = extract_europe1_urls(question)
                if url:
                    base64_img = image_url_to_base64(url[0])
                    img_description = await describe_base64_image(base64_img, OPEN_API_KEY, 3, question=question)
                    question = question + " " + img_description
            except:
                # Continue without image if extraction fails
                pass
        
        # Generate embeddings
        embedding_response = await get_embeddings(question, OPEN_API_KEY)
        
        # Find relevant content
        relevant_results = find_similar_content(embedding_response, CONTEXT, discourse_embeddings, markdown_embeddings)
        
        # Generate answer
        answer = await generate_answer(OPEN_API_KEY, question, relevant_results)
        llm_response = parse_llm_response(answer)
        # print(answer, '\n\n\n')
        # # llm_response = parse_llm_response(answer)
        # with open('answer.json', 'w') as f:
        #     json.dump(answer, f, indent = 4)
        # print(llm_response)
        # with open('response.json', 'w') as f:
        #     json.dump(llm_response, f, indent = 4)
        
        return llm_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint for processing queries with optional images
    """
    try:
        result = await process_query(request.question, request.image)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
