from embed import (get_embeddings, 
                   generate_answer,
                   describe_base64_image)
from config import GEMINI_API_KEY, OPEN_API_KEY
from helper import (load_embeddings, 
                    extract_europe1_urls,
                    image_url_to_base64)
from get_answer import find_similar_content, parse_llm_response
import asyncio
import json


async def result_response(question):
    try:
        # url = extract_europe1_urls(question)
        # base64_img = image_url_to_base64(url[0])
        # img_description = await describe_base64_image(base64_img,OPEN_API_KEY, 3, question=question)
        # print(img_description)
        # question = question + img_description
        CONTEXT = 10
        discourse_embeddings = load_embeddings('discourse_embeddings.npz')
        markdown_embeddings = load_embeddings('markdown_embeddings.npz')
        # print(discourse_embeddings, markdown_embeddings)
        # generate embeddings
        embedding_response = await get_embeddings(question, OPEN_API_KEY)
        # print(embedding_response)
        # # find relevant content
        relevant_results = find_similar_content(embedding_response, CONTEXT, discourse_embeddings, markdown_embeddings)
        # print(relevant_results)
        print('\n\n\n')
        answer = await generate_answer(OPEN_API_KEY, question, relevant_results)
        print(answer, '\n\n\n')
        llm_response = parse_llm_response(answer)
        with open('answer.json', 'w') as f:
            json.dump(answer, f, indent = 4)
        print(llm_response)
        with open('response.json', 'w') as f:
            json.dump(llm_response, f, indent = 4)
        
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    question = """ How is TDS course"""
    asyncio.run(result_response(question))