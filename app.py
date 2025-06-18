from embed import (get_embeddings, 
                   generate_answer,
                   describe_base64_image)
from config import GEMINI_API_KEY
from helper import (load_embeddings, 
                    extract_europe1_urls,
                    image_url_to_base64)
from get_answer import find_similar_content, parse_llm_response
import asyncio
import json


async def result_response(question):
    try:
        url = extract_europe1_urls(question)
        base64_img = image_url_to_base64(url[0])
        img_description = await describe_base64_image(base64_img,GEMINI_API_KEY, 3)
        # print(img_description)
        question = question + img_description
        CONTEXT = 10
        discourse_embeddings = load_embeddings('embeddings\discourse_embeddings.npz')
        markdown_embeddings = load_embeddings('embeddings\markdown_embeddings.npz')
        # generate embeddings
        embedding_response = await get_embeddings(question, GEMINI_API_KEY)
        # find relevant content
        relevant_results = find_similar_content(embedding_response, CONTEXT, discourse_embeddings, markdown_embeddings)
        answer = await generate_answer(GEMINI_API_KEY, question, relevant_results)
        llm_response = parse_llm_response(answer)
        print(llm_response)
        with open('response.json', 'w') as f:
            json.dump(llm_response, f, indent = 4)
        
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    question = """  The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo? https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/b/8/b86f7ddaff9b11e54f240e51d39ac9f51cb03e25.png
    """
    asyncio.run(result_response(question))