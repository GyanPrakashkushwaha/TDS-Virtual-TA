from pathlib import Path
import numpy as np
from tqdm import tqdm
from embed import get_chunks, get_embeddings
from config import GEMINI_API_KEY
from extract_text import extract_text_from_discourse, clean_html
import asyncio
import signal
import json, sys, os
import re
import base64
# from google.generativeai import Client, types
from google.genai import Client, types
import requests
import base64
from io import BytesIO
from PIL import Image
import urllib




def extract_europe1_urls(text):
    # Regex pattern to find URLs starting with 'https://europe1'
    pattern = r'https://europe1[^"\s]+'
    urls = re.findall(pattern, text)
    return urls

def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


async def process_save_discourse():
    files = [*Path("raw-data/Discourse-data").glob("*.json")]
    all_chunks = []
    all_embeddings = []
    all_original_urls = []
    c = 0

    for file_path in files:
        data = read_json_file(file_path)
        posts = data.get('post_stream', {}).get('posts', [])
        topic_id = data.get('id')
        topic_slug = data.get('slug', '')
        
        complete_post = ''
        for post in posts:
            post_number = post.get('post_number')
            content = post.get('cooked', '')
            question_img_url = extract_europe1_urls(content) if extract_europe1_urls(content) else None
            clean_content = clean_html(content)
            if question_img_url:
                print('================================================== IMG FOUND \n')
                url = question_img_url[0]

                b64IMG = image_url_to_base64(url)
                if b64IMG:
                    try:
                        description = describe_base64_image(b64IMG, GEMINI_API_KEY)
                        print(description)
                        complete_post += description
                    except Exception as e:
                        print(f"Error describing image {url}: {e}")
                else:
                    print(f"Skipping image {url} due to download error.")
                
            complete_post += clean_content

        chunks = get_chunks(complete_post)
        topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}"
        print(f'========= Topic URL ======================= \n {topic_url} \n\n')
        for chunk in chunks:
            all_chunks.append([chunk])
            all_original_urls.append([topic_url])
            # create embeddings for chunks
            try:
                embedding = await get_embeddings(chunk, GEMINI_API_KEY)
                all_embeddings.append([embedding])
                print(f'embedding is being generated for {file_path.name} the chunk is {chunk[:50]}... \n Embeddings {embedding[:5]} and url is {url}')
            except Exception as e:
                print(f"Error getting embedding for chunk: {e}")

    data_safe = {
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "original_urls": all_original_urls,
    }
    
    with open("discourse_embeddings_text.txt", "w", encoding="utf-8") as f:
        f.write(str(data_safe))
    
    with open("discourse_embeddings_safe.json", "w", encoding="utf-8") as f:
        json.dump(data_safe, f, indent=2, ensure_ascii=False)
        
    np.savez("discourse_embeddings.npz", 
             chunks=all_chunks, 
             embeddings=all_embeddings, 
             original_urls=all_original_urls)

    
if __name__ == "__main__":
    asyncio.run(process_save_discourse())
    print("Processing complete. Embeddings saved to 'discourse_embeddings.npz'.")  # Fixed syntax
