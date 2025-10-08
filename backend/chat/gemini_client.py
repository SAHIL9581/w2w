# backend/chat/gemini_client.py
import base64
import os
from typing import List
import sys

import httpx
from fastapi import HTTPException

from .prompt import SYSTEM_PROMPT as CHAT_SYSTEM_PROMPT

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"

# --- FIX: Removed the extra single quotes (') around the API key ---
# The key should be directly part of the URL string.



def get_mime_type(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext == "png": return "image/png"
    elif ext in ["jpg", "jpeg"]: return "image/jpeg"
    return "application/octet-stream"


async def call_gemini_api(question: str, image_paths: List[str], system_prompt: str = CHAT_SYSTEM_PROMPT) -> str:
    print(f"✅ Using API URL ending in: ...{GEMINI_API_URL[-6:]}") # Prints end of URL for verification
    parts = [{"text": question}]
    for image_path in image_paths:
        try:
            if not os.path.exists(image_path): continue
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
            mime_type = get_mime_type(os.path.basename(image_path))
            parts.append({"inline_data": {"mime_type": mime_type, "data": image_b64}})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image file: {os.path.basename(image_path)}")
    
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.4, "topK": 32, "topP": 1, "maxOutputTokens": 4096},
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(GEMINI_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            return "Error: Could not parse a valid response from the model."
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from Gemini API: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")