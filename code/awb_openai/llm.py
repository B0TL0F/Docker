import traceback
from openai import OpenAI

from logger import logger

client = OpenAI()

def extractions_llm(raw_text, model_name, json_schema, instructions):
    logger.info("Requesting OpenAI Chat Completion...")

    prompt = (
        f"The following text is extracted from a shipping document:\n\n"
        f"{raw_text}\n\n"
        f"Extract All details as per the given JSON schema and generate JSON output. JSON Schema - {json_schema}."
        f"{instructions}"	
    )
    
    try:
        response = client.chat.completions.create(
            model = model_name,
            messages=[
                {"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        logger.info("OpenAI response received.")
        return response.choices[0].message.content
    
    except Exception as e: 
        logger.error(f"OPENAI request failed: {e}\n{traceback.format_exc()}")
        raise Exception("OPENAI request failed.")
