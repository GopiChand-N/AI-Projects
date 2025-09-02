import openai
from utils import image_file_to_data_uri

SYSTEM_PROMPT = (
    "Answer the user's question using only the provided image. "
    "Be concise but complete; avoid markdown formatting in the response. "
    "If information is not in the image, say so."
)

def answer_with_gpt4o(question: str, image_path: str, openai_api_key: str) -> str:
    client = openai.OpenAI(api_key=openai_api_key)
    data_uri = image_file_to_data_uri(image_path)
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {
                "role":"user",
                "content":[
                    {"type":"text","text":question},
                    {"type":"image_url","image_url":{"url":data_uri}}
            ]
            }
        ],
        temperature = 0.2
    )
    print("response", resp)
    return resp.choices[0].message.content.strip()