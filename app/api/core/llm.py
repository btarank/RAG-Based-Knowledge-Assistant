from groq import Groq
from app.api.core.config import settings

# Single shared client instance
client = Groq(api_key=settings.groq_api_key)

def get_llm_response(messages: list[dict], temperature: float = 0.2) -> str:
    """
    Send messages to Groq and get a response.
    messages format: [{"role": "user", "content": "..."}]
    """
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def get_llm_streaming(messages: list[dict], temperature: float = 0.2):
    """
    Stream response from Groq — returns a generator.
    Use this for the chat API endpoint.
    """
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta