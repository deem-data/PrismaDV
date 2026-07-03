from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_langchain_model(model_name: str, temperature: float = 0.6):
    openai_api_model_list = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano",
                             "gpt-4.1-mini", "gpt-4.1", "gpt-5", "gpt-5-mini", "gpt-5-low", "gpt-5-high"]
    google_api_model_list = ['gemini-2.5-pro', 'gemini-2.5-flash']
    if model_name in openai_api_model_list:
        if model_name == "gpt-5-low":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "low", "summary": "auto"},
                                   temperature=temperature)
        elif model_name == "gpt-5":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "medium", "summary": "auto"},
                                   temperature=temperature)
        elif model_name == "gpt-5-high":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "high", "summary": "auto"},
                                   temperature=temperature)
        else:
            model_api = ChatOpenAI(model_name=model_name, temperature=temperature)
    elif model_name in google_api_model_list:
        model_api = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif model_name == "vllm":
        inference_server_url = "http://localhost:8000/v1"
        model_api = ChatOpenAI(
            model="",
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported.")
    return model_api
