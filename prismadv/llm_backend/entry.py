from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from prismadv.llm_backend.model_routes import google_api_key, split_effort

GEMINI_TIMEOUT_S = 600  # seconds per request
NO_TEMPERATURE_MODELS = {"gpt-6-sol"}


def _sampling(base_model: str, temperature: float) -> dict:
    return {} if base_model in NO_TEMPERATURE_MODELS else {"temperature": temperature}


def get_langchain_model(model_name: str, temperature: float = 0.6):
    openai_api_model_list = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-nano",
                             "gpt-4.1-mini", "gpt-4.1", "gpt-5", "gpt-5-mini", "gpt-5-low", "gpt-5-high",
                             "gpt-5.6-sol", "gpt-6-sol"]
    if model_name in openai_api_model_list:
        if model_name == "gpt-5-low":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "low"},
                                   temperature=temperature)
        elif model_name == "gpt-5":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "medium"},
                                   temperature=temperature)
        elif model_name == "gpt-5-high":
            model_api = ChatOpenAI(model_name="gpt-5", reasoning={"effort": "high"},
                                   temperature=temperature)
        else:
            model_api = ChatOpenAI(model_name=model_name, **_sampling(model_name, temperature))
    elif split_effort(model_name)[0] in openai_api_model_list:
        base, effort = split_effort(model_name)
        model_api = ChatOpenAI(model_name=base, reasoning={"effort": effort}, **_sampling(base, temperature))
    elif model_name.startswith("gemini-"):
        model_api = ChatGoogleGenerativeAI(model=model_name, temperature=temperature,
                                           google_api_key=google_api_key(),
                                           transport="rest", timeout=GEMINI_TIMEOUT_S)
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
