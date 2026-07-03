import os

from litellm import OpenAI

# In-memory log of every agent LLM exchange (prompt + response), in call order.
# Drivers reset it before a run and persist it afterwards as a trajectory.
MESSAGE_LOG: list[dict] = []


def reset_message_log():
    MESSAGE_LOG.clear()


def get_message_log():
    return list(MESSAGE_LOG)


def call_llm(prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # Agentic baselines use gpt-5 by default; override via POCKETFLOW_MODEL.
    model = os.environ.get("POCKETFLOW_MODEL", "gpt-5")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    MESSAGE_LOG.append({"model": model, "prompt": prompt, "response": content})
    return content
