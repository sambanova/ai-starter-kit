import os
from typing import Any

import litellm
import weave
from dotenv import load_dotenv

load_dotenv()

# Initialize Weave with your project name
weave.init('weave_integration_litellm')
SAMBANOVA_URL = os.getenv('SAMBANOVA_URL')
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
model = 'sambanova/Meta-Llama-3.3-70B-Instruct'


def litellm_completion(model: str, question: str) -> Any:
    response = litellm.completion(
        model=model,
        messages=[{'role': 'user', 'content': question}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


@weave.op()  # type: ignore
def litellm_translate(model: str, text: str, target_language: str) -> Any:
    response = litellm.completion(
        model=model, messages=[{'role': 'user', 'content': f"Translate '{text}' to {target_language}"}], max_tokens=1024
    )
    return response.choices[0].message.content


class LiteLLMTranslatorModel(weave.Model):  # type: ignore
    model: str
    temperature: float

    @weave.op()  # type: ignore
    def predict(self, text: str, target_language: str) -> Any:
        response = litellm.completion(
            model=self.model,
            messages=[
                {'role': 'system', 'content': f'You are a translator. Translate the given text to {target_language}.'},
                {'role': 'user', 'content': text},
            ],
            max_tokens=1024,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


if __name__ == '__main__':
    question = "Translate 'Hello, how are you?' to French."
    response = litellm_completion(model, question)
    print(response)

    print(litellm_translate(model, 'Hello, how are you?', 'French'))

    # Create an instance of the translator weave.Model
    translator = LiteLLMTranslatorModel(model=model, temperature=0.3)

    english_text = 'Hello, how are you today?'
    print(translator.predict(english_text, 'French'))
