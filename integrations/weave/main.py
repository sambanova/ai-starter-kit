from typing import Optional
import weave
from langchain_core.prompts import PromptTemplate
from langchain_sambanova import ChatSambaNovaCloud
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import requests, random
import litellm

load_dotenv()

# Initialize Weave with your project name
weave.init('weave_integration')
SAMBANOVA_URL = os.getenv('SAMBANOVA_URL')
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
sambanova_client = OpenAI(base_url=SAMBANOVA_URL, api_key=SAMBANOVA_API_KEY)
model = 'Meta-Llama-3.3-70B-Instruct'

PROMPT = """Emulate the Pokedex from early Pokémon episodes. State the name of the Pokemon and then describe it.
        Your tone is informative yet sassy, blending factual details with a touch of dry humor. Be concise, no more than 3 sentences. """
POKEMON = ['pikachu', 'charmander', 'squirtle', 'bulbasaur', 'jigglypuff', 'meowth', 'eevee']


@weave.op()  # 🐝 Decorator to track requests
def extract_fruit(sentence: str) -> dict:
    system_prompt = 'Parse sentences into a JSON dict with keys: fruit, color and flavor. Only return the JSON object.'
    response = sambanova_client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {'role': 'user', 'content': sentence},
        ],
        temperature=0.7,
        response_format={'type': 'json_object'},
    )
    extracted = response.choices[0].message.content
    extracted_json = json.loads(extracted)
    return extracted_json


@weave.op
def emoji_bot(question: str) -> str:
    response = sambanova_client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'You are AGI. You will be provided with a message, and your task is to respond using emojis only.',
            },
            {'role': 'user', 'content': question},
        ],
        temperature=0.8,
        max_tokens=64,
        top_p=1,
    )
    return response.choices[0].message.content


PROMPT = """Emulate the Pokedex from early Pokémon episodes. State the name of the Pokemon and then describe it.
        Your tone is informative yet sassy, blending factual details with a touch of dry humor. Be concise, no more than 3 sentences. """
POKEMON = ['pikachu', 'charmander', 'squirtle', 'bulbasaur', 'jigglypuff', 'meowth', 'eevee']
client = OpenAI()


@weave.op
def get_pokemon_data(pokemon_name: str) -> Optional[str]:
    # This is a step within your application, like the retrieval step within a RAG app
    url = f'https://pokeapi.co/api/v2/pokemon/{pokemon_name}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        name = data['name']
        types = [t['type']['name'] for t in data['types']]
        species_url = data['species']['url']
        species_response = requests.get(species_url)
        evolved_from = 'Unknown'
        if species_response.status_code == 200:
            species_data = species_response.json()
            if species_data['evolves_from_species']:
                evolved_from = species_data['evolves_from_species']['name']
        return {'name': name, 'types': types, 'evolved_from': evolved_from}
    else:
        return None


@weave.op
def pokedex(name: str, prompt: str) -> str:
    # This is your root op that calls out to other ops
    data = get_pokemon_data(name)
    if not data:
        return 'Error: Unable to fetch data'
    response = sambanova_client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': str(data)}],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
    )
    return response.choices[0].message.content


class GrammarCorrectorModel(weave.Model):
    model: str
    system_message: str

    @weave.op()
    def predict(self, user_input: str) -> str:  # Change to `predict`
        client = OpenAI()
        response = sambanova_client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': user_input}],
            temperature=0,
        )
        return response.choices[0].message.content


class TranslatorModel(weave.Model):
    model: str
    temperature: float

    @weave.op()
    def predict(self, text: str, target_language: str) -> str:
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


def main() -> None:
    weave.init('integration')  # 🐝

    # Fruit extraction
    sentence = (
        'There are many fruits that were found on the recently discovered planet Goocrux. '
        'There are neoskizzles that grow there, which are purple and taste like candy.'
    )
    print(extract_fruit(sentence))

    # Emoji bot
    question = 'How are you?'
    print(emoji_bot(question))

    # Get data for a specific Pokémon
    pokemon_data = pokedex(random.choice(POKEMON), PROMPT)

    # Correct grammar
    corrector = GrammarCorrectorModel(
        model=model, system_message='You are a grammar checker, correct the following user input.'
    )
    result = corrector.predict('That was so easy, it was a piece of pie!')
    print(result)

    # Create instances with different models
    gpt_translator = TranslatorModel(model='sambanova/' + model, temperature=0.3)
    claude_translator = TranslatorModel(model='sambanova/' + model, temperature=0.1)

    # Use different models for translation
    english_text = 'Hello, how are you today?'

    print('GPT-3.5 Translation to French:')
    print(gpt_translator.predict(english_text, 'French'))

    print('\nClaude-3.5 Sonnet Translation to Spanish:')
    print(claude_translator.predict(english_text, 'Spanish'))


if __name__ == '__main__':
    main()
