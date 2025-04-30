import json
import os
import random
from typing import Any, Dict, Optional

import litellm
import requests
import weave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize Weave with your project name
weave.init('weave_integration_openai')
SAMBANOVA_URL = os.getenv('SAMBANOVA_URL')
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')
sambanova_client = OpenAI(base_url=SAMBANOVA_URL, api_key=SAMBANOVA_API_KEY)
model = 'Meta-Llama-3.3-70B-Instruct'

PROMPT = """
Emulate the Pokedex from early Pokémon episodes. State the name of the Pokemon and then describe it.
Your tone is informative yet sassy, blending factual details with a touch of dry humor.
Be concise, no more than 3 sentences.
"""
POKEMON = ['pikachu', 'charmander', 'squirtle', 'bulbasaur', 'jigglypuff', 'meowth', 'eevee']


@weave.op()  # type: ignore
def extract_fruit(sentence: str) -> Optional[Any]:
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
    if isinstance(extracted, str):
        extracted_json = json.loads(extracted)
    return extracted_json


@weave.op  # type: ignore
def emoji_bot(question: str) -> Optional[str]:
    response = sambanova_client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': 'You are AGI. '
                'You will be provided with a message, and your task is to respond using emojis only.',
            },
            {'role': 'user', 'content': question},
        ],
        temperature=0.8,
        max_tokens=64,
        top_p=1,
    )
    return response.choices[0].message.content


@weave.op  # type: ignore
def get_pokemon_data(pokemon_name: str) -> Optional[Dict[str, Any]]:
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


@weave.op  # type: ignore
def pokedex(name: str, prompt: str) -> Optional[str]:
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


class GrammarCorrectorModel(weave.Model):  # type: ignore
    model: str
    system_message: str

    @weave.op()  # type: ignore
    def predict(self, user_input: str) -> Optional[str]:
        response = sambanova_client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': self.system_message}, {'role': 'user', 'content': user_input}],
            temperature=0,
        )
        return response.choices[0].message.content


if __name__ == '__main__':
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
