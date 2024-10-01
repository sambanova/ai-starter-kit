import os  # for using env variables

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple  # for type hint

import yaml  # for loading prompt example config file
from langchain.prompts import PromptTemplate, load_prompt  # for creating and loading prompting yaml files
from langchain_core.language_models.llms import LLM

from utils.model_wrappers.api_gateway import APIGateway

# define config path
CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')


@dataclass(init=False)
class LLMManager:
    """A class to manage the configuration, setup, and interaction with various LLMs."""

    def __init__(self) -> None:
        """Gets model information and prompt use cases from config file"""
        llm_info, model_info, prompt_use_cases = self._get_config_info()

        self.llm_info = llm_info
        self.model_info = model_info
        self.prompt_use_cases = prompt_use_cases

    def _get_config_info(self) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
        """Loads json config file"""

        # Read config file
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        model_info = config['models']
        llm_info = config['llm']
        prompt_use_cases = config['use_cases']

        return llm_info, model_info, prompt_use_cases

    def set_llm(self, model_expert: str) -> LLM:
        """Sets a langchain embedding model
        Args:
            llm_info (dict):
            coe_flag (bool):
            model_expert (str):
        Returns:
            langchain embedding model
        """
        llm = APIGateway.load_llm(
            type=self.llm_info['api'],
            streaming=False,
            coe=self.llm_info['coe'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=model_expert,
        )
        return llm

    def get_prompt_template(self, model: str, prompt_use_case: str) -> Any:
        """
        Reads a prompt template from an specified model and use case.

        Args:
            model: model name.
            prompt_use_case: use case name.

        Returns:
            The prompt template associated to the model and use case selected.
        """
        # Load prompt from the corresponding yaml file
        prompt_file_name = f"{model.lower()}-prompt_engineering-{prompt_use_case.lower().replace(' ','_')}_usecase.yaml"
        prompt = load_prompt(f'./prompts/{prompt_file_name}')

        # Check prompt template existance
        assert hasattr(prompt, 'template'), 'The loaded prompt has no attribute template.'

        return prompt.template

    def create_prompt_yamls(self) -> None:
        """Shows a way how prompt yamls can be created. We're going to save our prompts in yaml files."""

        # Given a set of prompts based on the use case and model used
        prompts_templates = {
            'General Assistant': {
                'Llama2': """[INST] <<SYS>> You are a helpful, respectful, positive, and honest assistant. Your answers
                should not include any unsafe, unethical, or illegal content. If you don't understand the question or
                don't know the answer, please don't share false information. <</SYS>>\n\nHow can I write better prompts
                for large language models? [/INST]"""
            },
            'Document Search': {
                'Llama2': """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. If
                the answer is not in context for answering, say that you don't know, don't try to make up an answer or
                provide an answer not extracted from provided context. <</SYS>>\nContext: Early Account Closure Fee $30
                (if account is closed within 6 months of opening) \nReplacement of ATM Card $5 per card \nReplacement Of
                Lost Passbook $15 per passbook \nQuestion: I lost my ATM card. How much will it cost to replace it?
                [/INST]"""
            },
            'Product Selection': {
                'Llama2': """[INST] <<SYS>> You are an electrical engineer ai assistant.  You will be given context that
                will help answer a question.  You will interpret the context for useful information that will answer
                the question.  Provide concise, helpful, technical information (focusing on numerical information). 
                <</SYS>> [/INST] \n    Here is the question and context: \n\n    question: 'what is the switching
                frequency of max20710?'\n    contexts: VREF  Range | | See Table 8 for accuracy vs. �REF VREF
                (Note 6 ) | 0.6016 | | 1.0 | V | | FEEDBACK LOOP | | | | | | | | Integrator Recovery-Time
                Constant | tREC tREC  | | | 20 | | �s μs | | Gain (see the Control Loop Stability section for details)
                | RGAIN RGAIN  | Selected by R_SELB or PMBus ( ( Note 5,6,7,8) 5,6,7,8) | 0.72 | 0.9 | 1.1 | mV/A mV/A
                | | | | | 1.4 | 1.8 | 2.2 | | | | | | 2.9 | 3.6 | 4.3 | | | SWITCHING FREQUENCY | | | | | | | | 
                Switching Frequency | ��� fSW  | 400kHz/600kHz/800kHz 400kHz/600kHz/800kHz selected by C_SELB; other
                values are set through PMBus (Note 6 ) | | 400 | | kHz kHz | | | | | | 500 | | | | | | | | 600 | | | |
                | | | | 700 | | | | | | | | 800 | | | | | | | | 900 | | | | Switching Frequency Accuracy | | ( ( Note
                5,7,8) 5,7,8) | -20 | | +20 | % % | | INPUT PROTECTION | | | | | | | | Rising VDDH VDDH  UVLO Threshold
                | VDDH UVLO | (Note 5) | | 4.25 | 4.47 | V | | Falling VDDH VDDH  UVLO Threshold | | | 3.7 | 3.9 | | |
                | Hysteresis | | | | 350 | | mV mV |\n\n    [INST] Given the context you received, answer the question.
                Please do not infer anything if the question cannot be deduced from the context or make anything up; 
                simply state you cannot find the information.\n\n    Sure, here is the answer to the question:[/INST]"""
            },
            'Code Generation': {
                'Llama2': """[INST] <<SYS>> You are a helpful code assistant. Generate a valid JSON object based on the
                inpput. \nExample: name: John lastname: Smith address: #1 Samuel St. would be converted to:\n{\n
                ""address"": ""#1 Samuel St."",\n""lastname"": ""Smith"",\n""name"": ""John""\n} <</SYS>>\n[INST] Input:
                name: Ted lastname: Pot address: #1 Bisson St. [/INST]"""
            },
            'Summarization': {
                'Llama2': """[INST] <<SYS>> You are a helpful assistant. Your answers should not include any unsafe,
                unethical, or illegal content. <</SYS>>\n\nSummarize the customer support call transcript below:\n\n
                Customer: Hi, I'm having trouble accessing my account.\n\nSupport Agent: Sure, can you please provide
                me with your account username?\n\nCustomer: It's 'example123'.\n\nSupport Agent: Thank you. I see
                that there was a recent security update that might have affected access. Let me check on that for you.
                \n\nCustomer: Okay, thank you.\n\nSupport Agent: It seems like there was a temporary glitch in the
                system. I've fixed it, and you should be able to access your account now.\n\nCustomer: Great, thank you
                so much for your help!\n\nSupport Agent: You're welcome. Is there anything else I can assist you with
                today?\n\nCustomer: No, that's all. Thank you again.\n\nSummarize the conversation highlighting the
                customer's issue, the steps taken by the support agent to resolve it, and the resolution outcome.
                [/INST]"""
            },
            'Question & Answering': {
                'Llama2': """[INST] <<SYS>> You are a helpful, respectful, positive, and honest assistant. Your answers
                should not include any unsafe, unethical, or illegal content. If you don't understand the question or
                don't know the answer, please don't share false information. <</SYS>>\n\nBased on the following excerpt
                extraction from an Amazon revenue report, answer the question.\n\nExcerpt:\n\nTotal revenue for Q4 2023
                reached $150 billion, marking a 20% increase compared to the same period last year. The largest
                contributor to this revenue growth was the North American market, which saw a 25% increase in sales,
                reaching $90 billion. International sales also showed significant growth, with a 15% increase, totaling
                $60 billion.\n\nQuestion:\n\nWhat was the percentage increase in total revenue for Amazon in Q4 2023
                compared to the same period last year? [/INST]"""
            },
            'Query decomposition': {
                'Llama2': """[INST] <<SYS>> Decompose a complex query into a list of questions that can be addressed
                individually. Follow these rules: 1. Only use the information given in the query. 2. The output is in
                JSON format. 3. No explanation or conclusions are necessary. <</SYS>>\n\nQuery example:\n\n'''Compare
                the prices, security features, engines, and overall user experience of Ford Escape and Toyota Rav4.'''
                \n\nOutput:\n\n<<<{questions: ['1. What is the price of the Ford Escape and how does it vary across
                different trim levels?','2. What is the price of the Toyota Rav4 and how does it vary across different
                trim levels?','3. What are the security features offered in the Ford Escape?','4. What are the security
                features offered in the Toyota Rav4?','5. What are the engine specifications of the Ford Escape?','6.
                What are the engine specifications of the Toyota Rav4?','7. What is the overall user experience of the
                Ford Escape?','8. What is the overall user experience of the Toyota Rav4?']}>>>\n\nNew query:\n\n'''
                Compare the features, performance, and prices of the iPhone models iPhone 12, iPhone 12 Pro, and iPhone
                12 Pro Max, including differences in camera capabilities, display specifications, battery life, and 
                overall user experience.'''\n\nOuput: [/INST]"""
            },
        }

        # Iterate and save prompts in yaml files
        for usecase_key, usecase_value in prompts_templates.items():
            for model_key, prommpt_template in usecase_value.items():
                prompt_file_name = f"""{model_key.lower().replace(' ','_')}-prompt_engineering-
                {usecase_key.lower().replace(' ','_')}_usecase"""
                prompt = PromptTemplate.from_template(prommpt_template)
                prompt.save(f'./prompts/{prompt_file_name}.yaml')
