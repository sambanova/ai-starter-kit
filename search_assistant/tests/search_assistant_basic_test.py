import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(file_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

print(kit_dir)
print(repo_dir)

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from search_assistant.src.search_assistant import SearchAssistant

method = 'basic_query'
input_disabled = False
tool = ['serpapi'] # serpapi, serper, openserp
search_engine = 'google' # google, bing, baidu
max_results = 5

search_assistant = SearchAssistant()

user_question = 'who is the president of America'
reformulated_query = search_assistant.reformulate_query_with_history(user_question)

response = search_assistant.basic_call(
                    query=user_question,
                    reformulated_query=reformulated_query,
                    search_method=tool[0],
                    max_results=max_results,
                    search_engine=search_engine,
                    conversational=True,
                )

print(response["sources"]) # list[str]
print(response["answer"]) # str