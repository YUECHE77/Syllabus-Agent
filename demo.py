from FlagEmbedding import FlagModel
from FlagEmbedding import FlagReranker

import os
import torch
import json
import warnings

from utilities.constants import tools_list, tool_prompt, answer_sys_prompt, answer_prompt
from utilities.utils import build_vector_database, parse_tool_response
from utilities.tool_functions import RAG, general_news_report, fetch_weather

from together import Together

from dotenv import load_dotenv
load_dotenv(dotenv_path=r'D:\CSCI544_project_code\.env')

warnings.filterwarnings("ignore")

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
together = Together(api_key=TOGETHER_API_KEY)

model = FlagModel(r'D:\HuggingFace_models\bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

reranker = FlagReranker(r'D:\HuggingFace_models\bge-reranker-base', use_fp16=True, device='cuda')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.model.to(device)

database = build_vector_database(model)
components = {
    'model': model,
    'reranker': reranker,
    'database': database
}

# query = 'Tell me the timeline of project for csci544. For example, when should I submit the final report?'
query = input('User: ')

messages = [{"role": "system", "content": tool_prompt},
            {"role": "user", "content": query}]

response = together.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=messages,
    max_tokens=1024,
    temperature=0,
    tools=tools_list,
    tool_choice="auto",
)

# parsed_response -> {'function': 'RAG', 'arguments': '{"query":"instructor for nlp class"}'}
parsed_response = parse_tool_response(response.choices[0].message)

if parsed_response:
    available_functions = {
        "RAG": RAG,
        "fetch_weather": fetch_weather,
        "general_news_report": general_news_report,
    }

    if parsed_response["function"] not in available_functions:
        available_function_names = "\n".join(available_functions.keys())
        raise NotImplementedError(
            f"Function {parsed_response['function']} is not implemented. "
            f"Our available functions are:\n\n{available_function_names}"
        )
    try:
        arguments = json.loads(parsed_response["arguments"])  # a dict -> {"query": "instructor for nlp class"}
        components.update(arguments)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse arguments: {e}")

    function_to_call = available_functions[parsed_response["function"]]
    result = function_to_call(**components)

    new_messages = [{"role": "system", "content": answer_sys_prompt},
                    {"role": "user", "content": answer_prompt.format(query=query, result=result)}]

    res = together.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=new_messages,
        max_tokens=1000,
        temperature=0.9,
    )

    print(res.choices[0].message.content)
else:
    print("No function call found in the response")
