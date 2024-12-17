from FlagEmbedding import FlagModel
from FlagEmbedding import FlagReranker

import os
import torch
import json
import warnings
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok

from utilities.constants import tools_list, tool_prompt, answer_sys_prompt, answer_prompt
from utilities.utils import build_vector_database, parse_tool_response, find_syllabus
from utilities.tool_functions import RAG, general_news_report, fetch_weather

from binary_classifier.nets import BertClassifier
from binary_classifier.utils import inference

from transformers import BertTokenizerFast

from together import Together

from dotenv import load_dotenv
load_dotenv(dotenv_path=r'D:\CSCI544_project_code\.env')

warnings.filterwarnings("ignore")

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
together = Together(api_key=TOGETHER_API_KEY)

# set ngrok authtoken
app = Flask(__name__)
NGROK_API_KEY = os.getenv('NGROK_API_KEY')
ngrok.set_auth_token(NGROK_API_KEY)

bge_embedding_path = r'D:\HuggingFace_models\bge-base-en-v1.5'
bge_reranker_path = r'D:\HuggingFace_models\bge-reranker-base'
bert_model_path = r'D:\HuggingFace_models\bert-base-uncased'
trained_classifier_path = r'D:\CSCI544_project_code\models\CSCI544_Bert_best.pth'

model = FlagModel(bge_embedding_path,
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

reranker = FlagReranker(bge_reranker_path, use_fp16=True, device='cuda')

bert_model = BertClassifier(bert_model_path, num_classes=1, dropout=0.2)
bert_model.load_state_dict(torch.load(trained_classifier_path))

bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.model.to(device)
bert_model.to(device)

database = build_vector_database(model)

# -------------------------------- Start --------------------------------
available_functions = {
        "RAG": RAG,
        "fetch_weather": fetch_weather,
        "general_news_report": general_news_report,
}

chat_history = []  # store the conversation
previous_query_class = None  # desired format: (previous_query, previous_class)

# start ngrok tunnel
public_url = ngrok.connect('5000')
print(f"Public URL: {public_url}")


@app.route('/')
def index():
    """
    Render the front-end HTML page.
    """
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """
    Receive user queries from the front-end and return responses, fully integrating logic from the main function.
    """
    global chat_history, previous_query_class

    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"response": "Please provide a valid query."})

    try:
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        messages = [{"role": "system", "content": tool_prompt}] + chat_history
        messages += [{"role": "user", "content": user_query}]

        response = together.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0,
            tools=tools_list,
            tool_choice="auto",
        )

        parsed_response = parse_tool_response(response.choices[0].message)

        if parsed_response:
            if parsed_response["function"] not in available_functions:
                available_function_names = "\n".join(available_functions.keys())
                raise NotImplementedError(
                    f"Function {parsed_response['function']} is not implemented. "
                    f"Our available functions are:\n\n{available_function_names}"
                )

            components = {
                'model': model,
                'reranker': reranker,
                'database': database
            }

            try:
                arguments = json.loads(parsed_response["arguments"])  # a dict -> {"query": "instructor for nlp class"}

                if parsed_response["function"] == 'RAG':
                    if previous_query_class is not None:
                        previous_query, previous_class = previous_query_class

                        if_changed = inference(previous_query, arguments['query'], bert_model, bert_tokenizer, device)
                        assert if_changed in ['Yes', 'No'], f'The result from classifier must be "Yes" or "No". But here is {if_changed}.'

                        if if_changed == 'Yes':
                            current_class = find_syllabus(arguments['query'], components['model'], components['database'])
                            current_class = current_class[:len('.txt')]
                        else:
                            current_class = previous_class
                            arguments['query'] = f'For {current_class}, ' + arguments['query']  # update the query

                    else:
                        current_class = find_syllabus(arguments['query'], components['model'], components['database'])
                        current_class = current_class[:len('.txt')]

                    previous_query_class = (arguments['query'], current_class)  # update the tuple

                components.update(arguments)

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse arguments: {e}")

            function_to_call = available_functions[parsed_response["function"]]
            result = function_to_call(**components)

            chat_history.append({"role": "user", "content": answer_prompt.format(query=user_query, result=result)})
            new_messages = [{"role": "system", "content": answer_sys_prompt}]
            new_messages += chat_history

            final_response = together.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=new_messages,
                max_tokens=1024,
                temperature=0.9,
            )

            assistant_response = final_response.choices[0].message.content

        else:
            chat_history.append({"role": "user", "content": user_query})

            assistant_response = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": assistant_response})

        return jsonify({"response": assistant_response})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"response": f"An error occurred: {str(e)}"})


# Start Flask application
if __name__ == '__main__':
    app.run(port=5000)
