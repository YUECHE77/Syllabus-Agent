import faiss
import numpy as np
import re
import json

from utilities.constants import courses_numbers, course_names, name_to_num, tools_list


def build_vector_database(model):
    """
    Build the faiss database
    :param model: Embedding model
    :return: Faiss database
    """
    course_embeddings = model.encode(course_names)
    np_course_embeddings = np.array(course_embeddings).astype('float32')

    faiss.normalize_L2(np_course_embeddings)

    index = faiss.IndexFlatIP(len(course_embeddings[0]))

    index.add(np_course_embeddings)

    return index


def search_(query, database, model, embed_dim, top_k=1):
    query_embed = model.encode_queries(query)
    query_embed = np.array(query_embed).astype('float32')

    _, idx = database.search(query_embed.reshape((1, embed_dim)), top_k)  # (768,) -> (1, 768)
    idx = idx.reshape(-1)

    ret = [course_names[i] for i in idx]

    return ret


def find_syllabus(query, model, database):
    """
    Find the most relevant course to the user's query
    :param query: User's query
    :param model: Embedding model
    :param database: Faiss database
    :return: Name of the txt file for the course -> For example: CSCI544.txt
    """
    def find_course():
        for num in courses_numbers:
            if num in query:
                return num

        result_ = search_(query, database, model, embed_dim=database.d, top_k=1)

        return result_[0]

    result = find_course()

    if result in courses_numbers:
        return 'CSCI' + result + '.txt'
    else:
        return 'CSCI' + name_to_num[result] + '.txt'


def read_and_split_file(file_path):
    """
    Split the txt files into segments -> using "\n\n" as seperator
    :param file_path: Path to the txt file
    :return: The segments -> a list -> [...]
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    segments = content.split('##')

    return [seg.strip() for seg in segments]


def compute_similarity(query, segments, model):
    """
    For embedding model. Compute the similarity between query and each segments
    :param model: Embedding model
    :param query: User's query
    :param segments: The knowledge base which result from read_and_split_file()
    :return: The sorted (descending) results
    """
    # encode_queries() -> especially for encoding queries
    query_embedding = model.encode_queries(query)  # [512, ]
    segment_embeddings = model.encode(segments)  # [num_segments, 512]

    # Already normalized while encoding
    similarities = [query_embedding @ segment_embedding.T for segment_embedding in segment_embeddings]

    # [(score, segment), ...,(score, segment)]
    results = sorted(zip(similarities, segments), key=lambda x: x[0], reverse=True)

    return results


def generate_query_passage_pairs(query, segments):
    """
    For reranker model. Generate the pairs of query and segment
    :param query: User's query [query]
    :param segments: List of segments [segment_1, segment_2, ..., segment_n]
    :return: List of pairs [(query, segment)]
    """
    return [(query[0], seg) for seg in segments]


def parse_tool_response(response_message):
    """
    Parses the tool response for function calls and arguments.

    Args:
        response_message: Response message object with content or tool_calls.

    Returns:
        dict or None: Parsed response with function name and arguments as a string,
                      or None if parsing fails.
    """
    # Check if tool_calls are already provided
    if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        try:
            parsed_response = {
                "function": response_message.tool_calls[0].function.name,
                "arguments": response_message.tool_calls[0].function.arguments
            }
            return parsed_response
        except Exception as e:
            print(f"Error parsing tool_calls arguments: {e}")
            return None

    # Regex pattern to extract function calls and arguments
    # function_regex = r"<function=([a-zA-Z_]\w*)>(\{.*?\})\s*(?:</function>|<function(?:/[\w]*)?>)"
    function_regex = r"<function=([a-zA-Z_]\w*)>(\{.*?\})\s*\"?\s*(?:</function>|<function(?:/[\w]*)?>)"
    match = re.search(function_regex, response_message.content)

    if match:
        function_name, args_string = match.groups()
        # print("Extracted function_name:", function_name)
        # print("Extracted args_string:", args_string)

        # Simply return the function name and raw arguments as they are valid JSON strings
        return {
            "function": function_name,
            "arguments": args_string
        }

    return None


def generate_tool_answer(input, tokenizer, model):
    """
    For localized llama to generate response for the first inference
    :param input: user query
    :param tokenizer: llama tokenizer
    :param model: llama model
    :return: tools response or direct response
    """
    inputs = tokenizer.apply_chat_template(input, tools=tools_list, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(out[0][len(inputs["input_ids"][0]):])

    return res


def parse_tool_call(response_str: str):
    """
    For localized llama to parse the response
    :param response_str: Output from generate_tool_answer()
    :return: The function name and params
    """
    pattern = r">(.*?)</"
    match = re.search(pattern, response_str, flags=re.DOTALL)
    if not match:
        return None

    tool_call_str = match.group(1).strip()

    try:
        tool_call_data = json.loads(tool_call_str)
    except json.JSONDecodeError:
        return None

    function_name = tool_call_data.get("name")
    arguments = tool_call_data.get("arguments", {})

    return {
        "function": function_name,
        "arguments": arguments
    }


def generate_answer(input, tokenizer, model):
    """
    For localized llama to generate response for the first inference.
    Difference: No tools_list needed.
    :param input: User query
    :param tokenizer: llama tokenizer
    :param model: llama model
    :return: response from LLM
    """
    inputs = tokenizer.apply_chat_template(input, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(out[0][len(inputs["input_ids"][0]):])


def parse_answer(answer: str):
    """
    For localized llama to parse the second response
    :param answer: Answer from llama
    :return: parsed response
    """
    cleaned = re.sub(r"\s*## Answer:\s*", "", answer)
    cleaned = cleaned.replace("<|im_end|>", "")

    return cleaned.strip()


def test_func(content):
    """
    Just for test
    """
    function_regex = r"<function=([a-zA-Z_]\w*)>(\{.*?\})\s*\"?\s*(?:</function>|<function(?:/[\w]*)?>)"
    match = re.search(function_regex, content)

    if match:
        function_name, args_string = match.groups()
        print("Extracted function_name:", function_name, "Extracted args_string:", args_string)
        print()


if __name__ == '__main__':
    test_func('<function=RAG>{"query": "instructor for my NLP class"}"</function>')
    test_func('<function=RAG>{"query": "final report submission deadline for NLP class"}"</function>')
    test_func('<function=RAG>{"query": "main topics in the NLP class"}"</function>')
    test_func('<function=RAG>{"query": "main topics in the deep learning class"}"</function>')
    test_func('<function=RAG>{"query": "overlaps between NLP and deep learning classes"}"</function>')
