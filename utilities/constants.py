import json
from datetime import datetime
import pytz

los_angeles_timezone = pytz.timezone('America/Los_Angeles')
los_angeles_date = datetime.now(los_angeles_timezone).date()
formatted_date = los_angeles_date.strftime("%B %d, %Y")

courses_numbers = ['544', '566', '585', '596', '599', '626', '677', '699']

course_names = ['Applied Natural Language Processing (NLP)', 'Deep Learning and its Applications (DL)',
                'Database Systems (database)', 'Scientific Computing and Visualization',
                'Distributed Systems', 'Text as Data',
                'Advanced Computer Vision (CV)', 'Robotic Perception (Robotics)']

full_course_info = dict(zip(['CSCI' + num for num in courses_numbers],
                            ['CSCI' + num + ' ' + name for num, name in zip(courses_numbers, course_names)]))

name_to_num = dict(zip(course_names, courses_numbers))

# all_courses and all_courses_prompt are used in Bert classifier
all_courses = ['CSCI' + num + ' ' + name for num, name in zip(courses_numbers, course_names)]
dataset_prompt = 'Are query 1: "{q_1}" and query 2: "{q_2}" asking about the same course?\nHere are all the courses we have:\n' + '\n'.join(all_courses)

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "RAG",
            "description": "Retrieve the relevant section in the given knowledge base when the user asks information about courses or syllabus. Call this function when user ask anything about the courses information. Any thing related to the courses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original question that the user asks exactly, no need to rephrase.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    },
     {
            "type": "function",
            "function": {
                "name": "fetch_weather",
                "description": "Fetches the current weather for a specified city with user-defined key selections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "params": {
                            "type": "string",
                            "description": (
                                "A string containing: "
                                "The name of the city (e.g., 'Los Angeles') and the Date of searching (e.g., December 16, 2024). So it should be strictly in the format just like: 'Los Angeles, December 16, 2024'"
                            ),
                        },
                    },
                    "required": ["params"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "general_news_report",
                "description": "Fetches recent news articles based on the user's query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's search query for the news (e.g., 'artificial intelligence').",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
]

tool_prompt = f"""
You have access to the following functions:

Use the function '{tools_list[0]['function']['name']}' to '{tools_list[0]['function']['description']}'.
The parameters are: {json.dumps(tools_list[0]['function']['parameters']['properties'])}, where {tools_list[0]['function']['parameters']['required']} are required.

Use the function '{tools_list[1]['function']['name']}' to '{tools_list[1]['function']['description']}':
The parameters are: {json.dumps(tools_list[1]['function']['parameters']['properties'])}, where {tools_list[1]['function']['parameters']['required']} are required.

Use the function '{tools_list[2]['function']['name']}' to '{tools_list[2]['function']['description']}':
The parameters are: {json.dumps(tools_list[2]['function']['parameters']['properties'])}, where {tools_list[2]['function']['parameters']['required']} are required.

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
- Just for your information, our current location is Los Angeles, and current date is {formatted_date}
"""

answer_sys_prompt = f"You are a very helpful assistant. Please answer user's question according to given information. Trust the given information, it is completely align with the user's question. Our current location is Los Angeles, and current date is {formatted_date}"

answer_prompt = """
## Question:
{query}

## Information:
{result}
"""

if __name__ == '__main__':
    print(tool_prompt)
