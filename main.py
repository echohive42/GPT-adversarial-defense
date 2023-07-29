import openai
import json
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
import os 

# enter your openai api key here or retrive it from your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

model = "gpt-3.5-turbo-16k-0613"
# model = "gpt-4-0613"

is_adversarial = False  
user_input = None

"""
Alternative approach would be to check for objectionable response with the entire regular response at the end. But this will consume a lot of tokens each time and will add additional sequential wait time.
"""

# Messages which includes a descriptive system messsage which might alter the check_objectionable_response function's response. Because we want gpt to respond as it normally would, we might want to use no system message as to not influence its response. 
messages = [
            {"role": "system", "content": """You are a powerful language model which can influence people's opinions. Some people may try to get you to generate objectionable content. Therefore this particular call to you is to generate the first 30 words of a response as you regualrly would and when you understadt whether the response so far is objectionable or not, stop, and return the is_objectionable_response value as true or false."""},
            {"role": "user", "content": f"{user_input}"},
        ]

# we can just simply use a "user" message
# messages = [
#             {"role": "user", "content": f"{user_input}"},
#         ]

### above issue would require additonal consideration and thinking ###


def check_objectionable_response(text, model):
    response = openai.ChatCompletion.create(
        model = model,
        temperature = 0,
        messages = messages,
        stream = True,
        functions = [
            {
                "name": "regular_response_and_objectionable_response_detection",
                "description": "takes in first 30 words of regular response from gpt and detects if it is objectionable or not.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "regular_response": {
                            "type": "string",
                            "description": "first 30 words of regular response from gpt"
                        },
                        "is_objectionable_response": {
                            "type": "boolean",
                            "description": "whether the regular response is objectionable or not"

                    },
                    },
                    "required": ["regular_response", "is_objectionable_response"]
                    
                }
            }
        ],
        function_call = {"name": "regular_response_and_objectionable_response_detection"}
    )

    responses = ''
    for chunk in response:
        # print(chunk)
        if chunk["choices"][0]["delta"].get("function_call"):
            chunk = chunk["choices"][0]["delta"]
            # print(chunk)
            response_chunk = chunk["function_call"]["arguments"]
            responses += response_chunk
            print(colored(response_chunk, "green"), end='', flush=True)

    is_objectionable_response = json.loads(responses)["is_objectionable_response"]

    return is_objectionable_response

def regular_response(user_input, model):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=2000,
        messages=[
            {'role': 'user', 'content': f"{user_input}"},
        ],
        stream=True  # this time, we set stream=True
    )

    # Our previous approach prints unwanted information (delta, finish_reason, etc.)
    # We should extract the 'content' value from the chunks
    responses = ""

    for chunk in response:
        response_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if response_content:
            responses += response_content
            print(colored(response_content, "yellow"), end='', flush=True)

    return responses

def main(user_input, model):
    with ThreadPoolExecutor() as executor:
        objectionable_future = executor.submit(check_objectionable_response, user_input, model)
        regular_response_future = executor.submit(regular_response, user_input, model)

        # Wait for the objectionable check to finish
        is_objectionable_response = objectionable_future.result()
        # test the objectionable check with True below
        # is_objectionable_response = True

        if is_objectionable_response:
            # If objectionable, we ignore the regular response
            print("Objectionable content detected.")
        else:
            # If not objectionable, we print the regular response
            regular_response_result = regular_response_future.result()
            print(regular_response_result)

user_input = input("Enter your text: ")
main(user_input, model)


