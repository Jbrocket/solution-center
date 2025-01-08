import json, os

import headers

with open("queries.json") as f:
    query_data = json.load(f)

cwd = os.getcwd()
responses = {}

for query in query_data:
    userMessage = query.pop("userMessage", None)
    responses[userMessage] = {"metadata": query, "responses": {}}
    for model in headers.MODEL:
        for search_type in headers.SEARCH_TYPE:
            for vector_string in headers.VECTOR_FIELDS_TO_USE:
                os.system(f"python3 test_prompt_flow.py --index {model} --query \"{userMessage}\" --semantic-config {search_type} --vector-fields {vector_string} --output-file \"{cwd}/file.json\"")
                with open("file.json") as f:
                    data = json.load(f)
                responses[userMessage]["responses"][f"{model}_{search_type}_{vector_string}"] = data[userMessage][:5]
                json.dump(responses, open("responses.json", "w"), indent=4)