import json

with open ("responses.json") as f:
    question_data = json.load(f)

combinations = list(question_data[list(question_data.keys())[0]]["responses"].keys())
results = {}

for combination in combinations:
    results[combination] = 0
results["base"] = 0

for question in question_data:
    response_data = question_data[question]
    metadata = response_data["metadata"]
    human_annotation = metadata["humanAnnotation"]
    initial_recommendation = metadata["recommendation"]
    query = question

    if human_annotation == initial_recommendation:
        results["base"] += 1

    for response in response_data["responses"]:
        for recommended in response_data["responses"][response]:
            if recommended["title"] == human_annotation:
                results[response] += 1

print(results)
json.dump(results, open("result_count.json", "w"), indent=4)
