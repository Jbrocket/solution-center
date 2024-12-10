import os, re, json, requests, logging
import numpy as np
import tiktoken
from openai import AzureOpenAI

FILE_PATH = "workloads/workloads.json"
DEPLOYMENT_MODEL = "text-embedding-3-small"
PAT = os.getenv("GITHUB_EMU_PAT")
HEADERS = {'Authorization': f'token {PAT}'}

def read_file(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_file(file_path: str, data: list[dict]):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    return 0

def get_readme(github_url: str) -> str:
    if github_url.startswith("https://github.com"):
        raw_github = "https://raw.githubusercontent.com"
        github_url = github_url.replace("https://github.com", raw_github)
        readme_url = f"{github_url}/refs/heads/main/README.md"
        response = requests.get(readme_url)
        if response.status_code == 404:
            try:
                github_api = github_url.replace(raw_github, "https://api.github.com/repos")
                res = requests.get(github_api, headers=HEADERS)
                repo_info = res.json()
                main_branch = repo_info.get("default_branch", "main")

                res = requests.get(f"{github_api}/contents", headers=HEADERS)
                contents = res.json()
                readme_file = [file['name'] for file in contents if 'readme' in file['name'].lower()][0]

                raw_github = "https://raw.githubusercontent.com"
                github_url = github_url.replace("https://github.com", raw_github)
                readme_url = f"{github_url}/refs/heads/{main_branch}/{readme_file}"

                response = requests.get(readme_url)
            except Exception as e:
                logging.error(f"Error getting readme file: {e}")
                raw_github = "https://raw.githubusercontent.com"
                github_url = github_url.replace("https://github.com", raw_github)
                readme_url = f"{github_url}/refs/heads/main/README.md"
                response = requests.get(readme_url)
    else: 
        readme_url = github_url
        response = requests.get(readme_url, headers=HEADERS)

    return response.text, response

def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def average_vectors(vectors):
    return np.mean(vectors, axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_embeddings():
    with open("scripts/generate_embeddings/similarities.json", "w") as f:
        f.write("Avg vs Combined; Combined vs Readme; Readme vs Baseline; Avg vs Baseline; Combined vs Baseline\n")

    workloads = read_file(FILE_PATH)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    client: AzureOpenAI = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        api_version = "2024-02-01",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    failed_workloads = []
    for workload in workloads:
        print(workload["source"])
        readmeString, raw = get_readme(workload["source"])
        readmeString = normalize_text(readmeString)
        readmeEncode = tokenizer.encode(readmeString)
        readmeDecode = tokenizer.decode_tokens_bytes(readmeEncode)
        print(len(readmeDecode), f'{len(readmeDecode)<8192} {workload["source"]}')
        if len(readmeDecode) == 4:
            print(raw)
            failed_workloads.append({
                "title": workload["title"],
                "source": workload["source"],
                "id": workload["id"]
            })

        keyFeaturesString = ""
        sampleQueriesString = ""

        for sampleQuery in workload["sampleQueries"]:
            sampleQueriesString += sampleQuery + " "
        sampleQueriesString = normalize_text(sampleQueriesString)
        for keyFeature in workload["keyFeatures"]:
            keyFeaturesString += keyFeature + ". "
        keyFeaturesString = normalize_text(keyFeaturesString)

        workload["readmeVector"] = client.embeddings.create(input=normalize_text(readmeString), model=DEPLOYMENT_MODEL).data[0].embedding
        workload["titleVector"] = client.embeddings.create(input=normalize_text(workload["title"]), model=DEPLOYMENT_MODEL).data[0].embedding
        workload["keyFeaturesVector"] = client.embeddings.create(input=keyFeaturesString, model=DEPLOYMENT_MODEL).data[0].embedding
        workload["descriptionVector"] = client.embeddings.create(input=normalize_text(workload["description"]), model=DEPLOYMENT_MODEL).data[0].embedding
        workload["sampleQueriesVector"] = client.embeddings.create(input=normalize_text(sampleQueriesString), model=DEPLOYMENT_MODEL).data[0].embedding
        averageVector = average_vectors([workload["titleVector"], workload["keyFeaturesVector"], workload["descriptionVector"], workload["sampleQueriesVector"]])
        workload["averageAllInOneVector"] = averageVector.tolist()

        baseline_vector = client.embeddings.create(input=normalize_text(workload["title"]), model=DEPLOYMENT_MODEL).data[0].embedding

        workload["allInOneVector"] = client.embeddings.create(input=normalize_text(workload["title"] + keyFeaturesString + sampleQueriesString + workload["description"]), model=DEPLOYMENT_MODEL).data[0].embedding
        # Calculate cosine similarity between the average vector and combined vector
        similarity = cosine_similarity([averageVector], workload["allInOneVector"])
        similarity_all_in_one_vs_readme = cosine_similarity([workload["allInOneVector"]], workload["readmeVector"])
        similarity_readme_vs_baseline = cosine_similarity(baseline_vector, workload["readmeVector"])
        similarity_all_in_one_average_vs_baseline = cosine_similarity(baseline_vector, averageVector)
        similarity_all_in_one_vs_baseline = cosine_similarity([workload["allInOneVector"]], baseline_vector)

        print(f"Similarity between average vector and combined vector: {similarity}")
        print(f"Similarity between all in one vector and readme vector: {similarity_all_in_one_vs_readme}")
        print(f"Similarity between readme vector and baseline vector: {similarity_readme_vs_baseline}")
        print(f"Similarity between all in one average vector and baseline vector: {similarity_all_in_one_average_vs_baseline}")
        print(f"Similarity between all in one vector and baseline vector: {similarity_all_in_one_vs_baseline}")

        with open("scripts/generate_embeddings/similarities.json", "+a") as f:
            f.write(f"{similarity};{similarity_all_in_one_vs_readme};{similarity_readme_vs_baseline};{similarity_all_in_one_average_vs_baseline};{similarity_all_in_one_vs_baseline}\n")

    write_file("workloads/workloads_readme_vector.json", workloads)

    return

if __name__ == "__main__":
    generate_embeddings()