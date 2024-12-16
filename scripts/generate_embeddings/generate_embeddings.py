import os, re, json, requests, logging
import numpy as np
import tiktoken
from openai import AzureOpenAI

LOG_FILENAME = 'info.log'

FILE_PATH = "workloads/workloads.json"
# DEPLOYMENT_MODEL = "text-embedding-ada-002"
DEPLOYMENT_MODEL = "text-embedding-3-small"
PAT = os.getenv("GITHUB_PERSONAL_PAT")
EMU_PAT = os.getenv("GITHUB_EMU_PAT")
HEADERS = {'Authorization': f'token {PAT}'}
EMU_HEADERS = {'Authorization': f'token {EMU_PAT}'}
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

def remove_contributor_section(text: str) -> str:
    patterns = [
        r"## Contributors.*",
        r"Contributors\n={4,}.*"
    ]
    
    combined_pattern = "|".join(patterns)
    
    new_text = re.sub(combined_pattern, "", text, flags=re.DOTALL)
    if text == new_text:
        logging.info("Contributor section not found\n")
    else:
        logging.info(f"Contributor section removed {new_text}\n")

    return new_text

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
                logging.info(f"Getting correct branch and readme from {github_url}")
                github_api = github_url.replace(raw_github, "https://api.github.com/repos")

                res = requests.get(f"{github_api}/contents", headers=HEADERS)
                if res.status_code != 200:
                    res = requests.get(f"{github_api}/contents", headers=EMU_HEADERS)
                contents = res.json()
                readme_file = [file['download_url'] for file in contents if 'readme' in file['name'].lower()][0]
                logging.info(f"Getting data from {readme_file}")

                response = requests.get(readme_file)
            except Exception as e:
                logging.error(f"Error getting readme file: {e}")
                raw_github = "https://raw.githubusercontent.com"
                github_url = github_url.replace("https://github.com", raw_github)
                readme_url = f"{github_url}/refs/heads/main/README.md"
                response = requests.get(readme_url)
    else: 
        logging.info(f"Getting data from {github_url}")
        readme_url = github_url
        response = requests.get(readme_url)

    return remove_contributor_section(response.text), response

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
    new_workloads = []
    for workload in workloads:
        logging.info(f"Processing workload: {workload['title']} {workload['source']}")
        readmeString, raw = get_readme(workload["source"])
        readmeString = normalize_text(readmeString)
        readmeEncode = tokenizer.encode(readmeString)
        readmeDecode = tokenizer.decode_tokens_bytes(readmeEncode)
        logging.info(f"Workload {workload['title']} decode length: {len(readmeDecode)}, {len(readmeDecode)<8192}")
        if len(readmeDecode) == 4:
            logging.error(f"Workload {workload['title']} no data")
            failed_workloads.append({
                "title": workload["title"],
                "source": workload["source"],
                "id": workload["id"]
            })
            continue

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

        new_workloads.append(workload)
        # Calculate cosine similarity between the average vector and combined vector
        similarity = cosine_similarity([averageVector], workload["allInOneVector"])
        similarity_all_in_one_vs_readme = cosine_similarity([workload["allInOneVector"]], workload["readmeVector"])
        similarity_readme_vs_baseline = cosine_similarity(baseline_vector, workload["readmeVector"])
        similarity_all_in_one_average_vs_baseline = cosine_similarity(baseline_vector, averageVector)
        similarity_all_in_one_vs_baseline = cosine_similarity([workload["allInOneVector"]], baseline_vector)

        logging.info(f"Similarity between average vector and combined vector: {similarity}")
        logging.info(f"Similarity between all in one vector and readme vector: {similarity_all_in_one_vs_readme}")
        logging.info(f"Similarity between readme vector and baseline vector: {similarity_readme_vs_baseline}")
        logging.info(f"Similarity between all in one average vector and baseline vector: {similarity_all_in_one_average_vs_baseline}")
        logging.info(f"Similarity between all in one vector and baseline vector: {similarity_all_in_one_vs_baseline}\n\n")

        with open("scripts/generate_embeddings/similarities.json", "+a") as f:
            f.write(f"{similarity};{similarity_all_in_one_vs_readme};{similarity_readme_vs_baseline};{similarity_all_in_one_average_vs_baseline};{similarity_all_in_one_vs_baseline}\n")

    write_file("workloads/workloads_readme_vector.json", new_workloads)
    write_file("workloads/failed_workloads.json", failed_workloads)

    return

if __name__ == "__main__":
    generate_embeddings()