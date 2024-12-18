import json, re, os, logging
import requests
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s')

VECTOR_FIELDS = {
    'r': "readmeVector", 
    't': "titleVector", 
    'k': "keyFeaturesVector", 
    'd': "descriptionVector", 
    's': "sampleQueriesVector", 
    'o': "allInOneVector", 
    'a': "averageAllInOneVector",
}

Configurations = {
    "ada": {
        "deployment_model": "text-embedding-ada-002",
        "index": "cn-workloads-index-with-vectors"
    },
    "embedding3": {
        "deployment_model": "text-embedding-3-small",
        "index": "cn-workloads-index-with-vectors-embedding3"
    },
}

fieldMap = {
    "id": ["id"],
    "title": ["title"],
    "url": ["source"],
    "filepath": ["source"],
    "content": ["description"],
    "author": ["author"],
    "tags": ["tags"],
    "sourceType": ["sourceType"],
    "deploymentOptions": ["deploymentOptions"],
    "deploymentConfig": ["deploymentConfig"],
    "tags":["tags"],
    "tech": ["tech"],
    "keyFeatures": ["keyFeatures"],
    "products": ["products"]
}
titleRegex = re.compile(r"title: (.*)\n")

def getIfString(doc, fieldName):
    try: 
        value = doc.get(fieldName)
        if isinstance(value, str) and len(value) > 0:
            return value
        return None
    except:
        return None

def get_truncated_string(string_value, max_length):
    return string_value[:max_length]

def getTitle(doc):
    max_title_length = 300
    title = getIfString(doc, 'title')
    if title:
        return get_truncated_string(title, max_title_length)
    else:
        title = getIfString(doc, 'content')
        if title: 
            titleMatch = titleRegex.search(title)
            if titleMatch:
                return get_truncated_string(titleMatch.group(1), max_title_length)
            else:
                return None
        else:
            return None

def getSourceType(doc):
    source_type = getIfString(doc, 'sourceType')
    return source_type

def getChunkId(doc):
    chunk_id = getIfString(doc, 'title')
    return chunk_id

def getSearchScore(doc):
    try:
        return doc['@search.score']
    except:
        return None

def getSearchRerankerScore(doc):
    try:
        return doc['@search.rerankerScore']
    except:
        return None

def getQueryList(query):
    try:
        config = json.loads(query)
        if not isinstance(config, list):
            config = [config]
        return config
    except Exception:
        return [query]
def getDeploymentOptions(doc):
    try:
        return doc['deploymentOptions']
    except:
        return None
def getDeploymentConfig(doc):
    try:
        return doc['deploymentConfig']
    except:
        return None

def getTags(doc):
    try:
        return doc['tags']
    except:
        return None

def getTech(doc):
    try:
        return doc['tech']
    except:
        return None

def getKeyFeatures(doc):
    try:
        return doc['keyFeatures']
    except:
        return None

def getProducts(doc):
    try:
        return doc['products']
    except:
        return None

def process_search_docs_response(docs):
    outputs = []
    for doc in docs:
        formattedDoc = {}
        for fieldName in fieldMap.keys():
            for fromFieldName in fieldMap[fieldName]:
                fieldValue = getIfString(doc, fromFieldName)
                if fieldValue:
                    formattedDoc[fieldName] = doc[fromFieldName]
                    break
        formattedDoc['title'] = getTitle(doc)
        formattedDoc['chunk_id'] = getChunkId(doc)
        formattedDoc['search_score'] = getSearchScore(doc)
        formattedDoc['search_rerankerScore'] = getSearchRerankerScore(doc)
        formattedDoc['sourceType'] = getSourceType(doc)
        formattedDoc['deploymentOptions'] = getDeploymentOptions(doc)
        formattedDoc['deploymentConfig'] = getDeploymentConfig(doc)
        formattedDoc['tags'] = getTags(doc)
        formattedDoc['tech'] = getTech(doc)
        formattedDoc['keyFeatures'] = getKeyFeatures(doc)
        formattedDoc['products'] = getProducts(doc)
        outputs.append(formattedDoc)
    return outputs

def get_query_embedding(query, endpoint, api_key, api_version, embedding_model_deployment):
    request_url = f"{endpoint}/openai/deployments/{embedding_model_deployment}/embeddings?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    request_payload = {
        'input': query
    }
    embedding_response = requests.post(request_url, json = request_payload, headers = headers, timeout=None)
    if embedding_response.status_code == 200:
        data_values = embedding_response.json()["data"]
        embeddings_vectors = [data_value["embedding"] for data_value in data_values][0]
        return embeddings_vectors
    else:
        raise Exception(f"failed to get embedding: {embedding_response.json()}")

def search_query_api(
    endpoint, 
    api_key,
    api_version, 
    index_name, 
    query_type, 
    query, 
    top_k,
    sourceFilter = None,
    embeddingModelName=None,
    semantic_configuration_name=None,
    vectorFields=VECTOR_FIELDS):
    request_url = f"{endpoint}indexes/{index_name}/docs/search?api-version={api_version}"
    request_payload = {
        'top': top_k,
        # 'queryLanguage': 'en-us'
    }
    logging.info(f"Querying endpoint: {request_url}")
    if query_type == 'simple':
        request_payload['search'] = query
        request_payload['queryType'] = query_type
    elif query_type == 'semantic':
        request_payload['search'] = query
        request_payload['queryType'] = query_type
        request_payload['semanticConfiguration'] = semantic_configuration_name
    elif query_type in ('vector', 'vectorSimpleHybrid', 'vectorSemanticHybrid'):
        if vectorFields and embeddingModelName:
            query_vector = get_query_embedding(
                query,
                "https://iaasexp-aml-workspace-aoai.openai.azure.com/",
                os.getenv("AZURE_OPENAI_API_KEY"),
                "2024-06-01",
                embeddingModelName
            )

            payload_vectors = [
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "exhaustive": True,
                    "fields": ", ".join(vectorFields),
                    "k": top_k
                }
            ]
            request_payload['vectorQueries'] = payload_vectors

        if query_type == 'vectorSimpleHybrid':
            request_payload['search'] = query
        elif query_type == 'vectorSemanticHybrid':
            request_payload['search'] = query
            request_payload['queryType'] = 'semantic'
            request_payload['semanticConfiguration'] = semantic_configuration_name
    else:
        raise Exception(f"unsupported query type: {query_type}")
    
    if sourceFilter:
        if sourceFilter.lower() == "azd":
            sourceFilter = "Azd"
        elif sourceFilter.lower() == "azuresamples":
            sourceFilter = "AzureSamples"
        request_payload['filter']=f"sourceType eq '{sourceFilter}'"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    retrieved_docs = requests.post(request_url, json = request_payload, headers = headers, timeout=None)
    logging.info(f"Query response status: {retrieved_docs.status_code}")
    if retrieved_docs.status_code == 200:
        return process_search_docs_response(retrieved_docs.json()["value"])
    else:
        raise Exception(f"failed to query search index : {retrieved_docs.json()}")

def search(queries: str, indexName: str, queryType: str, topK: int, semanticConfiguration: str, embeddingModelName: str, sourceFilter: str, vectorFields: list[str] = VECTOR_FIELDS):
    semanticConfiguration = semanticConfiguration if semanticConfiguration != "None" else None
    vectorFields = vectorFields if vectorFields != "None" else None
    embeddingModelName = embeddingModelName if embeddingModelName != None else None

    # Do search.
    allOutputs = [search_query_api(
        "https://workloads.search.windows.net/", 
        os.getenv("WorkloadsIndexApiKey"), 
        "2024-07-01", 
        indexName,
        queryType,
        query, 
        topK,
        sourceFilter, 
        embeddingModelName,
        semanticConfiguration,
        vectorFields) for query in getQueryList(queries)]

    includedOutputs = []
    while allOutputs and len(includedOutputs) < topK:
        for output in list(allOutputs):
            if len(output) == 0:
                allOutputs.remove(output)
                continue
            value = output.pop(0)
            if value not in includedOutputs:
                includedOutputs.append(value)
                if len(includedOutputs) >= topK:
                    break
    return includedOutputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search workloads index",
        usage="python3 test_prompt_flow.py [--vector_fields <rtkdsoa>] [--index <ada|emb3>] --query <query> [--output-mode <append|write>]"
    )
    
    parser.add_argument(
        "--index", 
        type=str, 
        help="Choose the index to use for the search, defaulted to index with ada-002 embeddings.", 
        choices=["ada", "emb3"],
        default="ada",
    )

    parser.add_argument(
        "--vector-fields", 
        type=str, 
        help="""Choose the vector fields to use for the search with a string map, defaulted to 'r'.
                Options map as follows:
                r - readmeVector
                t - titleVector
                k - keyFeatures
                d - descriptionVector
                s - sampleQueriesVector
                o - allInOneVector
                a - averageAllInOneVector""",
        default="r"
    )

    parser.add_argument(
        "--query", 
        type=str, 
        help="The query to search for in the index.",
        required=True
    )

    parser.add_argument(
        "--output-mode",
        type=str,
        help="The mode to use for writing to file, either append or write.",
        choices=["append", "write"],
        default="write"
    )

    args = parser.parse_args()

    selected_vector_fields = [VECTOR_FIELDS[char] for char in args.vector_fields if char in VECTOR_FIELDS]

    if args.index not in Configurations:
        config = "ada"
    else:
        config = args.index

    logging.info(f"Selected vector fields: {selected_vector_fields}")
    logging.info(f"Selected configuration: {config}")
    logging.info(f"Selected query: {args.query}")

    recommendations = search(
        embeddingModelName=Configurations[config]["deployment_model"],
        indexName=Configurations[config]["index"],
        queries=args.query,
        queryType="vector",
        semanticConfiguration="one-word",
        sourceFilter=None,
        topK=10,
        vectorFields=selected_vector_fields,
    )

    data = {args.query: recommendations}

    if args.output_mode == "append":
        if os.path.exists("workloads_recommendations.json"):
            with open("workloads_recommendations.json", "r") as f:
                existing_data = json.load(f)
                existing_data.update(data)
            with open("workloads_recommendations.json", "w") as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open("workloads_recommendations.json", "w") as f:
                json.dump(data, f, indent=4)
    else:
        with open("workloads_recommendations.json", "w") as f:
            json.dump(data, f, indent=4)