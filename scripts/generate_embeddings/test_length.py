import json

def read_file(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_file(file_path: str, data: list[dict]):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    return 0

def create_files():
    workloads = read_file("workloads/workloads_with_vectors.json")
    new_workloads = []

    num = 0
    current_length = 0
    for workload in workloads:
        current_length += len(str(workload))
        if current_length > 2_500_000:
            write_file(f"workloads/workloads_vectors/workloads_vector_{num}.json", new_workloads)
            new_workloads = []
            num += 1
            current_length = 0

        new_workloads.append(workload)
        print(current_length)

    if new_workloads:
        write_file(f"workloads/workloads_vector_{num}.json", new_workloads)

if __name__ == "__main__":
    create_files()