import argparse, json, requests

def main(root="."):
    workloads = json.load(open(f"{root}/workloads/workloads.json", "r"))
    keep_workloads = []

    for workload in workloads:
        res = requests.get(workload["source"])

        print(res.status_code, workload["source"])
        if res.status_code == 200:
            keep_workloads.append(workload)
    
    json.dump(keep_workloads, open(f"{root}/workloads/workloads.json", "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete unsupported workloads in workloads.json. Does so by looking at the source url.", 
                                      usage="%(prog)s --root ROOT")
    parser.add_argument("-r", "--root", help="Path to the root directory.", required=True)

    args = parser.parse_args()

    main(args.root)