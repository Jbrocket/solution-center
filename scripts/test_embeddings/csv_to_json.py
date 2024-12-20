# import csv, json

# workloads = []
# for i in range(2):
#     with open(f"export{i}.csv") as f:
#         reader = csv.DictReader(f)
#         workloads += list(reader)

# for workload in workloads:
#     workload.pop("\u00ef\u00bb\u00bf\"messageTimestamp\"")
#     workload["opened"] = workload["sourceType"]
#     workload.pop("sourceType")
#     workload["humanAnnotation"] = ""
    
# with open(f"queries.json", "w") as f:
#     json.dump(workloads, f, indent=4)