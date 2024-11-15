import json
import os

result_dir = "../../local/long_bench/pred"
print_dataset_name = True
for exp_name in os.listdir(result_dir):
    result_file = os.path.join(result_dir, exp_name, "result.json")
    if not os.path.exists(result_file):
        continue
    with open(result_file, 'r') as f:
        result = json.load(f)

    if print_dataset_name:
        # print(",,,".join([f"{dn}" for dn in result]))
        print(",".join([f"{dn}" for dn in result]))
        print_dataset_name = False
    row =f"{exp_name},"
    for dataset in result:
        scores = f"{result[dataset]},"
        # scores = result[dataset]
        # scores = f"{scores['0-4k']}\t{scores['4-8k']}\t{scores['8k+']}\t"
        row += scores

    print(row)