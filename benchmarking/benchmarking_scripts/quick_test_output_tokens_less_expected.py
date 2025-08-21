import os
import json
import glob

def find_invalid_summaries(directory="."):
    # Find all files with "_summary" in their filename
    summary_files = glob.glob(os.path.join(directory, "*_summary*"))

    bad_files = []

    for file in summary_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

            value = data.get("results_server_number_output_tokens_quantiles_p50", None)

            # Convert safely to float if it's not None
            if value is not None:
                try:
                    value = float(value)
                except ValueError:
                    continue  # skip if not numeric

                if value < 100:
                    bad_files.append((file, value))

        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")

    return bad_files


if __name__ == "__main__":
    directory = "./data/results/qwen_llama_switching_time/complete_fixed_version/20250814-165335.958040"  # change to your target directory
    results = find_invalid_summaries(directory)
    
    results.sort(key=lambda x: x[0])

    if results:
        print("Files with results_server_number_output_tokens_quantiles_p50 < 100:")
        for fname, val in results:
            print(f" - {fname}: {val}")
    else:
        print("✅ No files found with p50 < 100")