import json
import difflib

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def run_evaluation(data):
    results = []
    for item in data:
        # Mocking model output
        input_text = item['input']
        expected = item['expected_output']
        
        # In a real scenario, call the LLM here
        model_output = simulate_llm(input_text)
        
        match = model_output.strip() == expected.strip()
        results.append({
            "input": input_text,
            "expected": expected,
            "actual": model_output,
            "match": match
        })
    return results

def simulate_llm(input_text):
    # Hardcoded simulation for runner test
    if "capital of France" in input_text: return "Paris"
    if "2 + 2" in input_text: return "4"
    if "GIL" in input_text: return "Python"
    if "Hermes Agent" in input_text: return "Nous Research"
    if "1991" in input_text: return "1991"
    if "first version of Python" in input_text: return "1991"
    return "Unknown"

if __name__ == "__main__":
    dataset_path = "/Users/dgsmacbook/AI_Engineering_Portfolio/Groundtruth/evals/golden_dataset.json"
    data = load_data(dataset_path)
    results = run_evaluation(data)
    
    for r in results:
        print(f"Input: {r['input']} | Match: {r['match']}")
