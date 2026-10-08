import json
import sys

def run_regression_gate(baseline_path, current_report_path, threshold=0.05):
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(current_report_path, 'r') as f:
        current = json.load(f)
        
    # Assume report structure has 'recall' metric
    baseline_recall = baseline.get('recall', 0)
    current_recall = current.get('recall', 0)
    
    if current_recall < (baseline_recall - threshold):
        print(f"Regression detected: Current recall {current_recall} dropped below baseline {baseline_recall} - threshold {threshold}")
        sys.exit(1)
        
    print("Regression gate passed.")
    sys.exit(0)

if __name__ == "__main__":
    # Usage: python regression_gate.py <baseline_json> <current_json>
    if len(sys.argv) != 3:
        print("Usage: python regression_gate.py <baseline_json> <current_json>")
        sys.exit(1)
    
    run_regression_gate(sys.argv[1], sys.argv[2])
