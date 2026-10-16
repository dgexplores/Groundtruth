from src.evaluator import evaluate_faithfulness, evaluate_retrieval


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "d"]
    assert evaluate_retrieval(retrieved, relevant) == 0.5

def test_faithfulness():
    context = ["apple"]
    assert evaluate_faithfulness("apple is red", context) == 1.0
    assert evaluate_faithfulness("banana is yellow", context) == 0.0
