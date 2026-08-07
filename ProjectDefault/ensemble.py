import numpy as np
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_csv,
)
from item_response import irt, sigmoid


def bootstrap_sample(data):
    """Create a bootstrapped sample of the data.

    :param data: dict with keys user_id, question_id, is_correct which are lists 
    :return: data sampled with replacement still a dict with same keys
    """
    n = len(data["user_id"])
    indices = np.random.choice(n, size=n, replace=True)
    bootstrapped = {
        'user_id': [],
        'question_id': [],
        'is_correct': []
    }
    for i in indices:
        bootstrapped['user_id'].append(data['user_id'][i])
        bootstrapped['question_id'].append(data['question_id'][i])
        bootstrapped['is_correct'].append(data['is_correct'][i])
        
    return bootstrapped

def irt_predict_probs(data, theta, beta):
    """Return predicted probabilities for every entry in the data.

    :param data: dict with user_id, question_id
    :param theta: student ability vector
    :param beta: question difficulty vector
    :return: array of probabilities
    """
    return np.array([
        sigmoid(theta[data["user_id"][i]] - beta[data["question_id"][i]])
        for i in range(len(data["user_id"]))
    ])


def ensemble_acc(data, avg_probs):
    """Calculates the ensemble accuracy.

    :param data: dict with is_correct
    :param avg_probs: array of predicted probabilities
    :return: float accuracy
    """
    preds = (avg_probs >= 0.5).astype(int)
    correct = np.array(data["is_correct"])
    return np.sum(preds == correct) / len(correct)


def main():
    np.random.seed(42)

    train_data = load_train_csv("./data")
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    lr = 0.005
    iterations = 25
    num_models = 3

    # 3 IRT samples each with their own bootstrapped data set
    models = []
    for i in range(num_models):
        print(f"\n Model {i + 1}")
        boot_data = bootstrap_sample(train_data)
        theta, beta, val_acc_lst, _, _ = irt(boot_data, valid_data, lr, iterations)
        models.append((theta, beta))
        test_acc_i = ensemble_acc(test_data, irt_predict_probs(test_data, theta, beta))
        print(f"Model {i + 1} val accuracy: {val_acc_lst[-1]:.4f}  test accuracy: {test_acc_i:.4f}")
        
    # print(models)

    # Average predictions on validation set
    val_probs = np.zeros(len(valid_data["user_id"]))
    for theta, beta in models:
        val_probs += irt_predict_probs(valid_data, theta, beta)
    # Val_probs is the sum of all prediction probabilities
    val_probs /= num_models

    val_acc = ensemble_acc(valid_data, val_probs)
    print(f"\nEnsemble Validation Accuracy: {val_acc:.4f}")

    # Average predictions on test set
    test_probs = np.zeros(len(test_data["user_id"]))
    for theta, beta in models:
        test_probs += irt_predict_probs(test_data, theta, beta)
    # test_probs is the sum of all test probabilities
    test_probs /= num_models

    test_acc = ensemble_acc(test_data, test_probs)
    print(f"Ensemble Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
