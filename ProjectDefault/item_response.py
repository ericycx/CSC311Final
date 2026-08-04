from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    x = theta[user_ids] - beta[question_ids]
    log_lklihood = np.sum(is_correct * x - np.log(1 + np.exp(x)))
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    user_ids = np.array(data["user_id"])
    question_ids = np.array(data["question_id"])
    is_correct = np.array(data["is_correct"])
    x = theta[user_ids] - beta[question_ids]
    sig = sigmoid(x)
    # theta grad desc
    grad_theta = np.zeros_like(theta)
    np.add.at(grad_theta, user_ids, is_correct - sig)
    theta = theta + lr * grad_theta
    # do it agian for beta
    x = theta[user_ids] - beta[question_ids]
    sig = sigmoid(x)
    # beta grad desc
    grad_beta = np.zeros_like(beta)
    np.add.at(grad_beta, question_ids, sig - is_correct)
    beta = beta + lr * grad_beta
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # vector space allocation
    num_students = max(max(data["user_id"]), max(val_data["user_id"])) + 1
    num_questions = max(max(data["question_id"]), max(val_data["question_id"])) + 1
    theta = np.zeros(num_students)
    beta = np.zeros(num_questions)
    val_acc_lst = []
    train_negloglike_lst = []
    val_negloglike_lst = []
    for i in range(iterations):
        neg_log_like = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_log_like = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_negloglike_lst.append(-neg_log_like)
        val_negloglike_lst.append(-val_neg_log_like)
        print("Negloglike: {} \t Score: {}".format(neg_log_like, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta, val_acc_lst, train_negloglike_lst, val_negloglike_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # lr = 0.01 overfit
    lr = 0.005
    iterations = 25
    # iterations = 50

    theta, beta, val_acc_lst, train_nllk_lst, val_nllk_lst = irt(
        train_data, val_data, lr, iterations
    )
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Hyperparameters: lr={lr}, iterations={iterations}")
    plt.figure(figsize=(8, 5))
    plt.plot(range(iterations), train_nllk_lst, label="Training Log-Likelihood")
    plt.plot(range(iterations), val_nllk_lst, label="Validation Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Training Curve: Log-Likelihood vs. Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("irt_training_curve.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    sorted_indices = np.argsort(beta)
    j1 = sorted_indices[len(sorted_indices) // 4]      # easy question
    j2 = sorted_indices[len(sorted_indices) // 2]      # medium question
    j3 = sorted_indices[3 * len(sorted_indices) // 4]  # hard question

    theta_range = np.linspace(-4, 4, 200)

    plt.figure(figsize=(8, 5))
    for j, label in [(j1, f"Q{j1} (β={beta[j1]:.2f})"),
                      (j2, f"Q{j2} (β={beta[j2]:.2f})"),
                      (j3, f"Q{j3} (β={beta[j3]:.2f})")]:
        prob = sigmoid(theta_range - beta[j])
        plt.plot(theta_range, prob, label=label)

    plt.xlabel("Student Ability (θ)")
    plt.ylabel("P(correct)")
    plt.title("Item Characteristic Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("irt_item_curves.png")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
