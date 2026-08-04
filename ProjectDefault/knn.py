import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    val_accs = []

    for name, impute_fn in [("user", knn_impute_by_user), ("item", knn_impute_by_item)]:
        print("\n=== {}-based collaborative filtering ===".format(name))
        val_accs = []
        for k in k_values:
            print("k = {}".format(k))
            val_accs.append(impute_fn(sparse_matrix, val_data, k))

        plt.figure()
        plt.plot(k_values, val_accs, marker="o")
        plt.xlabel("k")
        plt.ylabel("Validation accuracy")
        plt.title("{}-based collaborative filtering: validation accuracy vs. k".format(name))
        plt.xticks(k_values)
        plt.grid(True)
        plt.savefig("knn_{}.png".format(name), dpi=150, bbox_inches="tight")

        k_star = k_values[int(np.argmax(val_accs))]
        test_acc = impute_fn(sparse_matrix, test_data, k_star)
        print("[{}] k* = {}, test accuracy = {}".format(name, k_star, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
