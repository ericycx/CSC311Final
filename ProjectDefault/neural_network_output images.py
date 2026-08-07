import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

#imports for matplotlib

import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        x = F.sigmoid(self.g(inputs))
        x = F.sigmoid(self.h(x))
        out = x #x.squeeze(1)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    training_accs = []
    training_losses = []
    val_accs = []
    val_losses = []

    epochs = []


    for epoch in range(0, num_epoch):
        epochs.append(epoch)
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            loss += (lamb / 2) * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        #train_acc = evaluate(model, zero_train_data, train_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
        


        val_loss = evaluate_validation_loss(model, zero_train_data, valid_data)

        #training_accs.append(train_acc)
        num_answered_questions = torch.sum(~torch.isnan(train_data)).item()
        training_losses.append(train_loss / num_answered_questions)
        val_losses.append(val_loss)
        val_accs.append(valid_acc)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label="Average Training Loss Per Question")
    plt.plot(epochs, val_losses, label="Average Validation Loss Per Question")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    #plt.plot(epochs, training_accs, label="Training Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

def evaluate_validation_loss(model, zero_train_data, valid_data):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for i, user_id in enumerate(valid_data["user_id"]):

            # Input vector for this student
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)

            # Reconstruct the student's answers
            output = model(inputs)

            # Validation question and true answer
            question_id = valid_data["question_id"][i]
            target = torch.tensor(valid_data["is_correct"][i], dtype=torch.float32)

            # Predicted probability for that question
            prediction = output[0][question_id]

            # Squared reconstruction error
            loss = (prediction - target) ** 2

            total_loss += loss.item()

    return total_loss / len(valid_data["user_id"])


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = 50
    
    # Set optimization hyperparameters.
    learning_rate = 0.01
    epoch = 50
    lamb = 1


    print("Training with k = " + str(k))
    print("Training with epoch = " + str(epoch))
    print("Training with lr = " + str(learning_rate))

    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, learning_rate, lamb, train_matrix, zero_train_matrix, valid_data, epoch)

    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print("Validation Accuracy =" + str(valid_acc))

    # Next, evaluate your network on validation/test data
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Test accuracy = " + str(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
