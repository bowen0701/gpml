from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(71)


class SoftmaxRegression(nn.Module):
    """PyTorch implementation of Softmax Regression."""

    def __init__(self, n_targets=10, batch_size=64, lr=0.01, n_epochs=1000):
        super(SoftmaxRegressionTorch, self).__init__()
        self.n_targets = n_targets
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

    def get_data(self, X_train, y_train, shuffle=True):
        """Get dataset and information."""
        self.X_train = X_train
        self.y_train = y_train

        # Get the numbers of examples and inputs.
        self.n_examples, self.n_inputs = self.X_train.shape

        if shuffle:
            idx = list(range(self.n_examples))
            random.shuffle(idx)
            self.X_train = self.X_train[idx]
            self.y_train = self.y_train[idx]

    def _create_model(self):
        """Create logistic regression model."""
        self.net = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_targets),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.net(x)
        return y

    def _create_loss(self):
        """Create cross entropy loss."""
        self.criterion = nn.CrossEntropyLoss()

    def _create_optimizer(self):
        """Create optimizer by stochastic gradient descent."""
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def build(self):
        """Build model, loss function and optimizer."""
        self._create_model()
        self._create_loss()
        self._create_optimizer()

    def _fetch_batch(self):
        """Fetch batch dataset."""
        idx = list(range(self.n_examples))
        for i in range(0, self.n_examples, self.batch_size):
            idx_batch = idx[i:min(i + self.batch_size, self.n_examples)]
            yield (self.X_train.take(idx_batch, axis=0), 
                   self.y_train.take(idx_batch, axis=0))

    def fit(self):
        """Fit model."""
        for epoch in range(1, self.n_epochs + 1):
            total_loss = 0
            for X_train_b, y_train_b in self._fetch_batch():
                # Convert to Tensor from NumPy array and reshape ys.
                X_train_b, y_train_b = (
                    torch.from_numpy(X_train_b), torch.from_numpy(y_train_b))

                y_pred_b = self.net(X_train_b)
                loss = self.criterion(y_pred_b, y_train_b.long())
                total_loss += loss * X_train_b.shape[0]

                # Zero grads, performs backward pass, and update weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: training loss: {total_loss / self.n_examples}')

    def get_coeff(self):
        """Get model coefficients."""
        # Detach var which require grad.
        return (self.net[0].bias.detach().numpy(),
                self.net[0].weight.detach().numpy())

    def predict(self, X):
        """Predict for new data."""
        with torch.no_grad():
            X_ = torch.from_numpy(X)
            return self.net(X_)


def main():
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression as LogisticRegressionSklearn

    import sys
    sys.path.append('../numpy/')
    from metrics import accuracy

    # Read breast cancer data.
    mnist_data = load_digits()
    X, y = mnist_data.data, mnist_data.target

    # Split data into training and test datasets.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=71, shuffle=True, stratify=y)

    # Feature engineering for standardizing features by min-max scaler.
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train_raw)
    X_test = min_max_scaler.transform(X_test_raw)

    # Convert arrays to float32.
    X_train, X_test, y_train, y_test = (
        np.float32(X_train), np.float32(X_test), 
        np.float32(y_train), np.float32(y_test)
    )

    # Train PyTorch logistic regression model.
    # Fit PyTorch Logistic Regression.
    softmax_torch = SoftmaxRegressionTorch(n_targets=10, batch_size=64, lr=0.5, n_epochs=1000)
    softmax_torch.get_data(X_train, y_train, shuffle=True)
    softmax_torch.build()
    softmax_torch.fit()

    p_train_ = softmax_torch.predict(X_train)
    print('Training accuracy: {}'.format(accuracy(p_train_.argmax(dim=1).numpy(), y_train)))
    p_test_ = softmax_torch.predict(X_test)
    print('Test accuracy: {}'.format(p_test_.argmax(dim=1).numpy(), y_test))

    # Benchmark with sklearn's Logistic Regression.
    print('Benchmark with logreg in Scikit-Learn.')
    softmax_sk = LogisticRegressionSklearn(C=1e4, multi_class='multinomial', max_iter=500)
    softmax_sk.fit(X_train, y_train.reshape(y_train.shape[0], ))
    
    y_train_ = softmax_sk.predict(X_train)
    print('Training accuracy: {}'.format(accuracy(y_train_, y_train)))
    y_test_ = softmax_sk.predict(X_test)
    print('Test accuracy: {}'.format(accuracy(y_test_, y_test)))


if __name__ == '__main__':
    main()
