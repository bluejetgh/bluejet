import torch
import matplotlib.pyplot as plt


class LinearRegression(torch.nn.Module):
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.w1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.optimizer = torch.optim.SGD(
            [self.w0, self.w1],
            lr=self.learning_rate
        )

        self.loss_function = torch.nn.MSELoss()

        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x):
        return self.w0 + self.w1 * x

    def fit(self, x_train, y_train, x_test, y_test):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            y_pred = self.forward(x_train)

            loss = self.loss_function(y_pred, y_train)

            loss.backward()

            self.optimizer.step()

            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        with torch.no_grad():
            y_test_pred = self.forward(x_test)

            ss_res = torch.sum((y_test - y_test_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)

            r2 = 1 - (ss_res / ss_tot)

        print("R^2:", r2.item())

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def analysis_plot(self, x, y):
        with torch.no_grad():
            y_pred = self.forward(x)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].scatter(x.numpy(), y.numpy())
        axes[0, 0].plot(x.numpy(), y_pred.numpy())
        axes[0, 0].set_title("Original Data and Regression Line")
        axes[0, 0].set_xlabel("BCR")
        axes[0, 0].set_ylabel("Annual Production")

        axes[0, 1].plot(self.w0_history)
        axes[0, 1].set_title("w0 During Training")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("w0")

        axes[1, 0].plot(self.w1_history)
        axes[1, 0].set_title("w1 During Training")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("w1")

        axes[1, 1].plot(self.loss_history)
        axes[1, 1].set_title("Loss During Training")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")

        plt.tight_layout()
        plt.show()
