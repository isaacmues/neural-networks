import torch
from torch import nn
from math import exp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):

        xy = torch.cat((x, y), 1)

        return A(x, y) + x * (1 - x) * y * (1 - y) * self.linear_sigmoid_stack(xy)


class coordinates(Dataset):
    def __init__(self, nx, ny):

        self.x = torch.linspace(0, 1, nx)
        self.y = torch.linspace(0, 1, ny)
        self.grid_x, self.grid_y = torch.meshgrid(self.x, self.y)
        self.length = nx * ny
        self.flat_x = torch.reshape(self.grid_x, (self.length, 1))
        self.flat_y = torch.reshape(self.grid_y, (self.length, 1))

    def __len__(self):

        return self.length

    def __getitem__(self, i):

        return [self.flat_x[i], self.flat_y[i]]

    def getall(self):

        return [self.flat_x, self.flat_y]

    def getgrid(self):

        return [self.grid_x, self.grid_y]


def f0(y):

    return y ** 3


def f1(y):

    return (y ** 3 + 1) * exp(-1)


def g0(x):

    return x * torch.exp(-x)


def g1(x):

    return (x + 1) * torch.exp(-x)


def A(x, y):

    e = exp(-1)
    a = (1 - x) * f0(y) + x * f1(y)
    a += (1 - y) * (g0(x) - x * e) + y * (g1(x) - (1 - x + 2 * x * e))

    return a


def psi_a(x, y):

    return torch.exp(-x) * (x + y ** 3)


def pde(sol, x_, y_):

    x = torch.nn.Parameter(x_, requires_grad=True)
    y = torch.nn.Parameter(y_, requires_grad=True)
    f = sol(x, y)
    dfdx, dfdy = torch.autograd.grad(f, (x, y), torch.ones_like(x), create_graph=True)
    d2fdx2 = torch.autograd.grad(dfdx, x, torch.ones_like(x), create_graph=True)[0]
    d2fdy2 = torch.autograd.grad(dfdy, y, torch.ones_like(x), create_graph=True)[0]

    error = d2fdx2 + d2fdy2 - torch.exp(-x) * (x - 2 + y ** 3 + 6 * y)

    return torch.mean(torch.square(error))


def training_loop(dataloader, sol, loss_fn, optimizer):

    total_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):

        loss = loss_fn(sol, x, y)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:

            print(f"Batch: {batch:>d} Loss: {loss:.2E}")

    print(f'--------------------------------\nTotal loss: {total_loss:.2E}')


epochs = 100
nx, ny = 50, 50
xy = coordinates(nx, ny)
psi_t = NeuralNetwork()
dataloader = DataLoader(xy, batch_size=20, shuffle=True)
optimizer = torch.optim.Adam(psi_t.parameters(), lr=0.01)

for t in range(epochs):

    print(f"\nEpoch {t+1}\n--------------------------------")
    training_loop(dataloader, psi_t, pde, optimizer)


x, y = xy.getall()
neural_solution = psi_t(x, y)
error = torch.abs(neural_solution - psi_a(x, y))
error = torch.max(error)
print(f"\n>> Maximum error {error:.2E} <<")

x, y = xy.getgrid()
x = torch.reshape(x, (nx, ny)).numpy()
y = torch.reshape(y, (nx, ny)).numpy()
neural_solution = torch.reshape(neural_solution, (nx, ny)).detach().numpy()

plt.contourf(x, y, neural_solution)
plt.contour(x, y, neural_solution, colors="black")
plt.show()
