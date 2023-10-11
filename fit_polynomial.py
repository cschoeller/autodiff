from random import shuffle

import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import seaborn as sns

from module import Module
from variable import Variable


_POLY_DEGREE = 5
_EPOCHS = 100
_NUM_POINTS = 20000

class PolynomialModel(Module):

    def __init__(self, degrees):
        super().__init__()
        self._degrees = degrees
        self.coefficients = [Variable(coeff) for coeff in nprand.normal(size=self._degrees + 1)]

    def __call__(self, x):
        out = Variable(0)
        for i, coeff in enumerate(self.coefficients):
            out += coeff * x**(self._degrees - i)
        return out

def make_dataset(num_points):

    def _ground_truth_polynome(x):
        return  (0.988 * x**5 - 4.96 * x**4 
                + 4.978 * x**3 + 5.015 * x**2
                - 6.043 * x - 1)
        
    X = np.arange(start=-2., stop=4., step=(6. / num_points)) + np.random.normal(loc=0., scale=0.08, size=num_points)# noise
    Y = np.array([_ground_truth_polynome(val) for val in X]) + np.random.normal(loc=0., scale=5.,size=num_points)
    return X, Y

def dataloader(dataset, batch_size):
    X, Y = dataset
    indices = list(range(len(X)))
    shuffle(indices)
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = np.array(indices[batch_start:batch_start + batch_size])
        if len(batch_indices) < batch_size: # drop last
            return

        yield X[batch_indices], Y[batch_indices]

def plot_results(model, dataset):
    X, Y = dataset
    sns.scatterplot(x=X, y=Y, s=5, alpha= 0.05)
    model_in = np.arange(np.min(X), np.max(X), step=0.1)
    model_out = np.array([model(x).val for x in model_in])
    sns.lineplot(x=model_in, y=model_out, color="red")
    plt.show()

def fit(model, dataset):
    lr = 1e-6
    batch_size = 16
    
    for epoch in range(_EPOCHS):
        epoch_losses = []
        for batch in dataloader(dataset, batch_size):
            Xb, Yb = batch
            preds = [model(xb) for xb in Xb]
            loss = 1 / batch_size * sum([(y - pred)**2 for y, pred in zip(Yb, preds)]) # mse loss
            epoch_losses.append(loss.val)
            
            model.zero_grad()
            loss.backward()
            
            # update model params
            for param in model.parameters:
                param.val -= lr * param.grad
                
        print(f"epoch: {epoch + 1}, loss: {sum(epoch_losses) / len(epoch_losses)}")
                    
def main():
    model = PolynomialModel(degrees=_POLY_DEGREE)
    dataset = make_dataset(_NUM_POINTS)
    print("Starting to fit polynomial:")
    fit(model, dataset)
    plot_results(model, dataset)

if __name__ == "__main__":
    main()