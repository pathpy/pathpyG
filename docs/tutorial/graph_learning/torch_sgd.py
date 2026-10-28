import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # pyTorch, autograd, and Stochastic Gradient Descent

    *August 4 2026*
    *Training Workshop: Introduction to Deep Graph Learning*
    *Ingo Scholtes, CAIDAS, Julius-Maximilians-Universität Würzburg (JMU), Germany*

    For the simple example discussed in the previous notebook, the parameters learned by the perceptron model (using gradient descent minimization of the loss function) are identical to the parameters that we obtained by fitting the logistic regression model (using gradient ascent maximization of the likelihood function). This is one of the simplest possible examples of a neural network, where the "network" actually consists of a single neuron that maps one or more inputs to a single output.

    We now implement this simple perceptron classifier in the popular neural network library `pytorch`. We also introduce the `autograd` feature, which is used to automatically calculate gradients of loss functions.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats
    import seaborn as sns
    import torch
    from scipy.special import expit

    plt.style.use('default')
    sns.set_style("whitegrid")
    return expit, np, pd, plt, scipy, sns, torch


@app.cell
def _(np):
    def response(x, boundary=0.2, prob_1=1):
        """Generate a probabilistic binary response based on a threshold."""
        if x > boundary:
            if np.random.random_sample() < prob_1:
                return 1
            else:
                return 0
        else:
            if np.random.random_sample() < 1-prob_1:
                return 1
            else:
                return 0

    return (response,)


@app.cell
def _(np, pd, response, scipy, sns):
    x = 2*scipy.stats.uniform.rvs(size=100)-1
    y = np.array([ response(x[i]) for i in range(len(x)) ])

    data = pd.DataFrame({'x': x, 'y_class': y})

    sns.scatterplot(x='x', y="y_class", data=data, hue='y_class')
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The machine learning library `pytorch` heavily builds on the `torch.tensor` data structure, i.e. all inputs and outputs are assumed to be tensors, i.e. multi-dimensional arrays or higher-dimensional generalizations of vectors and matrices.

    For our example, we convert our data to a one-dimensional tensor, i.e. a vector.
    """)
    return


@app.cell
def _(data, torch):
    train_x = torch.tensor(data['x'].values, dtype=torch.float32)
    train_y = torch.tensor(data['y_class'].values, dtype=torch.float32)

    print(train_x)
    print(train_y)
    return train_x, train_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now implement a simple perceptron model. All models in `pytorch` are derived from the base class `torch.nn.Module`. Such a module can consist of multiple "layers" of neural networks and we can control how those layers are interconnected and what activation functions are used as data pass between the layers. By implementing the `forward` function, we control what happens as the data (i.e. our features) are passed through the (possibly multiple) layers of our model.

    For our example with linearly separable classes, a single layer that consists of a single neuron with one input and one output is sufficient. In the forward function, we apply a linear transformation $f(\vec{x}) = \beta_0 + \beta_1 x_1 + \ldots$ of our input feature. We further "forward" a value that is transformed by a sigmoid "activation" function, i.e. we perform a logistic transformation of the linear combination of features which defines the output of our "network", i.e. the class probability.

    Conveniently, we do not need to implement the linear transformation of the feature(s) (based on the slope and the bias parameter) ourselves. We can simply use the `torch.nn.Linear` class, which implements a single layer perceptron model and automatically includes the necessary weight parameters (i.e. the bias and slope parameters for all perceptrons and input dimensions). With this, implementing a perceptron in `pytorch` is as simple as:
    """)
    return


@app.cell
def _(torch):
    class Perceptron(torch.nn.Module):
        """A single-layer perceptron model."""

        def __init__(self):
            """Initialize the model parameters."""
            super(Perceptron, self).__init__()
            self.linear = torch.nn.Linear(in_features=1, out_features=1, bias=True)
        
        def forward(self, x):
            """Compute the output of the perceptron."""
            # we use a logistic transformation to output class probabilities
            # Note that torch.sigmoid is an alias for this function 
            return torch.special.expit(self.linear(x))

    return (Perceptron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now initialize our model, choose a loss function that is used to optimize the model parameters, and pick one of the optimization algorithms implemented in `pytorch`. To understand how these optimization algorithms work, we need to explain some basics of the `autograd` module in `pytorch`. In our previous notebook, we have implemented a Gradient Descent optimization ourselves. For this, we used the gradients (calculated based on partial derivatives) that we calculated in the lecture based on the L2 loss function. We can only use those gradients for the special case of a model using a single perceptron performing a binary classification based on the logistic activation function and the L2 loss function.

    However, in `pytorch` we can use arbitrarily complex, multi-layer models with various activation functions, loss functions, etc. How can we compute the gradients that we need to iteratively update the model parameters during the learning step? For this, we can use the `autograd` feature of `pytorch`. A detailed explanation of the fundamental of `autograd` is available [here](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html). Simply speaking, thanks to the `autograd` feature all tensors trace the history of all computations, which allows to efficiently and automatically compute the local gradients that are needed by optimization algorithms. In each step of the optimization algorithm, these gradients then allow to nudge model parameters in the right direction. A simple example that illustrates the automatic gradient computation can be found [here](https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html).

    We can try this as follows: We create a tensor, specifying that we want `torch` to keep track of all operations that are performed on the tensor.
    """)
    return


@app.cell
def _(torch):
    x_1 = torch.tensor([42.0], requires_grad=True)
    print(x_1.detach().numpy())
    print(x_1.grad)
    print(x_1.grad_fn)
    return (x_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now compute a new tensor `y` by adding a value to the elements of tensor `x`. We find that this lead to the assignment of the `grad_fn` property, which stores that tensor `y` was obtained by adding a value to tensor `x`.
    """)
    return


@app.cell
def _(x_1):
    y_1 = x_1 + 0.5
    print(y_1.detach().numpy())
    print(y_1.grad)
    print(y_1.grad_fn)
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we assign a new tensor `z` my multiplying a value to tensor `y`, the `grad_fn` property stores that tensor `z` was obtained by multiplying a value to tensor `y`.
    """)
    return


@app.cell
def _(y_1):
    z = y_1 * 0.1
    print(z.detach().numpy())
    print(z.grad_fn)
    return (z,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You see that by this calculations we have created a computation graph of tensors connected via the `grad_fn` properties. The output tensor `z` points to `y` and `y` points to `x`, which is our input tensor. We can call the `backward()` function of the output tensor to compute the gradients, i.e. the partial derivative of the input tensor.
    """)
    return


@app.cell
def _(x_1, z):
    z.backward()
    print('gradient of x =', x_1.grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This gradient can be understood if we consider how the output $z=f(x)$ is calculated based on the input $x$. We have implemented the following expression:

    $ f(x) = 0.1*(x+0.5) = 0.1 x + 0.05$

    The derivative of this function (i.e. the gradient w.r.t x) is

    $ f'(x) = 0.1$

    and thus $f'(42) = 0.1$

    For the following example, for $x = 42$ we expect a gradient value of $171$:

    $ f(x) = 2 \cdot x^2 + 3 \cdot x + 5 $

    $ f'(x) = 4 \cdot x + 3 $

    $ f'(42) = 4 \cdot 42 + 3 = 171 $
    """)
    return


@app.cell
def _(torch):
    x_2 = torch.tensor([42.0], requires_grad=True)
    y_2 = 2 * x_2 ** 2 + 3 * x_2 + 5
    y_2.backward()
    print(x_2.grad.detach().numpy())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using this convenient `autograd` feature, in `pytorch` neural networks can be trained based on backpropagation, i.e. starting from the output that we wish to minimize (i.e. the loss function) we run the backpropagation and obtain gradients for our model parameters, which can be used in the iterative optimization. This works as follows:

    1.) Using the current parameter values, we calculate a **loss function**, i.e. a function that quantifies the difference between the target variable (i.e. our ground truth in the training data) and the current output of our model. Our goal is to minimize this loss function for our training data.

    2.) We propagate the loss function backwards through our model, i.e. starting from the output we pass the loss backwards to the inputs. In the process, we calculate the gradients of all model parameters, which will be used in the optimization step.

    For a single-layer perceptron, the **backpropagation** algorithm corresponds to the [delta rule](https://en.wikipedia.org/wiki/Delta_rule), which yields the partial derivatives (i.e. gradient) for a perceptron with activation function \sigma(x), i.e.

    $ \Delta \beta_{j}= \eta (\hat{y}_{j}-y_{j})\sigma'(h_{j})x_{i} $

    where $\alpha$ is the learning rate $y_i$ is the model output and $\hat{y}_i$ is the target variable, i.e. the ground truth classes.

    3.) We finally perform the actual optimization step, i.e. using the learning rate, we nudge the model parameters in the direction of the gradient.

    For the sigmoid activation function $\sigma(x) = \frac{1}{e+e^{-x}}$ we obtain the gradients

    $ \Delta \beta_{j}=  (\hat{y}_{s}-y_{s})y_s(1-y_s)x_{sj} $

    We are now ready to train our first `pytorch` model. We follow the same approach as before, i.e. we create the model, set the hyperparameters and initialize an optimizer. Here we use the implementation of stochastic gradient descent included in `pytorch`. This actually follows the same approach as our implementation in the previous notebooks, however here the gradients are (automatically) computed such that we minimize a loss function (rather than maximizing the likelihood function).

    The name "stochastic" gradient descent refers to the fact that - rather than calculating the gradient for the whole data set it is estimated based on a small sample (i.e. for the batch size used in each iteration).
    """)
    return


@app.cell
def _(Perceptron, torch, train_x, train_y):
    model = Perceptron()
    loss_func = torch.nn.MSELoss()
    x_3 = train_x[0].reshape(1)
    label = train_y[0].reshape(1)
    y_3 = model(x_3)
    print(y_3)
    _loss = loss_func(y_3, label)
    print(_loss)
    _loss.backward()
    print('bias gradient =', model.linear.bias.grad)
    print('weight gradients =', model.linear.weight.grad)
    return label, loss_func, model, x_3, y_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now compare this to a analytical derivation of the gradient, which we can calculate as (verify this as an home exercise!):

    $\nabla L = (\frac{\partial L}{\partial \beta_0}, \frac{\partial L}{\partial \beta_1}) = \left(2 \cdot (\hat{y}_i-y_i)\cdot y_i\cdot (1-y_i), 2 \cdot (\hat{y}_i-y_i)\cdot y_i\cdot (1-y_i)\cdot x_i \right)$
    """)
    return


@app.cell
def _(label, x_3, y_3):
    y_val = y_3.detach().numpy()[0]
    x_val = x_3.detach().numpy()[0]
    label_val = label.detach().numpy()[0]
    beta_0_grad = -2 * (label_val - y_val) * y_val * (1 - y_val)
    print(beta_0_grad)
    beta_1_grad = -2 * (label_val - y_val) * y_val * (1 - y_val) * x_val
    print(beta_1_grad)
    return


@app.cell
def _(model, torch):
    # the number of epochs gives the number of times we run the 
    # optimization algorithm on all examples in our training set
    epochs = 500

    # the learning rate controls how much parameters are 
    # changed (based on the gradients) for each training sample
    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return epochs, optimizer


@app.cell
def _(epochs, loss_func, model, optimizer, plt, train_x, train_y):
    model.train()
    errors = []
    for epoch in range(epochs):
        error = 0
        for i in range(len(train_x)):
            x_4 = train_x[i].reshape(1)
            label_1 = train_y[i].reshape(1)
            optimizer.zero_grad()
            output = model(x_4)
            _loss = loss_func(output, label_1)
            _loss.backward()
            optimizer.step()
            error = error + _loss
        errors.append(error.detach().numpy())
    plt.plot(range(epochs), errors)
    print('bias =', model.linear.bias.data)
    print('weight =', model.linear.weight.data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that the loss function decreases as the SGD optimizer moves the model parameters along the automatically calculated gradients. Note that those gradients are calculated when we call the `backward` function of the tensor `loss` (the output we wish to minimize). Using `autograd`, this will calculate the gradients of the model parameters. Note that we have to reset the previously calculated gradients for each iteration of the algorithm, i.e. we use gradients calculated for each data point separately.
    """)
    return


@app.cell
def _(model):
    params = [x.detach().numpy()[0] for x in model.parameters()]
    print(params)
    return (params,)


@app.function
def decision(output):
    """Dichotomize output based on threshold of 0.5."""
    if output>0.5:
        return 1
    else:
        return 0


@app.cell
def _(data, expit, model, params, sns, torch):
    model.eval()

    xx = data['x'].values
    data['prediction'] = [ decision(model.forward(torch.tensor([x], dtype=torch.float32)).detach().numpy()[0]) for x in xx]
    print(data)
    sns.relplot(x='x', y="y_class", hue='prediction', data=data, alpha=0.5)
    sns.lineplot(x=xx, y=expit(params[1] + xx * params[0][0]).ravel(), color='red', linewidth=1)
    return


if __name__ == "__main__":
    app.run()
