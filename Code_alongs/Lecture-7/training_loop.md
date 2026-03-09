import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup (The "Brain" and the "Goal")
# We always need to have a model. We typically define
# many of its parts (input_size, hidden layers
# (number of nodes AND number of layers), output size)
# Sometimes we use other people's models.
model = MyModel()

# Then we define how our loss should be calculated
# Typically CrossEntropyLoss for classification
# Mean square error for regression
# There are many others
# Loss is therefore: How wrong our model is (amount of error)
# Loss is what our model tries to minimize
criterion = nn.CrossEntropyLoss() # or MSELoss for regression

# The next step is typically to define an optimizer.
# That is: a module that can optimize our loss function for us
# BASELINE: is typically Stochastic Gradient Descent
# BUT: We often start with ADAM anyway, it is an advanced SGD
# Learning Rate is one of, perhaps the MOST IMPORTANT hyperparameters
# One that is too large can lead to us not finding the optimum, one that is too small
# can risk local minima, or that we do not converge at all
# OFTEN typical values are some negative power of ten, that is (0.1, 0.01, 0.001 etc.)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate is something that can have a big impact on the result
# It is often something we experiment with. There are specific
# libraries that help us optimize e.g. learning rate
# BUT: we can make our own loop, as below
# Optuna for example
# lr_list = [0.1, 0.01, 0.001]

# for lr in lr_list: 
    

# 2. The Training Loop (The "Practice")
# The training loop aims to train our network (our model)
# on our data. We do this by sending the data in batches multiple times
# and comparing the results with the correct answers (= calculating the loss),
# and gradually adjusting the weights.

# we do this process in epochs, that is a certain number of times
# (we decide how many)
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # A. Reset: Clear previous gradients
        # Vi säger till optimizern att glömma vad den gjorde förra varvet
        # (det kan gå snett annars)
        optimizer.zero_grad()
        
        # B. Forward: Build the graph & get prediction
        # We put our data into the model and produce some outputs
        # (the data makes one forward pass through the model)
        outputs = model(inputs)

        # We put our outputs and targets into our Loss model
        # to calculate our loss function (how much error our model had)
        # We are therefore comparing the model's predictions with y (target)
        loss = criterion(outputs, targets)
        
        # C. Backward: AutoDiff calculates the "blame" (gradients)
        # We start by calculating the adjustment of the weights that
        # leads to the largest decrease in Loss.
        loss.backward()
        
        
        # D. Update: Optimizer moves weights down the hill
        # Then we update the weights (jump in the direction of the slope)
        optimizer.step()
        
        # As a rule, we want to save the model. Often we also save
        # a checkpoint during training (that is, the model
        # after a certain number of epochs). If storage is not a factor
        # one should save every epoch.
        
    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")