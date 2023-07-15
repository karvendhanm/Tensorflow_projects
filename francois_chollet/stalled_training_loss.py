"""
In a ML/DL problem there are 3 common types of problems while training the model
    1) training loss doesn't go down, it stalls
    2) training loss goes down, but the model doesn't generalize. Unable to beat a trivial baseline.
    3) training and validation loss both goes down, but the model isn't overfitting.

    things to look out for when confronted with problem number 1:
    it is usually the case of 1) choice of optimizer
                              2) the distribution of initial values in the weights of the model.
                              3) learning rate
                              4) batch size

        Even as the 4 aforementioned parameters are interdependent, usually we can tune learning rate and batch size
        and get the training loss to go down.
"""