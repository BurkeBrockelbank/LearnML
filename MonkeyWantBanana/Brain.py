import torch
import Roomgen
import Grid

class Brain0(torch.nn.Module):
    def __init__(self):
        """ 
        This first iteration of the monkey brain has no memory, and has
        583 inputs (1 for food and 582 for vision). The net is of the form
        200 ReLu, 150 ReLu, 50 ReLu, 20 ReLu, 5 ReLu, Softmax. All symmetry
        considerations are currently ignored.
        """
        super(Brain0, self).__init__()
        self.s1 = 583
        self.s2 = 100
        self.s3 = 100
        self.s4 = 30
        self.s5 = 25
        self.s6 = 5
        self.s7 = 5
        self.s8 = 5
        self.linear1 = torch.nn.Linear(self.s1, self.s2)
        self.linear2 = torch.nn.Linear(self.s2, self.s3)
        self.linear3 = torch.nn.Linear(self.s3, self.s4)
        self.linear4 = torch.nn.Linear(self.s4, self.s5)
        self.linear5 = torch.nn.Linear(self.s5, self.s6)
        self.linear6 = torch.nn.Linear(self.s6, self.s7)
        self.linear7 = torch.nn.Linear(self.s7, self.s8)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        """
        Takes the length 583 input vector and outputs a softmax vector.
        """
        r = torch.nn.ReLU()
        lr = torch.nn.LeakyReLU()
        s = torch.nn.Sigmoid()
        h1 = lr(self.linear1(x))
        h2 = s(self.linear2(h1))
        h3 = s(self.linear3(h2))
        h4 = s(self.linear4(h3))
        h5 = lr(self.linear5(h4))
        h6 = lr(self.linear6(h5))
        h7 = lr(self.linear7(h6))
        y_pred = self.softmax(h7)
        return y_pred

class Brain1(torch.nn.Module):
    """ 
    This second iteration of the monkey brain has no memory, is linear,
    and much simpler. Although it is linear, I have chosen to have two
    hidden layers as it aids in comprehension.    
    """
    def __init__(self):
        """
        Initialization of all the functions.
        """
        super(Brain1, self).__init__()
        self.s1 = 583
        self.s2 = 10
        self.s3 = 5
        self.linear1 = torch.nn.Linear(self.s1, self.s2)
        self.linear2 = torch.nn.Linear(self.s2, self.s3)
        self.softmax = torch.nn.Softmax(-1)
        self.r = torch.nn.ReLU()
        self.lr = torch.nn.LeakyReLU()
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Takes the length 583 input vector and outputs a softmax vector.
        """
        h1 = self.r(self.linear1(x))
        h2 = self.linear2(h1)
        y_pred = self.softmax(h2)
        return y_pred

class Brain2(torch.nn.Module):
    """
    An update on the monkey brain to deal with a new sight field of size 11x11.
    This gives 11*11*5+1 input neurons. We may need slightly more complexity
    because are also planning on implementing danger blocks.
    """
    def __init__(self):
        """
        Initialize sizes, get functions from functionals, establish layers of
        NN.
        """
        # Initializing Neural Net object
        super(Brain2, self).__init__()
        # Initialize sizes
        self.s1 = len(Grid.SIGHT)*len(Grid.SIGHT[0])*len(Roomgen.BLOCKTYPES)+1
        self.s2 = 10
        self.s3 = 5
        # Run functionals
        self.softmax = torch.nn.Softmax(-1)
        self.r = torch.nn.ReLU()
        self.lr = torch.nn.LeakyReLU()
        self.s = torch.nn.Sigmoid()
        # Define units
        self.linear1 = torch.nn.Linear(self.s1, self.s2)
        self.linear2 = torch.nn.Linear(self.s2, self.s3)

    def forward(self, x):
        """
        A neural net with two hidden layrs of size 10 and 5 respectively.
        """
        h1 = self.s(self.linear1(x))
        h2 = self.s(self.linear2(h1))
        y_pred = self.softmax(h2)
        return y_pred

class Brain3(torch.nn.Module):
    """
    Update Brain2 to include some convolutions on the wall and danger chnnels
    to sense features like impasses and passages.
    """
    pass