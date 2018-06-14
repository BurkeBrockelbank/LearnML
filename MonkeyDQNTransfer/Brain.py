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

class BrainDQN(torch.nn.Module):
    """
    Try training the monkey with reinforcement learning.
    This class approximates quality the function Q.
    """
    def __init__(self, memoryLength):
        """
        Initialize the structure of the neural net. We have the standard
        11*11*6+1=727 size for the state vector of a single turn. The input S
        will be multiple times this length to account for memory. The action
        vector is size 5. The output is of course size 1.

        To begin with, we will have three hidden layers of length 50, 30,
        and 10.
        """
        # Initialize for parent class
        super(BrainDQN, self).__init__()
        # Count the number of blocks in vision
        vision = [x.count(1) for x in Grid.SIGHT]
        vision = sum(vision)
        # Calculate input size
        inputSize = (vision*len(Roomgen.BLOCKTYPES) + 1)*memoryLength + len(Roomgen.WASD)
        self.l1 = torch.nn.Linear(inputSize, 300)
        self.l2 = torch.nn.Linear(300, 121)
        self.l3 = torch.nn.Linear(121, 5)
        self.l4 = torch.nn.Linear(5, 1)
        # Define the ReLU function
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.l1.weight.data)
        torch.nn.init.xavier_uniform_(self.l2.weight.data)
        torch.nn.init.xavier_uniform_(self.l3.weight.data)
        torch.nn.init.xavier_uniform_(self.l4.weight.data)

    def forward(self, s, a):
        """
        This calculates the reward for a given state and action.
        State is taken to be shape (1,N) and a is shape (1,N) as well
        for some N

        Args:
            s: The state vector.
            a: The action.
        """
        h = self.sigmoid(self.l1(torch.cat((s,a), -1)))
        h = self.sigmoid(self.l2(h))
        h = self.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

    def maxa(self, s):
        """
        This returns the action that will maximize the quality from a,
        given state. Does not factor into gradient calculations.

        Args:
            s: The state of the system.
        Returns:
            0: The action to do.
        """
        # Turn off autograd
        with torch.no_grad():
            # Check all the possibilities for movement
            a = torch.eye(len(Roomgen.WASD))
            # Make copies of the state for each test
            sCopies = torch.empty(len(Roomgen.WASD), len(s))
            for i in range(len(sCopies)):
                sCopies[i] = s*1
            Q = self.forward(sCopies, a)
            # Maximize Q with respect to a
            maxIndex = int(Q.max(0)[1])
            # Return the action corresponding to this
            return a[maxIndex]

    def pi(self, s, epsilon, loud=False):
        """
        This is the policy for the brain that returns the action that
        maximizes the reward.
        This does not factor into the gradient calculations.

        Args:
            s: The state of the system.
            epsilon: The threshold to do a random action. Random actions are
                done a proportion of epsilon of the time.
            loud: Default False. If True, reports when the movement is random.
        returns:
            0: The action to do.
        """
        # Turn off autograd
        with torch.no_grad():
            # Check if we will be doing a random movement
            if (torch.rand(1) <= epsilon) == 1:
                # We rolled a random number less than epsilon, so we should
                # take a random move
                a = torch.eye(len(Roomgen.WASD))
                randomIndex = int(torch.randint(len(a),(1,)))
                if loud:
                    print('Random movement (',round(epsilon*100),'%)', sep='')
                return a[randomIndex]
            else:
                a = self.maxa(s)
                if loud:
                    print('Delibe movement (',round((1-epsilon)*100),\
                        '%) Q = ', self.forward(s,a).item(),sep='')
                return self.maxa(s)


