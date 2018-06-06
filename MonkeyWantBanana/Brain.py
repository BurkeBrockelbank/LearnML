import torch

class Brain0(torch.nn.Module):
    def __init__(self):
        """ 
        This first iteration of the monkey brain has no memory, and has
        583 inputs (1 for food and 582 for vision). The net is of the form
        200 ReLu, 150 ReLu, 50 ReLu, 20 ReLu, 5 ReLu, Softmax. All symmetry
        considerations are currently ignored.
        """
        super(Brain0, self).__init__()
        self.linear1 = torch.nn.Linear(583, 200)
        self.linear2 = torch.nn.Linear(200, 150)
        self.linear3 = torch.nn.Linear(150, 50)
        self.linear4 = torch.nn.Linear(50, 20)
        self.linear5 = torch.nn.Linear(20, 5)
        self.softmax6 = torch.nn.Softmax(0)

    def forward(self, x):
        """
        Takes the length 583 input vector and outputs a softmax vector.
        """
        r = torch.nn.ReLU()
        h1 = r(self.linear1(x))
        h2 = r(self.linear2(h1))
        h3 = r(self.linear3(h2))
        h4 = r(self.linear4(h3))
        h5 = r(self.linear5(h4))
        y_pred = self.softmax6(h5)
        return y_pred