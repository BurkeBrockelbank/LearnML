Hello,

  Welcome to the very first portion of my Monkey Simulator. Although this project will eventually be expanded to include interactions with monkeys and simulate tribal behavior and cooperation, the first job is to get monkeys to survive in a test tube environment. What I mean is, the monkey needs to learn that it needs food to survive.

  This was attempted with supervised learning. I played the game for aproximately 300 turns and trained the monkey based on this. It ended up not working that well because the monkey would get stuck in infinite loops, especially if it ended up in odd areas that it was never trained for. It became clear that the monkey needed some amount of memory in order to function. I first attempted to implement this by having some of the outputs of the neural net be fed back into the inputs, but I ran into programming issues. There is no mathematical reason this can't work. This kind of memory is the most attractive because it allows for an abstract form of memory with planning.

  I switched the monkey over to a reinforcement learning algorithm (DQN), but have run into the issue that the monkey is too stupid to learn. I am planning on implementing transfer learning by doing supervised learning on some handmade training data before refining the brain with reinforcement learning.

Burke Brockelbank
bbrockel@ucalgary.ca
burkelibrockelbank.herokuapp.com



DEVNOTES
Indexing is to be done in matrix indexing format ((row, column) with (0,0) at the top left) unless otherwise stated.

TESTING LINEAR MODEL AGAINST AI
An AI was developed to navigate the monkey game. This AI was then used to generate over 1.3 million turns of gameplay. A linear model was trained with this data. Reinforcement learning was attempted after this, but no increase in the accuracy of the quality function was found so this training was reverted.

The models were scored against eachother based on the amount of food they could collect in 30 turns averaged over 1000 random placements. This was repeated five times each.

 Model | Score
-------+-------
Linear |  6.45, 6.923, 6.739, 7.158, 6.906
    AI | 39.341, 38.701, 39.334, 40.7, 39.474

It is clear that the linear model for the monkey brain is insufficiently complex for modelling the artificial intelligence.
