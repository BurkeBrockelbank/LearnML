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

PROGRESSION TO A FULL CNN
I added a convolutional layer to the progressional brain. The net architecture is
(4x11x11) & 1
(4x11x11) & 1 Conv2d ReLU
485 Flatten
9 Relu
8 Relu
5

It behaves quite well after training on AIDATA\AIData0.txt for 30 epochs in 3 batches each with randomized order between each epoch. This is in contrast to more rudimentary nets. Clearly the convolutional layer makes the network more capable of learning the AI algorithm.

Built gif of performance in ./img/OneConvLayer showing this. The qualities are clearly wrong, but at least the net is making the right decisions. After reinforcement elarning the qualities improve drastically. I will add in more layers.

Added yet another fully-connected layer
(4x11x11) & 1
(4x11x11) Conv2d ReLU & 1
484 Flatten & 1
25 ReLU & 1
26
9 Relu
8 Relu
5

This seems to be important. The supervised training is going a lot better with this layer
added in! The loss is looking to stagnate around 0.45 whereas without this layer it was getting stuck around 0.7.

The result of supervised training actually wasn't too good because the AI often gets stuck, the monkey often just got stuck itself.

This bad result carried over to the curated training. I will try again with fewer epochs.

Did 10 epochs and the loss stopped at 0.67 like the previous successful test. It still didn't work too well. The performance is much better. Adding in the rest of the convolutional layers.

(4x11x11) & 1
(4x11x11) Conv2d ReLU & 1
(6x9x9) Conv2d ReLU & 1
(4x7x7) Conv2d ReLU & 1
(1x5x5) Conv2d ReLU & 1
25 Flatten & 1
26 Concat
9 Relu
8 Relu
5

Supervised training started with higher than expected loss (2.95). Each epoch is now taking less time, but the training rate has decreased and stagnated at a loss of 1.2. Monkey behavior seems fairly random. The monkey is not making intelligent choices. This is not a surprise because the amount of training data was not increased. We will add in another two data files.

Using AIDATA\AIData0.txt, AIDATA\AIData1.txt, AIDATA\AIData2.txt for training. Supervised training seemed totally ineffective. The monkey just sits still. Not staying still anymore. I am going to increase the exploration rate to a constant 0.8 and go for another round of curated reinforcement learning.

The net seems to be insensitive to training.