from collections import deque
from Arena import Arena
import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import chess

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, agent, nnet, game, args):
        self.agent = agent
        self.agent_class = agent.__class__
        self.nnet = nnet
        self.pnet = self.nnet.__class__(game, args)  # the competitor network
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.args = args
        self.agent_name = self.agent.name
        self.agent_is_white = self.agent.is_white
        self.game = game
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = chess.Board()
        self.curPlayer = 1
        episodeStep = 0
        num_iterations = self.args.numIterations
        while True:
            episodeStep += 1
            canonicalBoard = board if self.curPlayer == 1 else board.mirror()
            temp = int(episodeStep < self.args.tempThreshold)

            action = self.agent._AZ_search(canonicalBoard.fen(), num_iterations),

            trainExamples.append([self.game.toArray(canonicalBoard), self.curPlayer, action, None])

            board.push(chess.Move.from_uci(action[0]))
            self.curPlayer *= -1
            
            res = self.game.getGameEnded(board, self.curPlayer)
            if res == "1-0":
                r = self.curPlayer
            elif res == "0-1":
                r = -self.curPlayer
            elif res == "1/2-1/2":
                r = 0.5 
            else:
                r = res
            if res !=0:
                print(board)
                print(board.result()," ",board.fen())
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            for eps in range(self.args.numEps):
                self.agent = self.agent.__class__(self.agent_name, self.nnet, self.game) #reset search tree
                iterationTrainExamples += self.executeEpisode()

            # save the iteration examples to the history 
            self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            pmcts = self.agent.__class__(name='opponent', nnet=self.pnet,is_white=not self.agent_is_white)
            
            self.nnet.train(trainExamples)
            nmcts = self.agent.__class__(name=self.agent_name,nnet=self.nnet,is_white=self.agent_is_white)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts._AZ_Search(x, num_iterations, return_probs=True)),
                          lambda x: np.argmax(nmcts._AZ_Search(x, num_iterations, return_probs=True)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # TODO try this -> if (not (pwins+nwins == 0)) and (float(nwins)/(pwins+nwins) < self.args.updateThreshold):
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
