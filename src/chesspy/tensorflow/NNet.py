import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from NeuralNet import NeuralNet
from utils import *
from .ChessNNet import ChessNNet as onnet

np.random.seed(11)
sys.path.append('../../')

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 256,  # 512 need 5.88 GB RAM free in GPU
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        tf.compat.v1.disable_eager_execution()
        super().__init__(game)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.nnet = onnet(game, args)
        self.action_size = game.getActionSize()

        self.sess = tf.compat.v1.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.compat.v1.Session() as temp_sess:
            temp_sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        t_bar = tqdm(range(args.epochs), position=0, leave=True, desc="EPOCH :::")
        for epoch in t_bar:
            # print('EPOCH ::: ' + str(epoch + 1))

            for batch_idx in tqdm(range(int(len(examples) / args.batch_size)), position=0, leave=True):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict
                input_dict = {
                    self.nnet.input_boards: boards,
                    self.nnet.target_pis: pis,
                    self.nnet.target_vs: vs,
                    self.nnet.dropout: args.dropout,
                    self.nnet.isTraining: True
                }

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict)

                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))
                t_bar.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

    def predict(self, board):
        """
        board: np array with board (18,8,8)
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        prob, v = self.sess.run([self.nnet.prob, self.nnet.v],
                                feed_dict={self.nnet.input_boards: board, self.nnet.dropout: 0,
                                           self.nnet.isTraining: False})

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return prob[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver is None:
            self.saver = tf.compat.v1.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + '.meta'):
            raise ("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.compat.v1.train.Saver()
            self.saver.restore(self.sess, filepath)
