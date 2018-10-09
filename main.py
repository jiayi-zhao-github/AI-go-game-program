#!/usr/bin/env python3
import argparse
import argh
from contextlib import contextmanager
import os
import random
import re
import sys
import time
import time
import save_train as save_t
import gtp as gtp_lib
import  change as ch

from policy import PolicyNetwork
from MCTS import RandomPlayer, PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS
from data import DataSet, parse_data_sets

TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))


def gtp(strategy, read_file=None):
    n = PolicyNetwork(use_cpu=True)
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
    else:
        sys.stderr.write("错误")
        sys.exit()

    gtp_engine = gtp_lib.Engine(instance)
    sys.stderr.write("GTP\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = input()

        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)

            sys.stdout.write(engine_reply)
            sys.stdout.flush()



def self_play(strategy, read_file=None):
    n = PolicyNetwork(use_cpu=True)
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
    else:
        sys.stderr.write("Unknown strategy")
        sys.exit()
        #instance神经网络
    gtp_engine = gtp_lib.Engine(instance)
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()

    p1 = -1
    save = ''
    inpt = 'genmove b'
    n = 500
    while n>0 :
        inpt = 'genmove b'
        if n%2 == 1:
            inpt = 'genmove b'
        else:
            inpt = 'genmove w'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            if engine_reply == '= pass\n\n':
                #engine_reply == '= pass\n\n'
                n = 0
            else:
                o1 = ''
                if len(engine_reply) == 7:
                    o1= engine_reply[3]+engine_reply[4]
                else:
                    o1 =engine_reply[3]

                if n%2 == 1:
                    o2=ch.change(engine_reply[2])+ch.change(o1)
                    save = save+';B['+ch.change(engine_reply[2])+ch.change(o1)+']'
                else:
                    o2=ch.change(engine_reply[2])+ch.change(o1)
                    save = save+';W['+ch.change(engine_reply[2])+ch.change(o1)+']'

            sys.stdout.flush()

        n= n-1
    p7 = instance.position.result()
    save2 = '(;GM[1]\n SZ[19]\nPB[go1]\nPW[go2]\nKM[6.50]\nRE['+p7[0]+']\n'


    save2 = save2+save+')'

    wenjian = ''

    wenjian =str(time.time())
    p3 = '4'
    save_t.make_folder(wenjian+'_selfplay')
    save_t.save_txt(wenjian+'_selfplay',p3,save2)

#self_play('mcts','tmp/savemodel')
#self_play('policy','tmp2')





def preprocess(*data_sets, processed_dir="processed_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    test_chunk, training_chunks = parse_data_sets(*data_sets)


    print("写 chunk")
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)

    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("写chunk %s" % i)
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    print("%s chunks " % (i+1))


def train(processed_dir="processed_data"):
    checkpoint_freq=10000
    read_file=None
    save_file='tmp2'
    epochs=10
    logdir='logs2'

    #
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(processed_dir, fname) 
        for fname in os.listdir(processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    if read_file is not None:
        read_file = os.path.join(os.getcwd(), save_file)
    n = PolicyNetwork()
    n.initialize_variables(read_file)
    if logdir is not None:
        n.initialize_logging(logdir)
    last_save_checkpoint = 0
    for i in range(epochs):
        random.shuffle(train_chunk_files)
        for file in train_chunk_files:
            print("提取 %s" % file)
            with timer("load dataset"):
                train_dataset = DataSet.read(file)
            with timer("training"):
                n.train(train_dataset)
            with timer("save model"):
                n.save_variables(save_file)
            if n.get_global_step() > last_save_checkpoint + checkpoint_freq:
                with timer("test set evaluation"):
                    n.check_accuracy(test_dataset)
                last_save_checkpoint = n.get_global_step()

#gtp('mcts','tmp/savemodel')

#train()
self_play('mcts','tmp2')
#preprocess('preprocess2','data/kgs-test/')


