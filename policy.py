
import math
import os
import sys
import tensorflow as tf

import features
import go
import p1

EPSILON = 1e-35

class PolicyNetwork(object):
    def __init__(self, features=features.DEFAULT_FEATURES, k=32, num_int_conv_layers=3, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features)
        self.features = features
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.test_stats = StatisticsCollector()
        self.training_stats = StatisticsCollector()
        self.session = tf.Session()
        if use_cpu:#内存
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            self.set_up_network()
        tmp = 9
    def set_up_network(self):

        global_step = tf.Variable(0, name="global_step", trainable=False)
        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.placeholder(tf.float32, shape=[None, go.N ** 2])


        def _weight_variable(shape, name):

            number_inputs_added = p1.product(shape[:-1])
            stddev = 1 / math.sqrt(number_inputs_added)

            return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

        def _conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

        # 5*5 卷积28-》32卷积核
        W_conv_init = _weight_variable([5, 5, self.num_input_planes, self.k], name="W_conv_init")
        h_conv_init = tf.nn.relu(_conv2d(x, W_conv_init), name="h_conv_init")

        #多个 3x3 卷积 32 -32
        W_conv_intermediate = []
        h_conv_intermediate = []
        _current_h_conv = h_conv_init
        for i in range(self.num_int_conv_layers):
            with tf.name_scope("layer"+str(i)):
                W_conv_intermediate.append(_weight_variable([3, 3, self.k, self.k], name="W_conv"))
                h_conv_intermediate.append(tf.nn.relu(_conv2d(_current_h_conv, W_conv_intermediate[-1]), name="h_conv"))
                _current_h_conv = h_conv_intermediate[-1]
        
        W_conv_final = _weight_variable([1, 1, self.k, 1], name="W_conv_final")
        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32), name="b_conv_final")
        h_conv_final = _conv2d(h_conv_intermediate[-1], W_conv_final)

        logits = tf.reshape(h_conv_final, [-1, go.N ** 2]) + b_conv_final


       #gdadsad
        self.output = tf.nn.softmax(tf.reshape(h_conv_final, [-1, go.N ** 2]) +b_conv_final)
        #loss
        log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
        #训练
        train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood_cost, global_step=global_step)
        #最大概率
        was_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        #准确性
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        weight_summaries = tf.summary.merge([
            tf.summary.histogram(weight_var.name, weight_var)
            for weight_var in [W_conv_init] +  W_conv_intermediate + [W_conv_final, b_conv_final]],
            name="weight_summaries"
        )
        activation_summaries = tf.summary.merge([
            tf.summary.histogram(act_var.name, act_var)
            for act_var in [h_conv_init] + h_conv_intermediate + [h_conv_final]],
            name="activation_summaries"
        )
        #保存
        saver = tf.train.Saver()

        #转
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "test"), self.session.graph)
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)

    def initialize_variables(self, save_file=None):
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            #读取保存
            self.saver.restore(self.session, save_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file):
        if save_file is not None:
            print("Saving checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file)

    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            _, accuracy, cost = self.session.run(
                [self.train_step, self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
        if self.training_summary_writer is not None:
            activation_summaries = self.session.run(
                self.activation_summaries,
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_summary_writer.add_summary(activation_summaries, global_step)
            self.training_summary_writer.add_summary(accuracy_summaries, global_step)
#dd1
    def connect(self,tmp1,tmp2):


        h_fc1 = tf.add(tmp1, tmp2)
        return h_fc1
    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]
        return probabilities.reshape([go.N, go.N])

    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        weight_summaries = self.session.run(self.weight_summaries)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("Step %s test data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)

class StatisticsCollector(object):

    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary
