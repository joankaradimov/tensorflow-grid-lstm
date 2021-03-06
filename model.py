import numpy as np
import tensorflow as tf
from tensorflow.contrib import grid_rnn, rnn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq


class Model(object):
    def __init__(self, args, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell = rnn.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            cell = rnn.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            cell = rnn.BasicLSTMCell(args.rnn_size)
        elif args.model == 'gridlstm':
            cell = grid_rnn.Grid2LSTMCell(args.rnn_size, output_is_tuple = False, use_peepholes = True, forget_bias = 1.0)
        elif args.model == 'gridgru':
            cell = grid_rnn.Grid2GRUCell(args.rnn_size, output_is_tuple = False)
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.cell = rnn.MultiRNNCell([cell] * args.num_layers)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

        vocab_size = 128
        softmax_w = tf.get_variable("softmax_w", [args.rnn_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])

        embedding = tf.get_variable("embedding", [vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = tf.split(inputs, args.seq_length, axis=1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell,
                                                  loop_function=loop if infer else None)
        # output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.ones([args.batch_size * args.seq_length])],
                                                vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, num=200, prime='The '):
        prime = list(map(ord, prime))
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            [state] = sess.run([self.final_state], {
                self.input_data: np.array([[char]]),
                self.initial_state: state,
            })

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return (int(np.searchsorted(t, np.random.rand(1) * s)))

        def random_pick(prob):
            return int(np.random.choice(len(prob), p=prob))

        ret = prime[:]
        sample = ret[-1]
        for n in range(num):
            [probs, state] = sess.run([self.probs, self.final_state], {
                self.input_data: np.array([[sample]]),
                self.initial_state: state,
            })
            # sample = random_pick(probs[0])
            sample = weighted_pick(probs[0])
            ret.append(sample)
        return ''.join(map(chr, ret))
