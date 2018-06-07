import tensorflow as tf

class LSTM(object):
    def __init__(self, inp_size, num_units):
        """
        LSTM implementation
        Args:
            inp_size: input_size
            hidden_size: number of units of hidden variable
        """
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        self.W = tf.get_variable('W', shape=[num_units, 4 * num_units], initializer=xav_init())
        self.U = tf.get_variable('U', shape=[inp_size, 4 * num_units], initializer=xav_init())
        self.b = tf.get_variable('b', shape=[4 * num_units], initializer=tf.constant_initializer(0.))

    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size)

        c, h = tf.scan(self._step,
                elems=[inp, msk],
                initializer=initial_state,
                )
        
        if return_final:
            c, h = c[-1],  h[-1]
        else:
            c = tf.transpose(c, [1, 0, 2])
            h = tf.transpose(h, [1, 0, 2])

        return c, h

    def zero_state(self, batch_size):
        return (tf.zeros([batch_size, self.num_units]), tf.zeros([batch_size, self.num_units]))

    def _step(self, states, inputs):
        prev_c, prev_h = states
        x_t, m_t = inputs
        i, f, o, g = tf.split(tf.matmul(x_t, self.U) + tf.matmul(prev_h, self.W) + self.b,
                num_or_size_splits=4,
                axis=1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c = prev_c*f + g*i
        h = tf.tanh(c) * o
        # slow
        #c = m_t[:, None] * c + (1 - m_t[:, None]) * prev_c
        #h = m_t[:, None] * h + (1 - m_t[:, None]) * prev_h
        c = tf.where(tf.equal(m_t, 1), c, prev_c)
        h = tf.where(tf.equal(m_t, 1), h, prev_h)
        return c, h

if __name__ == '__main__':
    net = LSTM(3,4)
    inp = tf.zeros([10, 11, 3])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, return_final=True)
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.initialize_all_variables())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(np.shape(res[0]))
