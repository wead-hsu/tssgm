import tensorflow as tf

class FcLSTM(object):
    def __init__(self, inp_size, num_units, num_classes, cell_clip=None, use_peepholes=True):
        """
        Gated LSTM is LSTM equipped with gate mechanism to ignore the
        irrelevant part of input.
        Args:
            inp_size: input_size
            hidden_size: number of units of hidden variable
        """
        # intializer
        self.inp_size = inp_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.cell_clip = cell_clip
        self.use_peepholes = use_peepholes
        xav_init = tf.contrib.layers.xavier_initializer

        # parameters
        self.W = tf.get_variable('W', shape=[num_units, 3 * (num_units - num_classes)], initializer=xav_init())
        self.U = tf.get_variable('U', shape=[inp_size, 3 * (num_units - num_classes)], initializer=xav_init())
        self.b = tf.get_variable('b', shape=[3 * (num_units - num_classes)], initializer=tf.constant_initializer(0.))

        self.W_o = tf.get_variable('W_o', shape=[num_units, num_units], initializer=xav_init())
        self.U_o = tf.get_variable('U_o', shape=[inp_size,  num_units], initializer=xav_init())
        self.b_o = tf.get_variable('b_o', shape=[num_units], initializer=tf.constant_initializer(0.))

        #self.W_yc = tf.get_variable('W_yc', shape=[num_classes, num_units], initializer=xav_init())

        if self.use_peepholes:
            self._w_f_diag = tf.get_variable("w_f_diag", shape=[num_units - num_classes], initializer=xav_init())
            self._w_i_diag = tf.get_variable("w_i_diag", shape=[num_units - num_classes], initializer=xav_init())
            self._w_o_diag = tf.get_variable("w_o_diag", shape=[num_units], initializer=xav_init())

    def forward(self, inp, msk, label, initial_state=None, time_major=False, return_final=False, scope='ScLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            label = tf.transpose(label, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size)
        else:
            initial_state = (initial_state[0][:, :self.num_units - self.num_classes], initial_state[1])

        c, h = tf.scan(self._lstm_step,
                elems=[inp, label, msk],
                initializer=initial_state,
                )

        if return_final:
            c = c[-1]
            h = c[-1]
        else:
            if not time_major:
                c = tf.transpose(c, [1, 0, 2])
                h = tf.transpose(h, [1, 0, 2])

        return c, h

    def zero_state(self, batch_size):
        # (c, h)
        return (tf.zeros([batch_size, self.num_units - self.num_classes]),
                tf.zeros([batch_size, self.num_units]),
                )

    def _lstm_step(self, states, inputs):
        prev_c, prev_h = states
        x_t, y_t, m_t = inputs
        i, f, _c = tf.split(tf.matmul(x_t, self.U) + tf.matmul(prev_h, self.W) + self.b,
                num_or_size_splits=3,
                axis=1)
        o = tf.matmul(x_t, self.U_o) + tf.matmul(prev_h, self.W_o) + self.b_o

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        
        if self.use_peepholes:
            print(prev_c.shape, self._w_i_diag.shape)
            i += prev_c * self._w_i_diag
            f += prev_c * self._w_f_diag

        o = tf.sigmoid(o)
        _c = tf.tanh(_c)
        c = prev_c*f + _c*i #+ tf.tanh(tf.matmul(y_t, self.W_yc))
 
        if self.cell_clip is not None and self.cell_clip > 0:
            # pylint: disable=invalid-unary-operand-type
            c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)
            # pylint: enable=invalid-unary-operand-type

        if self.use_peepholes:
            o += tf.concat([c, y_t], axis=1) * self._w_o_diag

        h = tf.tanh(tf.concat([c, y_t], axis=1)) * o
       
        # slow
        #c = m_t[:, None] * c + (1 - m_t[:, None]) * prev_c
        #h = m_t[:, None] * h + (1 - m_t[:, None]) * prev_h
        #c = tf.where(tf.equal(m_t, 1), c, prev_c)
        #h = tf.where(tf.equal(m_t, 1), h, prev_h)
        return c, h

if __name__ == '__main__':
    net = FcLSTM(3, 4, 2, cell_clip=10)
    inp = tf.zeros([10, 11, 3])
    label = tf.zeros([10, 11, 2])
    msk = tf.ones([10, 11])
    y = net.forward(inp, msk, label, return_final=True, initial_state=(tf.zeros([10, 2]),tf.zeros([10,4])))
    print(type(y[0]))

    sess = tf.Session()
    #sess.run(tf.global_variable_initializer())
    sess.run(tf.global_variables_initializer())
    #print(sess.run(y))
    res = sess.run(y)
    import numpy as np
    print(np.shape(res[0]))
