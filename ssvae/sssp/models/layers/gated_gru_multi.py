import tensorflow as tf

class MultiGatedGRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_size, num_units, num_tasks, key_size, keys, scope_name='MultiGatedGRUCell'):
        self.inp_size = input_size
        self.num_units = num_units
        self.num_tasks = num_tasks
        self.key_size = key_size
        self.keys = keys
        xav_init = tf.contrib.layers.xavier_initializer

        with tf.variable_scope(scope_name):
            self.W_0 = tf.get_variable('W0', shape=[num_units, 2 * num_units], )#initializer=xav_init())
            self.U_0 = tf.get_variable('U0', shape=[input_size, 2 * num_units], )#initializer=xav_init())
            self.V_0 = tf.get_variable('V0', shape=[key_size, 2 * num_units], )#initializer=xav_init())
            self.b_0 = tf.get_variable('b0', shape=[2 * num_units], initializer=tf.constant_initializer(0.))
    
            self.W_1 = tf.get_variable('W1', shape=[num_units, num_units], )#initializer=xav_init())
            self.U_1 = tf.get_variable('U1', shape=[input_size, num_units], )#initializer=xav_init())
            self.V_1 = tf.get_variable('V1', shape=[key_size, num_units], )#initializer=xav_init())
            self.b_1 = tf.get_variable('b1', shape=[num_units], initializer=tf.constant_initializer(0.))
    
            self.W_g = tf.get_variable('Wg', shape=[num_units, num_units], )#initializer=xav_init())
            self.U_g = tf.get_variable('Ug', shape=[input_size, num_units], )#initializer=xav_init())
            self.V_g = tf.get_variable('Vg', shape=[key_size, num_units], )#initializer=xav_init())
            self.b_g = tf.get_variable('bg', shape=[num_units], initializer=tf.constant_initializer(0.))
            self.u_g = tf.get_variable('ug', shape=[num_units], )#initializer=xav_init())
    
    def forward(self, inp, msk, initial_state=None, time_major=False, return_final=False, scope='GatedLSTM'):
        """ to build the graph, run forward """
        if not time_major:
            inp = tf.transpose(inp, [1, 0, 2])
            msk = tf.transpose(msk, [1, 0])
        
        # after transposition, the shape is [seqlen, batch_size, inp_size]
        batch_size = tf.shape(inp)[1]

        if initial_state is None:
            initial_state = self.zero_state(batch_size, tf.float32)

        res = tf.scan(self._mask_step,
                elems=[inp, msk],
                initializer=initial_state,
                )

        if return_final:
            res = [(r[0][-1], r[1][-1]) for r in res]
        else:
            if not time_major:
                res = [(tf.transpose(r[0], [1,0,2]), tf.transpose(r[1], [1,0,2])) for r in res]
        
        return res

    def zero_state(self, batch_size, dtype):
        init_states = [(tf.zeros([batch_size, self.num_units], dtype=dtype),
                tf.zeros([batch_size, 1], dtype=dtype),) for _ in range(self.num_tasks)]
        return init_states

    def _mask_step(self, states_tm1, inputs):
        x_t, m_t = inputs
        states = self._step(states_tm1, x_t)
        new_states = []
        for i in range(len(states)):
            new_states_i = (tf.where(tf.equal(m_t, 1), states[i][0], states_tm1[i][0]),
                    tf.where(tf.equal(m_t, 1), states[i][1], states_tm1[i][1]))
            new_states.append(new_states_i)
        return new_states

    def _gate_step(self, states, inputs):
        h_tm1, g_tm1 = states
        x_t, k_t = inputs
        
        g = tf.matmul(h_tm1, self.W_g)
        g += tf.matmul(x_t, self.U_g) + tf.matmul(k_t[None,:], self.V_g) + self.b_g
        g = tf.reduce_sum(g * self.u_g, axis=1, keep_dims=True)
        g = tf.sigmoid(g)

        h = self._gru_step(h_tm1, [x_t, k_t])
        h = g * h + (1 - g) * h_tm1
        
        return h, g

    def _gru_step(self, h_tm1, inputs):
        x_t, k_t = inputs
        z, r = tf.split(tf.matmul(x_t, self.U_0) + tf.matmul(h_tm1, self.W_0) + tf.matmul(k_t[None,:], self.V_0) + self.b_0,
                num_or_size_splits=2,
                axis=1)
        z, r = tf.sigmoid(z), tf.sigmoid(r)
        h = tf.tanh(tf.matmul(x_t, self.U_1) + tf.matmul(r * h_tm1, self.W_1) + tf.matmul(k_t[None,:], self.V_1) + self.b_1)
        h = (1 - z) * h + z * h_tm1
        return h

    def _step(self, states_tm1, inputs):
        new_states = []
        for i in range(self.num_tasks):
            states_i = self._gate_step(states_tm1[i], [inputs, self.keys[i]])
            new_states.append(states_i)
        return new_states

    def __call__(self, inputs, states_tm1):
        states = self._step(states_tm1, inputs)
        return states, states

    @property
    def state_size(self):
        return [(self.num_units, 1) for _ in range(self.num_tasks)]

    @property
    def output_size(self):
        return [(self.num_units, 1) for _ in range(self.num_tasks)]

if __name__ == '__main__':
    #keys = tf.ones([7,8])

    keys = [tf.get_variable("key_%d" % i, [8]) 
                for i in range(7)]

    net = MultiGatedGRUCell(3, 5, 7, 8, keys)
    inp = tf.ones([1, 2, 3])
    msk = tf.ones([1, 2])
    y = net.forward(inp, msk, return_final=True)
    print(type(y[0]))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run(y)
    import numpy as np
    print(res)
    '''
    for i in res:
        print()
        for j in i:
            print(j.shape)
    '''

    a, b = tf.nn.dynamic_rnn(cell=net,
            inputs=inp,
            sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
            dtype=tf.float32,
            )

    #sess.run(tf.global_variables_initializer())
    res, final = sess.run([a, b])
    
    print(res)
    for i in res:
        print()
        for j in i:
            print(j.shape)

    print(final)
