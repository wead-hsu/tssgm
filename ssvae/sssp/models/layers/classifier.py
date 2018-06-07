import tensorflow as tf

def create_encoder(self, inp, msk, keep_rate, scope_name, args):
    with tf.variable_scope(scope_name):
        emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
        
        if args.classifier_type == 'LSTM':
            with tf.variable_scope('init', initializer=tf.random_normal_initializer(0, 0.1)):
                cell = tf.contrib.rnn.LSTMCell(args.num_units, state_is_tuple=True, use_peepholes=True, cell_clip=10)
                if args.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
                _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                        dtype=tf.float32,
                        sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
                if args.num_layers == 1:
                    enc_state = enc_state[-1]
                else:
                    enc_state = enc_state[-1][-1]
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
        elif args.classifier_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(args.num_units)
            if args.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
            _, enc_state = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'BiGRU':
            cell = tf.contrib.rnn.GRUCell(args.num_units/ 2)
            _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, 
                    cell_bw=cell, 
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                    dtype=tf.float32)
            enc_state = tf.concat(enc_state, axis=1)
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'BiGRU+maxpooling':
            cell = tf.contrib.rnn.GRUCell(args.num_units/ 2)
            enc_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, 
                    cell_bw=cell, 
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                    dtype=tf.float32)
            enc_state = tf.concat(enc_states, axis=2)
            enc_state = tf.reduce_max(enc_state, axis=1)
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'GatedGRU':
            from sssp.models.layers.gated_gru import GatedGRU
            """
            enc_layer = GatedGRU(emb_inp.shape[2], args.num_units)
            enc_state, weights = enc_layer.forward(emb_inp, msk, return_final=True)
            """
            cell = GatedGRU(emb_inp.shape[2], args.num_units)
            enc_states, enc_state = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    initial_state=cell.zero_state(tf.shape(emb_inp)[0], dtype=tf.float32),
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            enc_state = enc_state[0]
            weights = enc_states[1]
            self.gate_weights = weights
            self._logger.debug(enc_state.shape)
            self._logger.debug(weights.shape)
        elif args.classifier_type == 'GatedCtxGRU':
            from sssp.models.layers.gated_gru_with_context import GatedGRU

            def _reverse(input_, seq_lengths, seq_dim, batch_dim):
                if seq_lengths is not None:
                    return array_ops.reverse_sequence(
                        input=input_, seq_lengths=seq_lengths,
                        seq_dim=seq_dim, batch_dim=batch_dim)
                else:
                    return array_ops.reverse(input_, axis=[seq_dim])
            
            sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
            time_dim = 1
            batch_dim = 0
            cell_bw = tf.contrib.rnn.GRUCell(args.num_units)
            with tf.variable_scope("bw") as bw_scope:
              inputs_reverse = _reverse(
                      tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                      seq_lengths=sequence_length,
                      seq_dim=time_dim, batch_dim=batch_dim)
              tmp, output_state_bw = tf.nn.dynamic_rnn(
                      cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
                      dtype=tf.float32,
                      scope=bw_scope)
        
            output_bw = _reverse(
                    tmp, seq_lengths=sequence_length,
                    seq_dim=time_dim, batch_dim=batch_dim)
        
            enc_layer = GatedGRU(emb_inp.shape[2], args.num_units, args.num_units)
            enc_state, weights = enc_layer.forward(
                      tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)), output_bw, msk, return_final=True)
            self.gate_weights = weights
            weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
            #weights = msk / tf.reduce_sum(msk, axis=1, keep_dims=True)
        elif args.classifier_type == 'tagGatedGRU':
            from sssp.models.layers.gated_gru_with_context import GatedGRU
            sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
            cell_tag = tf.contrib.rnn.GRUCell(args.num_units)
            with tf.variable_scope("tagrnn") as scope:
                outputs, final_state = tf.nn.dynamic_rnn(
                        cell=cell_tag, 
                        inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                        sequence_length=sequence_length,
                        dtype=tf.float32,
                        scope=scope)
            
            #inp = tf.concat([emb_inp, outputs], axis=2)
            gatedctxgru_layer = GatedGRU(emb_inp.shape[2], args.num_units, args.num_units)
            enc_state, weights = gatedctxgru_layer.forward(emb_inp, outputs, msk, return_final=True)
            self.gate_weights = weights
            weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
        elif args.classifier_type == 'Tag=bigru;Gatedgru=bwctx+tag':
            from sssp.models.layers.gated_gru_with_context import GatedGRU
            sequence_length = tf.to_int64(tf.reduce_sum(msk, axis=1))
            
            tag_states, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.contrib.rnn.GRUCell(100),
                    cell_bw=tf.contrib.rnn.GRUCell(100),
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                    dtype=tf.float32)
            tag_states = tf.concat(tag_states, axis=2)
    
            def _reverse(input_, seq_lengths, seq_dim, batch_dim):
                if seq_lengths is not None:
                    return array_ops.reverse_sequence(
                        input=input_, seq_lengths=seq_lengths,
                        seq_dim=seq_dim, batch_dim=batch_dim)
                else:
                    return array_ops.reverse(input_, axis=[seq_dim])
            
            time_dim = 1
            batch_dim = 0
            cell_bw = tf.contrib.rnn.GRUCell(args.num_units)
            with tf.variable_scope("bw") as bw_scope:
                inputs_reverse = _reverse(
                        tf.concat([emb_inp, tag_states], axis=2), seq_lengths=sequence_length,
                        seq_dim=time_dim, batch_dim=batch_dim)
                tmp, output_state_bw = tf.nn.dynamic_rnn(
                        cell=tf.contrib.rnn.GRUCell(100), 
                        inputs=tf.nn.dropout(inputs_reverse, tf.where(self.is_training_plh, keep_rate, 1.0)),
                        sequence_length=sequence_length,
                        dtype=tf.float32,
                        scope=bw_scope)
        
            output_bw = _reverse(
                    tmp, seq_lengths=sequence_length,
                    seq_dim=time_dim, batch_dim=batch_dim)
        
            enc_layer = GatedGRU(
                    inp_size=emb_inp.shape[2], 
                    context_size=100+args.embd_dim, 
                    num_units=args.num_units)
            enc_state, weights = enc_layer.forward(
                    inp=emb_inp, 
                    ctx=tf.nn.dropout(tf.concat([output_bw, emb_inp], axis=2), tf.where(self.is_training_plh, keep_rate, 1.0)),
                    msk=msk, 
                    return_final=True)
            self.gate_weights = weights
            #weights = weights / tf.reduce_sum(weights, axis=1, keep_dims=True)
        elif args.classifier_type == 'EntityNetwork':
            from sssp.models.layers.entity_network import EntityNetwork
            cell = tf.contrib.rnn.GRUCell(args.num_units)
            if args.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
            enc_states, _ = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))

            init = tf.random_normal_initializer(stddev=0.1)
            keys = [tf.get_variable("key_%d" % i, [args.num_units], initializer=init) for i in range(1)]
            cell = EntityNetwork(memory_slots=1,
                    memory_size=args.num_units,
                    keys=keys)
            output, final_state = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(enc_states, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            enc_state = final_state[0][0]
            weights = output[1][0]
            self.gate_weights = weights
        elif args.classifier_type == 'CNN':
            filter_shape_1 = [args.filter_size, args.embd_dim, 1, args.num_filters]
            filter_shape_2 = [args.filter_size, 1, args.num_filters, args.num_filters]

            #initialize word embedding, task embedding parameters, sentence embedding matrix
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            emb_inp = tf.expand_dims(emb_inp, -1)
            #emb_inp = tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)) # dropout is bad

            #initialize convolution, pooling parameters
            W1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b1")
            W2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b2")

            #conv1+pool1
            conv1 = tf.nn.conv2d(emb_inp, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

            #conv2+pool2
            conv2 = tf.nn.conv2d(pooled1,W2,strides=[1, 1, 1, 1],padding="VALID",name="conv2")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
            pooled2 = tf.nn.max_pool(h2, ksize=[1, h2.shape[1], 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool2")

            h_pool_flat = tf.reshape(pooled2, [-1, args.num_filters])
            enc_state = h_pool_flat
            self._logger.info('enc_state.shape: {}'.format(enc_state.shape))
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'CNN3layer':
            filter_shape_1 = [args.filter_size, args.embd_dim, 1, args.num_filters]
            filter_shape_2 = [args.filter_size, 1, args.num_filters, args.num_filters]
            filter_shape_3 = [args.filter_size, 1, args.num_filters, args.num_filters]

            #initialize word embedding, task embedding parameters, sentence embedding matrix
            emb_inp = tf.nn.embedding_lookup(self.embedding_matrix, inp)
            emb_inp = tf.expand_dims(emb_inp, -1)
            #emb_inp = tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)) # dropout is bad

            #initialize convolution, pooling parameters
            W1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b1")
            W2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b2")
            W3 = tf.Variable(tf.truncated_normal(filter_shape_3, stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b3")

            #conv1+pool1
            conv1 = tf.nn.conv2d(emb_inp, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

            #conv2+pool2
            conv2 = tf.nn.conv2d(pooled1, W2, strides=[1, 1, 1, 1], padding="VALID",name="conv2")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
            pooled2 = tf.nn.max_pool(h2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],padding='VALID',name="pool2")
            print(h2.shape)
            print(pooled2.shape)

            #conv3+pool3
            conv3 = tf.nn.conv2d(pooled2, W3, strides=[1, 1, 1, 1], padding="VALID",name="conv3")
            h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
            pooled3 = tf.nn.max_pool(h3, ksize=[1, h3.shape[1], 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool2")
            print(h3.shape)
            print(pooled3.shape)

            h_pool_flat = tf.reshape(pooled3, [-1, args.num_filters])
            enc_state = h_pool_flat
            self._logger.info('enc_state.shape: {}'.format(enc_state.shape))
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'BiGRU+CNN':
            cell = tf.contrib.rnn.GRUCell(args.num_units/ 2)
            enc_states, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, 
                    cell_bw=cell, 
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)),
                    dtype=tf.float32)
            enc_states = tf.concat(enc_states, axis=2)
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights

            filter_shape_1 = [args.filter_size, args.num_units, 1, args.num_filters]
            filter_shape_2 = [args.filter_size, 1, args.num_filters, args.num_filters]

            enc_states = tf.expand_dims(enc_states, -1)
            enc_states = tf.nn.dropout(enc_states, tf.where(self.is_training_plh, keep_rate, 1.0))

            #initialize convolution, pooling parameters
            W1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b1")
            W2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b2")

            #conv1+pool1
            conv1 = tf.nn.conv2d(enc_states, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            pooled1 = tf.nn.max_pool(h1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")

            #conv2+pool2
            conv2 = tf.nn.conv2d(pooled1,W2,strides=[1, 1, 1, 1],padding="VALID",name="conv2")
            h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
            pooled2 = tf.nn.max_pool(h2, ksize=[1, h2.shape[1], 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool2")
            h_pool_flat = tf.reshape(pooled2, [-1, args.num_filters])
            enc_state = h_pool_flat
            self._logger.info('enc_state.shape: {}'.format(enc_state.shape))
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        elif args.classifier_type == 'GRU+selfatt':
            def cal_attention(states, msk):
                # context is in the layers
                logits_att = tf.contrib.layers.fully_connected(inputs=states,
                        num_outputs=args.num_units,
                        activation_fn=tf.tanh,
                        scope='attention_0')
                logits_att = tf.contrib.layers.fully_connected(inputs=logits_att, 
                        num_outputs=1, 
                        activation_fn=None,
                        biases_initializer=None,
                        scope='attention_1')


                max_logit_att = tf.reduce_max(logits_att-1e20*(1-msk[:,:,None]), axis=1)[:,None,:]
                logits_att = tf.exp(logits_att - max_logit_att) * msk[:, :, None]
                weights = logits_att / tf.reduce_sum(logits_att, axis=1)[:, None, :]
                return weights
    
            cell = tf.contrib.rnn.GRUCell(args.num_units)
            if args.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
            states, _ = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            weights = cal_attention(states, msk)
            enc_state = tf.reduce_sum(states * weights, axis=1)
            self.gate_weights = weights
        elif args.classifier_type == 'LSTM+selfatt':
            def cal_attention(states, msk):
                # context is in the layers
                logits_att = tf.contrib.layers.fully_connected(inputs=states,
                        num_outputs=args.num_units,
                        activation_fn=tf.tanh,
                        scope='attention_0')
                logits_att = tf.contrib.layers.fully_connected(inputs=logits_att, 
                        num_outputs=1, 
                        activation_fn=None,
                        biases_initializer=None,
                        scope='attention_1')


                max_logit_att = tf.reduce_max(logits_att-1e20*(1-msk[:,:,None]), axis=1)[:,None,:]
                logits_att = tf.exp(logits_att - max_logit_att) * msk[:, :, None]
                weights = logits_att / tf.reduce_sum(logits_att, axis=1)[:, None, :]
                return weights
    
            cell = tf.contrib.rnn.LSTMCell(args.num_units, state_is_tuple=True, use_peepholes=True, cell_clip=10)
            if args.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
            states, _ = tf.nn.dynamic_rnn(cell=cell,
                    inputs=tf.nn.dropout(emb_inp, tf.where(self.is_training_plh, keep_rate, 1.0)),
                    dtype=tf.float32,
                    sequence_length=tf.to_int64(tf.reduce_sum(msk, axis=1)))
            weights = cal_attention(states, msk)
            enc_state = tf.reduce_sum(states * weights, axis=1)
            self.gate_weights = weights
        elif args.classifier_type == 'CNN+multiscale':
            emb_inp = tf.expand_dims(emb_inp, -1)
            list_filter_size = [1,2,3,4,5]
            list_pool_output = []
            for fs in list_filter_size:
                with tf.name_scope('conv_maxpool_{}'.format(fs)):
                    filter_shape = [fs, int(emb_inp.shape[2]), 1, args.num_filters]
                    conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv_b = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b")
                    conv = tf.nn.conv2d(emb_inp, conv_W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    conv = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                    conv = tf.squeeze(conv, 2)
                    print(conv.shape)
                    #pooled = tf.nn.max_pool(conv, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")
                    pooled = tf.reduce_max(tf.transpose(conv, [0, 2, 1]), [2])
                    print(pooled.shape)
                    list_pool_output.append(pooled)
            enc_state = tf.concat(list_pool_output, 1)
            weights = tf.zeros([tf.shape(inp)[0], tf.shape(inp)[1], 1])
            self.gate_weights = weights
        else:
            raise 'Encoder type {} not supported'.format(args.classifier_type)

        self._logger.info("Encoder done")
        weights = weights * msk[:, :, None]
        return enc_state, weights

def create_fclayers(self, enc_state, num_classes, scope, args):
    with tf.variable_scope(scope):
        #enc_state = tf.nn.dropout(enc_state, tf.where(self.is_training_plh, 0.5, 1.0)) # add == worse
        enc_state = tf.contrib.layers.fully_connected(enc_state, 30, scope='fc0')
        enc_state = tf.nn.tanh(enc_state)
        #enc_state = tf.nn.softmax(enc_state) # add == slow
        enc_state = tf.nn.dropout(enc_state, tf.where(self.is_training_plh, 0.5, 1.0))
    
        logits = tf.contrib.layers.fully_connected(
                inputs=enc_state,
                num_outputs=num_classes,
                activation_fn=None,
                scope='fc1')

    return logits
