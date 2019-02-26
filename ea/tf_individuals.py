import tensorflow as tf
from ea.individuals import NEAIndividualException
from sklearn.utils import shuffle
import numpy as np

from ea.tf_dna import MNISTSmallDNA
from ea.individuals import NEAIndividual
from ea.representations.tf_nodes import TFNConv2D, TFInputNode


# <editor-fold desc=Individual for small MNIST search space>
class MNISTSmallIndividual(NEAIndividual):
  DICT_DNA = 'dna'
  DICT_FITNESS = 'fitness'

  def __init__(self, **kwargs):
    super(MNISTSmallIndividual, self).__init__(**kwargs)
    if self._dna is None:
      self._dna = MNISTSmallDNA()

  def __getstate__(self):
    result = super(MNISTSmallIndividual, self).__getstate__()
    return result

  def __sess_conf(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

  def evaluate(self, **kwargs):
    X_train, Y_train = kwargs.get('X_train'), kwargs.get('Y_train')
    X_valid, Y_valid = kwargs.get('X_valid'), kwargs.get('Y_valid')
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_train is None or Y_train is None or X_valid is None or Y_valid is None or X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    folder = kwargs.get('log_path') + '/'

    def add_summary(_writer, scalar, tag, epoch):
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = scalar
      summary_value.tag = tag
      _writer.add_summary(summary, epoch)

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      # <editor-fold desc=Setup graph>
      y = tf.placeholder(tf.int64, shape=None, name='target_labels')
      x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_input')
      x_ = x
      for i, (block, filter) in enumerate(
              zip([self._dna.representation[i:i + 3] for i in range(0, len(self._dna.representation), 3)],
                  [32, 64])):
        k_size = block[1]
        p_size = block[2]
        if block[0]:
          folder += 'Conv_%i_%i_' % (k_size, p_size)
          x_ = tf.layers.conv2d(inputs=x_,
                                filters=filter,
                                kernel_size=[k_size, k_size],
                                strides=[p_size, p_size],
                                padding='SAME',
                                name='Conv2D_%i' % (i + 1),
                                activation=tf.nn.relu)
        else:
          folder += 'Conv_%i_MaxPool_%i_' % (k_size, p_size)
          x_ = tf.layers.conv2d(inputs=x_,
                                filters=filter,
                                kernel_size=[k_size, k_size],
                                strides=[1, 1],
                                padding='SAME',
                                name='Conv2D_%i' % (i + 1),
                                activation=tf.nn.relu)
          x_ = tf.layers.max_pooling2d(inputs=x_,
                                       pool_size=[p_size, p_size],
                                       strides=[p_size, p_size],
                                       padding='VALID',
                                       name='MaxPool_%i' % (i + 1))
      folder = folder[:-1]
      x_ = tf.layers.flatten(x_)
      x_ = tf.layers.dense(inputs=x_, units=1024, activation=tf.nn.relu)
      x_ = tf.layers.dense(inputs=x_, units=10)
      classes = tf.argmax(input=x_, axis=1)
      probabilities = tf.nn.softmax(x_, name='softmax')
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_, labels=y, name='cross_entropy')
      loss_operation = tf.reduce_mean(cross_entropy)
      train_step = tf.train.AdamOptimizer().minimize(loss=loss_operation,
                                                     global_step=tf.train.get_or_create_global_step())
      acc_sum = tf.reduce_sum(tf.cast(tf.equal(classes, y), tf.float32))

      # </editor-fold>

      # <editor-fold desc=Train graph>
      sess.run(tf.global_variables_initializer())

      _writer = tf.summary.FileWriter(folder, sess.graph)
      saver = tf.train.Saver()

      best = np.Inf
      wait = 0
      epoch = 0
      while wait < 5:
        # Training
        training_loss = 0
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, len(Y_train), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
          feed_dict = {x: batch_x, y: batch_y}
          loss, _ = sess.run([loss_operation, train_step], feed_dict=feed_dict)
          training_loss += (len(batch_y) * loss)
        training_loss /= len(Y_train)
        add_summary(_writer, training_loss, 'train_loss', epoch)

        # Validation
        acc_, loss_ = 0, 0
        for offset in range(0, len(Y_valid), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_valid[offset:end], Y_valid[offset:end]
          sess_out = sess.run([loss_operation, acc_sum], feed_dict={x: batch_x, y: batch_y})
          loss_ += (sess_out[0] * len(batch_y))
          acc_ += sess_out[1]
        acc_ /= len(Y_valid)
        loss_ /= len(Y_valid)

        add_summary(_writer, acc_, 'valid_acc', epoch)
        add_summary(_writer, loss_, 'valid_loss', epoch)
        if ((loss_ + 1e-3) < best):
          best = loss_
          wait = 0

          saver.save(sess, folder + '/latest.ckpt')
        else:
          wait += 1
        epoch += 1
        _writer.flush()

      _writer.close()
      # </editor-fold>

      # restore latest checkpoint
      saver.restore(sess, folder + '/latest.ckpt')

      # <editor-fold desc=Test graph>
      fitness = 0.0
      for offset in range(0, len(Y_test), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
        prob = sess.run([probabilities], feed_dict={x: batch_x, y: batch_y})[0]
        fitness += sum([i[l] for i, l in zip(prob, batch_y) if i[l] > .5])
      # </editor-fold>
      trainable_variables = tf.trainable_variables()
      trained_weights = sess.run(trainable_variables)

      trained_state = dict([(key.name, value) for key, value in zip(trainable_variables, trained_weights)])

    self._fitness = fitness / len(Y_test)
    return trained_state

  def test(self, **kwargs):
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')
    if not isinstance(self._meta_data, dict):
      raise NEAIndividualException('No meta data provided to reconstruct trained network!')

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    tensorboard_folder = kwargs.get('folder', None)

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_input')
      x_ = x
      for i, (block, filter) in enumerate(
              zip([self._dna.representation[i:i + 3] for i in range(0, len(self._dna.representation), 3)],
                  [32, 64])):
        k_size = block[1]
        p_size = block[2]
        kernel_weights = self._meta_data.get('Conv2D_%i/kernel:0' % i, None)
        kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
        bias_weights = self._meta_data.get('Conv2D_%i/bias:0' % i, None)
        bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
        if block[0]:
          x_ = tf.layers.conv2d(inputs=x_,
                                filters=filter,
                                kernel_size=[k_size, k_size],
                                strides=[p_size, p_size],
                                padding='SAME',
                                name='Conv2D_%i' % (i + 1),
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)
        else:
          x_ = tf.layers.conv2d(inputs=x_,
                                filters=filter,
                                kernel_size=[k_size, k_size],
                                strides=[1, 1],
                                padding='SAME',
                                name='Conv2D_%i' % (i + 1),
                                activation=tf.nn.relu,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer)
          x_ = tf.layers.max_pooling2d(inputs=x_,
                                       pool_size=[p_size, p_size],
                                       strides=[p_size, p_size],
                                       padding='VALID',
                                       name='MaxPool_%i' % (i + 1))
      x_ = tf.layers.flatten(x_)
      kernel_weights = self._meta_data.get('dense/kernel:0' % i, None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense/bias:0' % i, None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=x_, units=1024, activation=tf.nn.relu,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
      kernel_weights = self._meta_data.get('dense_1/kernel:0' % i, None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense_1/bias:0' % i, None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=x_, units=10,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
      classes = tf.argmax(input=x_, axis=1)
      sess.run(tf.global_variables_initializer())
      if tensorboard_folder is not None:
        _writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

      # <editor-fold desc=Test graph>
      accuracy = 0.0
      for offset in range(0, len(Y_test), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
        _class = sess.run([classes], feed_dict={x: batch_x})[0]
        accuracy += sum([c_ == y_ for c_, y_ in zip(_class, batch_y)])
      accuracy = accuracy / len(Y_test)
      # </editor-fold>

      global_variables = tf.global_variables()
      total_parameters = 0
      for variable in global_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
    return accuracy, total_parameters

  def __pow__(self, other):
    super(MNISTSmallIndividual, self).__pow__(other)
    state = self.__getstate__()
    result = list()
    for dna in self._dna ** other._dna:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTSmallIndividual(**state))
    return result

  def __mod__(self, probability):
    super(MNISTSmallIndividual, self).__mod__(probability)
    state = self.__getstate__()
    result = list()
    for dna in self._dna % probability:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTSmallIndividual(**state))
    return result


# </editor-fold>

from ea.tf_dna import TFKTreeDNA
import os


# <editor-fold desc=Individual for k-ary tree MNIST search space>
class MNISTKTreeIndividual(NEAIndividual):
  def __init__(self, **kwargs):
    super(MNISTKTreeIndividual, self).__init__(**kwargs)
    if self._dna is None:
      self._dna = TFKTreeDNA()
    self._parameters = kwargs.get('parameters', False)
    self._param_min = kwargs.get('parameters_minimum', 2257910)
    self._param_max = kwargs.get('parameters_maximum', 11289550)

  def __getstate__(self):
    result = super(MNISTKTreeIndividual, self).__getstate__()
    result['parameters'] = self._parameters
    result['parameters_minimum'] = self._param_min
    result['parameters_maximum'] = self._param_max
    return result

  def __pow__(self, other):
    super(MNISTKTreeIndividual, self).__pow__(other)
    state = self.__getstate__()
    result = list()
    for dna in self._dna ** other._dna:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTKTreeIndividual(**state))
    return result

  def __mod__(self, probability):
    super(MNISTKTreeIndividual, self).__mod__(probability)
    state = self.__getstate__()
    result = list()
    for dna in self._dna % probability:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTKTreeIndividual(**state))
    return result

  @property
  def parameters(self):
    return self._parameters

  @parameters.setter
  def parameters(self, parameters):
    self._parameters = parameters

  @property
  def parameters_minimum(self):
    return self._param_min

  @parameters_minimum.setter
  def parameters_minimum(self, _min):
    self._param_min = _min

  @property
  def parameters_maximum(self):
    return self._param_max

  @parameters_maximum.setter
  def parameters_maximum(self, _max):
    self._param_max = _max

  def __sess_conf(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

  def evaluate(self, **kwargs):
    X_train, Y_train = kwargs.get('X_train'), kwargs.get('Y_train')
    X_valid, Y_valid = kwargs.get('X_valid'), kwargs.get('Y_valid')
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_train is None or Y_train is None or X_valid is None or Y_valid is None or X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    # folder = kwargs.get('log_path') + '/'+datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    suffix = ''
    idx = 0
    folder = kwargs.get('log_path') + '/%i' % (hash(self._dna))
    while os.path.isdir(folder + suffix):
      suffix = '_%02i' % idx
      idx += 1
    folder = folder + suffix

    def add_summary(_writer, scalar, tag, epoch):
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = scalar
      summary_value.tag = tag
      _writer.add_summary(summary, epoch)

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      # <editor-fold desc=Build graph>
      y = tf.placeholder(tf.int64, shape=None, name='target_labels')
      feature_extraction = self._dna._representation.create_nn()
      x = tf.get_default_graph().get_tensor_by_name('X:0')

      x_ = tf.layers.dense(inputs=feature_extraction, units=1024, activation=tf.nn.relu)
      x_ = tf.layers.dense(inputs=x_, units=10)
      classes = tf.argmax(input=x_, axis=1)
      probabilities = tf.nn.softmax(x_, name='softmax')
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_, labels=y, name='cross_entropy')
      loss_operation = tf.reduce_mean(cross_entropy)
      train_step = tf.train.AdamOptimizer().minimize(loss=loss_operation,
                                                     global_step=tf.train.get_or_create_global_step())
      acc_sum = tf.reduce_sum(tf.cast(tf.equal(classes, y), tf.float32))
      sess.run(tf.global_variables_initializer())

      _writer = tf.summary.FileWriter(folder, sess.graph)
      saver = tf.train.Saver()
      # </editor-fold>

      # <editor-fold desc=Train graph>
      best = np.Inf
      wait = 0
      epoch = 0
      while wait < 5:
        # Training
        training_loss = 0
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, len(Y_train), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
          feed_dict = {x: batch_x, y: batch_y}
          loss, _ = sess.run([loss_operation, train_step], feed_dict=feed_dict)
          training_loss += (len(batch_y) * loss)
        training_loss /= len(Y_train)
        add_summary(_writer, training_loss, 'train_loss', epoch)

        # Validation
        acc_, loss_ = 0, 0
        for offset in range(0, len(Y_valid), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_valid[offset:end], Y_valid[offset:end]
          sess_out = sess.run([loss_operation, acc_sum], feed_dict={x: batch_x, y: batch_y})
          loss_ += (sess_out[0] * len(batch_y))
          acc_ += sess_out[1]
        acc_ /= len(Y_valid)
        loss_ /= len(Y_valid)

        # early stopping and checkpoint
        add_summary(_writer, acc_, 'valid_acc', epoch)
        add_summary(_writer, loss_, 'valid_loss', epoch)
        if loss_ < best:
          saver.save(sess, folder + '/latest.ckpt')
        if ((loss_ + 1e-3) < best):
          best = loss_
          wait = 0
        else:
          wait += 1
        epoch += 1
        _writer.flush()

      _writer.close()
      # </editor-fold>

      # restore latest checkpoint
      saver.restore(sess, folder + '/latest.ckpt')

      # <editor-fold desc=Test graph>
      fitness = 0.0
      for offset in range(0, len(Y_test), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
        prob = sess.run([probabilities], feed_dict={x: batch_x, y: batch_y})[0]
        fitness += sum([i[l] for i, l in zip(prob, batch_y) if i[l] > .5])
      fitness = fitness / len(Y_test)
      # </editor-fold>

      global_variables = tf.global_variables()
      total_parameters = 0
      for variable in global_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
      trainable_variables = tf.trainable_variables()
      trained_weights = sess.run(trainable_variables)

      trained_state = dict([(key.name, value) for key, value in zip(trainable_variables, trained_weights)])

    if self._parameters:
      # shift 0 to param_min
      dim2 = max(0, total_parameters - self._param_min)
      # rescale in the way that param_max = 1
      dim2 = min(dim2 / (self._param_max - self._param_min), 1)
      # invert axis so that param_min = 1 and param_max = 0
      dim2 = 1 - dim2
      # [(p_2-norm{f,dim2})^2]/2
      self._fitness = (fitness ** 2 + dim2 ** 2) / 2
    else:
      self._fitness = fitness
    return trained_state

  def test(self, **kwargs):
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')
    if not isinstance(self._meta_data, dict):
      raise NEAIndividualException('No meta data provided to reconstruct trained network!')

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    tensorboard_folder = kwargs.get('folder', None)

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      feature_extraction = self._dna._representation.create_nn(**self._meta_data)
      x = tf.get_default_graph().get_tensor_by_name('X:0')

      kernel_weights = self._meta_data.get('dense/kernel:0', None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense/bias:0', None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=feature_extraction, units=1024, activation=tf.nn.relu,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

      kernel_weights = self._meta_data.get('dense_1/kernel:0', None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense_1/bias:0', None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=x_, units=10,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
      # probabilities = tf.nn.softmax(x_, name='softmax')
      classes = tf.argmax(input=x_, axis=1)

      sess.run(tf.global_variables_initializer())
      if tensorboard_folder is not None:
        _writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

      # <editor-fold desc=Test graph>
      misclass = kwargs.get('misclassification', False)
      accuracy = 0.0
      if misclass:
        classification = list()
      for offset in range(0, len(Y_test), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
        _class = sess.run([classes], feed_dict={x: batch_x})[0]
        _classification = [c_ == y_ for c_, y_ in zip(_class, batch_y)]
        accuracy += sum(_classification)
        if misclass:
          classification.extend(_classification)
        # prob = sess.run([probabilities], feed_dict={x: batch_x})[0]
        # accuracy += sum([i[l] for i, l in zip(prob, batch_y) if i[l] > .5])
      accuracy = accuracy / len(Y_test)
      if misclass:
        misclassifications = [i for i, b in enumerate(classification) if not b]
      # </editor-fold>

      global_variables = tf.global_variables()
      total_parameters = 0
      for variable in global_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
    result = [accuracy, total_parameters]
    if misclass:
      result.append(misclassifications)
    return tuple(result)

  def flops(self, **kwargs):
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    shape = kwargs.get('shape')
    if shape is None:
      shape = (1, 28, 28, 1)

    for n in self.dna.representation.nodes:
      node = n[0]
      if isinstance(node, TFInputNode):
        node.shape = shape
    with tf.Session(config=config, graph=tf.Graph()) as sess:
      feature_extraction = self._dna._representation.create_nn(**self._meta_data)
      x = tf.get_default_graph().get_tensor_by_name('X:0')

      kernel_weights = self._meta_data.get('dense/kernel:0', None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense/bias:0', None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=feature_extraction, units=1024, activation=tf.nn.relu,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

      kernel_weights = self._meta_data.get('dense_1/kernel:0', None)
      kernel_initializer = tf.constant_initializer(value=kernel_weights) if kernel_weights is not None else None
      bias_weights = self._meta_data.get('dense_1/bias:0', None)
      bias_initializer = tf.constant_initializer(value=bias_weights) if bias_weights is not None else None
      x_ = tf.layers.dense(inputs=x_, units=10,
                           kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
      classes = tf.argmax(input=x_, axis=1)
      opts = tf.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.profiler.profile(sess.graph, run_meta=tf.RunMetadata(), cmd='op', options=opts)
    return flops.total_float_ops

  pass


# </editor-fold>

# <editor-fold desc=Individual for k-ary tree MNIST search space but training only the last dense layers>
class MNISTKTreeIndividualPartialTrain(MNISTKTreeIndividual):
  def __sess_conf(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

  def evaluate(self, **kwargs):
    X_train, Y_train = kwargs.get('X_train'), kwargs.get('Y_train')
    X_valid, Y_valid = kwargs.get('X_valid'), kwargs.get('Y_valid')
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_train is None or Y_train is None or X_valid is None or Y_valid is None or X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')

    for node, _ in self.dna.representation.nodes:
      if isinstance(node, TFNConv2D):
        node.trainable = False

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    suffix = ''
    idx = 0
    folder = kwargs.get('log_path') + '/%i' % (hash(self._dna))
    while os.path.isdir(folder + suffix):
      suffix = '_%02i' % idx
      idx += 1
    folder = folder + suffix

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      # <editor-fold desc=Build graph>
      y = tf.placeholder(tf.int64, shape=None, name='target_labels')
      feature_extraction = self._dna._representation.create_nn()
      x = tf.get_default_graph().get_tensor_by_name('X:0')

      x_ = tf.layers.dense(inputs=feature_extraction, units=1024, activation=tf.nn.relu)
      x_ = tf.layers.dense(inputs=x_, units=10)
      classes = tf.argmax(input=x_, axis=1)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_, labels=y, name='cross_entropy')
      loss_operation = tf.reduce_mean(cross_entropy)
      train_step = tf.train.AdamOptimizer().minimize(loss=loss_operation,
                                                     global_step=tf.train.get_or_create_global_step())
      acc_sum = tf.reduce_sum(tf.cast(tf.equal(classes, y), tf.float32))
      init_vars = tf.global_variables_initializer()
      saver = tf.train.Saver()
      # </editor-fold>

      accs = list()
      test_n = 3
      for _ in range(test_n):
        # <editor-fold desc=Train graph>
        sess.run(init_vars)
        best = np.Inf
        wait = 0
        epoch = 0
        while wait < 5:
          # Training
          training_loss = 0
          X_train, Y_train = shuffle(X_train, Y_train)
          for offset in range(0, len(Y_train), batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            feed_dict = {x: batch_x, y: batch_y}
            loss, _ = sess.run([loss_operation, train_step], feed_dict=feed_dict)
            training_loss += (len(batch_y) * loss)
          training_loss /= len(Y_train)

          # Validation
          acc_, loss_ = 0, 0
          for offset in range(0, len(Y_valid), batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_valid[offset:end], Y_valid[offset:end]
            sess_out = sess.run([loss_operation, acc_sum], feed_dict={x: batch_x, y: batch_y})
            loss_ += (sess_out[0] * len(batch_y))
            acc_ += sess_out[1]
          acc_ /= len(Y_valid)
          loss_ /= len(Y_valid)

          # early stopping and checkpoint
          if loss_ < best:
            saver.save(sess, folder + '/latest.ckpt')
          if ((loss_ + 1e-3) < best):
            best = loss_
            wait = 0
          else:
            wait += 1
          epoch += 1
        # </editor-fold>

        # restore latest checkpoint
        saver.restore(sess, folder + '/latest.ckpt')

        # <editor-fold desc=Test graph>
        accuracy = 0.0
        for offset in range(0, len(Y_test), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
          _class = sess.run([classes], feed_dict={x: batch_x})[0]
          accuracy += sum([c_ == y_ for c_, y_ in zip(_class, batch_y)])
        accuracy = accuracy / len(Y_test)
        accs.append(accuracy)
        # </editor-fold>
      fitness = sum(accs) / test_n

      global_variables = tf.global_variables()
      total_parameters = 0
      for variable in global_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters

    if self._parameters:
      # shift 0 to param_min
      dim2 = max(0, total_parameters - self._param_min)
      # rescale in the way that param_max = 1
      dim2 = min(dim2 / (self._param_max - self._param_min), 1)
      # invert axis so that param_min = 1 and param_max = 0
      dim2 = 1 - dim2
      # [(p_2-norm{f,dim2})^2]/2
      self._fitness = (fitness ** 2 + dim2 ** 2) / 2
    else:
      self._fitness = fitness
    return None

  def test(self, **kwargs):
    X_train, Y_train = kwargs.get('X_train'), kwargs.get('Y_train')
    X_valid, Y_valid = kwargs.get('X_valid'), kwargs.get('Y_valid')
    X_test, Y_test = kwargs.get('X_test'), kwargs.get('Y_test')

    if X_train is None or Y_train is None or X_valid is None or Y_valid is None or X_test is None or Y_test is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')

    X_test_final, Y_test_final = kwargs.get('X'), kwargs.get('Y')

    if X_test_final is None or Y_test_final is None:
      raise NEAIndividualException('Train, validation and test data set is needed!')

    for node, _ in self.dna.representation.nodes:
      if isinstance(node, TFNConv2D):
        node.trainable = True

    batch_size = kwargs.get('batch_size', 32)
    config = kwargs.get('session_config')
    if config is None:
      config = self.__sess_conf()

    folder = kwargs.get('folder', None)

    def add_summary(_writer, scalar, tag, epoch):
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = scalar
      summary_value.tag = tag
      _writer.add_summary(summary, epoch)

    with tf.Session(config=config, graph=tf.Graph()) as sess:
      # <editor-fold desc=Build graph>
      y = tf.placeholder(tf.int64, shape=None, name='target_labels')
      feature_extraction = self._dna._representation.create_nn()
      x = tf.get_default_graph().get_tensor_by_name('X:0')

      x_ = tf.layers.dense(inputs=feature_extraction, units=1024, activation=tf.nn.relu)
      x_ = tf.layers.dense(inputs=x_, units=10)
      classes = tf.argmax(input=x_, axis=1)
      probabilities = tf.nn.softmax(x_, name='softmax')
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_, labels=y, name='cross_entropy')
      loss_operation = tf.reduce_mean(cross_entropy)
      train_step = tf.train.AdamOptimizer().minimize(loss=loss_operation,
                                                     global_step=tf.train.get_or_create_global_step())
      acc_sum = tf.reduce_sum(tf.cast(tf.equal(classes, y), tf.float32))
      sess.run(tf.global_variables_initializer())

      _writer = tf.summary.FileWriter(folder, sess.graph)
      saver = tf.train.Saver()
      # </editor-fold>

      # <editor-fold desc=Train graph>
      best = np.Inf
      wait = 0
      epoch = 0
      while wait < 5:
        # Training
        training_loss = 0
        X_train, Y_train = shuffle(X_train, Y_train)
        for offset in range(0, len(Y_train), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
          feed_dict = {x: batch_x, y: batch_y}
          loss, _ = sess.run([loss_operation, train_step], feed_dict=feed_dict)
          training_loss += (len(batch_y) * loss)
        training_loss /= len(Y_train)
        add_summary(_writer, training_loss, 'train_loss', epoch)

        # Validation
        acc_, loss_ = 0, 0
        for offset in range(0, len(Y_valid), batch_size):
          end = offset + batch_size
          batch_x, batch_y = X_valid[offset:end], Y_valid[offset:end]
          sess_out = sess.run([loss_operation, acc_sum], feed_dict={x: batch_x, y: batch_y})
          loss_ += (sess_out[0] * len(batch_y))
          acc_ += sess_out[1]
        acc_ /= len(Y_valid)
        loss_ /= len(Y_valid)

        # early stopping and checkpoint
        add_summary(_writer, acc_, 'valid_acc', epoch)
        add_summary(_writer, loss_, 'valid_loss', epoch)
        if loss_ < best:
          saver.save(sess, folder + '/latest.ckpt')
        if ((loss_ + 1e-3) < best):
          best = loss_
          wait = 0
        else:
          wait += 1
        epoch += 1
        _writer.flush()

      _writer.close()
      # </editor-fold>

      # restore latest checkpoint
      saver.restore(sess, folder + '/latest.ckpt')

      # <editor-fold desc=Test graph>
      accuracy = 0.0
      for offset in range(0, len(Y_test_final), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test_final[offset:end], Y_test_final[offset:end]
        _class = sess.run([classes], feed_dict={x: batch_x})[0]
        accuracy += sum([c_ == y_ for c_, y_ in zip(_class, batch_y)])
      accuracy = accuracy / len(Y_test_final)

      fitness = 0.0
      for offset in range(0, len(Y_test), batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_test[offset:end], Y_test[offset:end]
        prob = sess.run([probabilities], feed_dict={x: batch_x, y: batch_y})[0]
        fitness += sum([i[l] for i, l in zip(prob, batch_y) if i[l] > .5])
      fitness = fitness / len(Y_test)
      # </editor-fold>

      # <editor-fold desc=Get # of trainable weights>
      global_variables = tf.global_variables()
      total_parameters = 0
      for variable in global_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
          variable_parameters *= dim.value
        total_parameters += variable_parameters
      # </editor-fold>

      if self._parameters:
        # shift 0 to param_min
        dim2 = max(0, total_parameters - self._param_min)
        # rescale in the way that param_max = 1
        dim2 = min(dim2 / (self._param_max - self._param_min), 1)
        # invert axis so that param_min = 1 and param_max = 0
        dim2 = 1 - dim2
        # [(p_2-norm{f,dim2})^2]/2
        fitness = (fitness ** 2 + dim2 ** 2) / 2

    return fitness, accuracy, total_parameters

  def __pow__(self, other):
    super(MNISTKTreeIndividual, self).__pow__(other)
    state = self.__getstate__()
    result = list()
    for dna in self._dna ** other._dna:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTKTreeIndividualPartialTrain(**state))
    return result

  def __mod__(self, probability):
    super(MNISTKTreeIndividual, self).__mod__(probability)
    state = self.__getstate__()
    result = list()
    for dna in self._dna % probability:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNISTKTreeIndividualPartialTrain(**state))
    return result
# </editor-fold>
