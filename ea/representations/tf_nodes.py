from ea.representations.nodes import AbstractNode, AbstractInputNode, AbstractOutputNode
from ea.representations.k_ary_tree import TreeException
import tensorflow as tf
import numpy as np
import math


class TFNodeUtil():
  CHANNELS_LAST = 'channels_last'
  CHANNELS_FIRST = 'channels_first'
  SAME = 'same'
  VALID = 'valid'

  @staticmethod
  def get_bhwc(size_, data_format):
    if data_format == TFNodeUtil.CHANNELS_LAST:
      return tuple(size_)
    else:
      return (size_[0], size_[2], size_[3], size_[1])

  @staticmethod
  def from_bhwc(b, h, w, c, data_format):
    return (b, h, w, c) if data_format == TFNodeUtil.CHANNELS_LAST else (b, c, h, w)

  pass


class TFNConv2D(AbstractNode):
  Index = 0

  def __init__(self, **kwargs):
    super(TFNConv2D, self).__init__(**kwargs)
    self._filters = kwargs.get('filters', round(np.random.beta(2, 2) * 63 + 3))
    self._kernel_w = kwargs.get('kernel_width', round(np.random.beta(2, 2) * 9 + .5))
    self._kernel_h = kwargs.get('kernel_height', round(np.random.beta(2, 2) * 9 + .5))
    self._stride_w = kwargs.get('stride_width', 1)
    self._stride_h = kwargs.get('stride_height', 1)
    # self._dilation_w = kwargs.get('dilation_width', 1)
    # self._dilation_h = kwargs.get('dilation_height', 1)
    self._padding = kwargs.get('padding', TFNodeUtil.SAME)
    self._data_format = kwargs.get('data_format', TFNodeUtil.CHANNELS_LAST)
    self._activation = kwargs.get('activation')
    self._use_bias = kwargs.get('use_bias', True)
    self._kernel_initializer = kwargs.get('kernel_initializer')
    self._bias_initializer = kwargs.get('bias_initializer')
    self._kernel_regularizer = kwargs.get('kernel_regularizer')
    self._bias_regularizer = kwargs.get('bias_regularizer')
    self._activity_regularizer = kwargs.get('activity_regularizer')
    self._kernel_constraint = kwargs.get('kernel_constraint')
    self._bias_constraint = kwargs.get('bias_constraint')
    self._trainable = kwargs.get('trainable', True)
    self._name = kwargs.get('name')
    self._reuse = kwargs.get('reuse')

    if not self._bias_initializer:
      self._bias_initializer = tf.initializers.zeros()

    if not self._name:
      TFNConv2D.Index += 1
      self._name = "Conv2D_%02i" % TFNConv2D.Index
    pass

  def __getstate__(self):
    result = super(TFNConv2D, self).__getstate__()
    result['filters'] = self._filters
    result['kernel_width'] = self._kernel_w
    result['kernel_height'] = self._kernel_h
    result['stride_width'] = self._stride_w
    result['stride_height'] = self._stride_h
    # result['dilation_width'] = self._dilation_w
    # result['dilation_height'] = self._dilation_h
    result['padding'] = self._padding
    result['data_format'] = self._data_format
    result['activation'] = self._activation
    result['use_bias'] = self._use_bias
    result['kernel_initializer'] = self._kernel_initializer
    result['bias_initializer'] = self._bias_initializer
    result['kernel_regularizer'] = self._kernel_regularizer
    result['bias_regularizer'] = self._bias_regularizer
    result['activity_regularizer'] = self._activity_regularizer
    result['kernel_constraint'] = self._kernel_constraint
    result['bias_constraint'] = self._bias_constraint
    result['trainable'] = self._trainable
    result['name'] = self._name
    result['reuse'] = self._reuse
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __copy__(self):
    state = self.__getstate__()
    state['name'] = None
    result = TFNConv2D(**state)
    return result

  def __str__(self):
    return "Conv2D: %s, filter: %i, kernel: (%i, %i), stride: (%i, %i), padding: %s" % \
           (self._name, self._filters, self._kernel_h, self._kernel_w, self._stride_h, self._stride_w, self._padding)

  def hash_without_name(self):
    if self._pseudo_hash is not None:
      return self._pseudo_hash
    result = 0
    for elem in [
      self._filters,
      self._kernel_w,
      self._kernel_h,
      self._stride_w,
      self._stride_h,
      # self._dilation_w,
      # self._dilation_h,
      int.from_bytes(self._padding.encode('utf-8'), byteorder='big'),
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
      int.from_bytes(str(self._activation.__class__).encode('utf-8'),
                     byteorder='big') if self._activation != None else 0,
      self._use_bias,
      int.from_bytes(str(self._kernel_initializer.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_initializer != None else 0,
      int.from_bytes(str(self._bias_initializer.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_initializer != None else 0,
      int.from_bytes(str(self._kernel_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_initializer != None else 0,
      int.from_bytes(str(self._bias_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_regularizer != None else 0,
      int.from_bytes(str(self._activity_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._activity_regularizer != None else 0,
      int.from_bytes(str(self._kernel_constraint.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_constraint != None else 0,
      int.from_bytes(str(self._bias_constraint.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_constraint != None else 0,
      self._trainable,
      self._reuse,
    ]:
      result = result * 31 + hash(elem)
    self._pseudo_hash = result
    return result

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    result = super(TFNConv2D, self).__hash__()
    for elem in [
      self._filters,
      self._kernel_w,
      self._kernel_h,
      self._stride_w,
      self._stride_h,
      # self._dilation_w,
      # self._dilation_h,
      int.from_bytes(self._padding.encode('utf-8'), byteorder='big'),
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
      int.from_bytes(str(self._activation.__class__).encode('utf-8'),
                     byteorder='big') if self._activation != None else 0,
      self._use_bias,
      int.from_bytes(str(self._kernel_initializer.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_initializer != None else 0,
      int.from_bytes(str(self._bias_initializer.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_initializer != None else 0,
      int.from_bytes(str(self._kernel_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_initializer != None else 0,
      int.from_bytes(str(self._bias_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_regularizer != None else 0,
      int.from_bytes(str(self._activity_regularizer.__class__).encode('utf-8'),
                     byteorder='big') if self._activity_regularizer != None else 0,
      int.from_bytes(str(self._kernel_constraint.__class__).encode('utf-8'),
                     byteorder='big') if self._kernel_constraint != None else 0,
      int.from_bytes(str(self._bias_constraint.__class__).encode('utf-8'),
                     byteorder='big') if self._bias_constraint != None else 0,
      self._trainable,
      int.from_bytes(self._name.encode('utf-8'), byteorder='big'),
      self._reuse,
    ]:
      result = result * 31 + hash(elem)
    self._hash = result
    return result

  def similar(self, other):
    if not super(TFNConv2D, self).similar(other) or \
            self._filters != other._filters or \
            self._kernel_w != other._kernel_w or \
            self._kernel_h != other._kernel_h or \
            self._stride_w != other._stride_w or \
            self._stride_h != other._stride_h or \
            self._padding != other._padding or \
            self._data_format != other._data_format:
      return False
    return True

  def __eq__(self, other):
    def check_class(self_object, other_object):
      return self_object != other_object and \
             ((self_object != None and other_object != None and self_object.__class__ != other_object.__class__) or
              (self_object == None or other_object == None))

    if not super(TFNConv2D, self).__eq__(other) or \
            self._filters != other._filters or \
            self._kernel_w != other._kernel_w or \
            self._kernel_h != other._kernel_h or \
            self._stride_w != other._stride_w or \
            self._stride_h != other._stride_h or \
            self._padding != other._padding or \
            self._data_format != other._data_format or \
            check_class(self._activation, other._activation) or \
            self._use_bias != other._use_bias or \
            check_class(self._kernel_initializer, other._kernel_initializer) or \
            check_class(self._bias_initializer, other._bias_initializer) or \
            check_class(self._kernel_regularizer, other._kernel_regularizer) or \
            check_class(self._bias_regularizer, other._bias_regularizer) or \
            check_class(self._activity_regularizer, other._activity_regularizer) or \
            check_class(self._kernel_constraint, other._kernel_constraint) or \
            check_class(self._bias_constraint, other._bias_constraint) or \
            self._trainable != other._trainable or \
            self._name != other._name or \
            self._reuse != other._reuse:
      # self._dilation_w != other._dilation_w or self._dilation_h != other._dilation_h or \
      return False
    return True

  @property
  def data_format(self):
    return self._data_format

  @property
  def trainable(self):
    return self._trainable

  @trainable.setter
  def trainable(self, t):
    self._trainable = t

  def outputsize(self, inputsizes: list):
    if len(inputsizes) == 0:
      raise TreeException('Error input list is empty!')
    b, h, w, _ = TFNodeUtil.get_bhwc(inputsizes[0], self._data_format)
    bhw = (b, h, w)
    c = 0
    for input_s in inputsizes:
      _b, _h, _w, _c = TFNodeUtil.get_bhwc(input_s, self._data_format)
      if (_b, _h, _w) != bhw:
        raise TreeException('All input nodes must have same batchsize, height and width!\n'
                             'Found: %s, %s, %s and %s, %s, %s' % (b, h, w, _b, _h, _w))
      if (_h is not None and _h <= 0) or \
              (_w is not None and _w <= 0) or \
              (_c is not None and _c <= 0):
        raise TreeException('Every input dimension must be greater than 0!\n'
                             'Found h:%s, w:%s, c:%s' % (_h, _w, _c))
      c += _c

    out_w = math.ceil((w if self._padding == TFNodeUtil.SAME else w - self._kernel_w + 1) / self._stride_w)
    out_h = math.ceil((h if self._padding == TFNodeUtil.SAME else h - self._kernel_h + 1) / self._stride_h)
    if self._data_format == TFNodeUtil.CHANNELS_LAST:
      return [b, out_h, out_w, self._filters]
    else:
      return [b, self._filters, out_h, out_w]

  def min_inputsize(self, outputsize):
    b, h, w, c = TFNodeUtil.get_bhwc(outputsize, self._data_format)
    if self._padding == TFNodeUtil.SAME:
      _h = (h - 1) * self._stride_h + 1
      _w = (w - 1) * self._stride_w + 1
    else:
      _h = (h - 1) * self._stride_h + self._kernel_h
      _w = (w - 1) * self._stride_w + self._kernel_w

    return (b, _h, _w, 1) if self._data_format == TFNodeUtil.CHANNELS_LAST else (b, 1, _h, _w)

  def max_inputsize(self, outputsize):
    b, h, w, c = TFNodeUtil.get_bhwc(outputsize, self._data_format)
    if self._padding == TFNodeUtil.SAME:
      _h = h * self._stride_h
      _w = w * self._stride_w
    else:
      _h = h * self._stride_h + self._kernel_h - 1
      _w = w * self._stride_w + self._kernel_w - 1
    return (b, _h, _w, 1) if self._data_format == TFNodeUtil.CHANNELS_LAST else (b, 1, _h, _w)

  @staticmethod
  def parameter_range(inputsize, outputsize, padding=None, data_format=TFNodeUtil.CHANNELS_LAST):
    def get_bhwc(size_):
      if data_format == TFNodeUtil.CHANNELS_LAST:
        return tuple(size_)
      else:
        return (size_[0], size_[2], size_[3], size_[1])

    b_i, h_i, w_i, c_i = get_bhwc(inputsize)
    b_o, h_o, w_o, c_o = get_bhwc(outputsize)

    def stride(in_, out_):
      if out_ == 1:
        return [in_]
      else:
        lower_limit = math.ceil(in_ / out_)
        upper_limit = math.ceil(in_ / (out_ - 1)) - 1
        if lower_limit == 0 or \
                math.ceil(in_ / lower_limit) < out_ or \
                upper_limit == 0 or \
                math.ceil(in_ / upper_limit) > out_:
          return None
        return list(range(lower_limit, upper_limit + 1))

    def filter(in_, out_):
      lower_limit = 1
      upper_limit = in_ - out_ + 1
      return list(range(lower_limit, upper_limit + 1))

    def with_padding(padding):
      if padding == TFNodeUtil.SAME:
        h_range = stride(h_i, h_o)
        w_range = stride(w_i, w_o)
        if h_range is None or w_range is None:
          return []

        return [{'padding': padding, 'stride_height': h, 'stride_width': w} for h in h_range for w in w_range]
      elif padding == TFNodeUtil.VALID:
        result = []
        for f_w in filter(w_i, w_o):
          for f_h in filter(h_i, h_o):
            h_range = stride(h_i - f_h + 1, h_o)
            w_range = stride(w_i - f_w + 1, w_o)
            if h_range is None or w_range is None:
              continue
            result += [{'padding': padding,
                        'kernel_height': f_h,
                        'kernel_width': f_w,
                        'stride_height': h,
                        'stride_width': w} for h in h_range for w in w_range]
        return result

    if padding == None:
      result = []
      for p in [TFNodeUtil.VALID, TFNodeUtil.SAME]:
        result += with_padding(p)
    else:
      result = with_padding(padding)
    return result

  def create_nn(self, inputs: list, **kwargs):
    with tf.variable_scope(name_or_scope=self._name, reuse=tf.AUTO_REUSE):
      if len(inputs) > 1:
        _input = tf.concat(values=inputs, axis=3 if self._data_format == TFNodeUtil.CHANNELS_LAST else 1)
      else:
        if len(inputs) == 0:
          raise TreeException('Error input list is empty!')
        _input = inputs[0]

      kernel_weights = kwargs.get(self._name + '/conv2d/kernel:0', None)
      if kernel_weights is not None:
        kernel_initializer = tf.constant_initializer(value=kernel_weights)
      else:
        kernel_initializer = self._kernel_initializer
      bias_weights = kwargs.get(self._name + '/conv2d/bias:0', None)
      if bias_weights is not None:
        bias_initializer = tf.constant_initializer(value=bias_weights)
      else:
        bias_initializer = self._bias_initializer

      result = tf.layers.conv2d(inputs=_input,
                                filters=self._filters,
                                kernel_size=(self._kernel_h, self._kernel_w),
                                strides=(self._stride_h, self._stride_w),
                                padding=self._padding,
                                data_format=self._data_format,
                                # dilation_rate=(self._dilation_h, self._dilation_w),
                                activation=self._activation,
                                use_bias=self._use_bias,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=self._kernel_regularizer,
                                bias_regularizer=self._bias_regularizer,
                                activity_regularizer=self._activity_regularizer,
                                kernel_constraint=self._kernel_constraint,
                                bias_constraint=self._bias_constraint,
                                trainable=self._trainable,
                                reuse=self._reuse)
    return result

  pass


class TFNMaxPool2D(AbstractNode):
  Index = 0

  def __init__(self, **kwargs):
    super(TFNMaxPool2D, self).__init__(**kwargs)
    self._pool_w = kwargs.get('pool_width', round(np.random.beta(2, 2) * 6 + .5))
    self._pool_h = kwargs.get('pool_height', round(np.random.beta(2, 2) * 6 + .5))
    self._stride_w = kwargs.get('stride_width', self._pool_w)
    self._stride_h = kwargs.get('stride_height', self._pool_h)
    self._padding = kwargs.get('padding', TFNodeUtil.VALID)
    self._data_format = kwargs.get('data_format', TFNodeUtil.CHANNELS_LAST)
    self._name = kwargs.get('name')
    if not self._name:
      TFNMaxPool2D.Index += 1
      self._name = "MaxPool2D_%02i" % TFNMaxPool2D.Index
    pass

  def __str__(self):
    return "MaxPool2D: %s, pool: (%i, %i), stride: (%i, %i), padding: %s" % \
           (self._name, self._pool_h, self._pool_w, self._stride_h, self._stride_w, self._padding)

  def hash_without_name(self):
    if self._pseudo_hash is not None:
      return self._pseudo_hash
    result = 0
    for elem in [self._pool_w,
                 self._pool_h,
                 self._stride_w,
                 self._stride_h,
                 int.from_bytes(self._padding.encode('utf-8'), byteorder='big'),
                 int.from_bytes(self._data_format.encode('utf-8'), byteorder='big')]:
      result = result * 31 + hash(elem)
    self._pseudo_hash = result
    return result

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    result = super(TFNMaxPool2D, self).__hash__()
    for elem in [self._pool_w,
                 self._pool_h,
                 self._stride_w,
                 self._stride_h,
                 int.from_bytes(self._padding.encode('utf-8'), byteorder='big'),
                 int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
                 int.from_bytes(self._name.encode('utf-8'), byteorder='big')]:
      result = result * 31 + hash(elem)
    self._hash = result
    return result

  def similar(self, other):
    if not super(TFNMaxPool2D, self).similar(other) or \
            self._pool_w != other._pool_w or \
            self._pool_h != other._pool_h or \
            self._stride_w != other._stride_w or \
            self._stride_h != other._stride_h or \
            self._padding != other._padding or \
            self._data_format != other._data_format:
      return False
    return True

  def __eq__(self, other):
    if not super(TFNMaxPool2D, self).__eq__(other) or \
            self._pool_w != other._pool_w or \
            self._pool_h != other._pool_h or \
            self._stride_w != other._stride_w or \
            self._stride_h != other._stride_h or \
            self._padding != other._padding or \
            self._data_format != other._data_format or \
            self._name != other._name:
      return False
    return True

  def __getstate__(self):
    result = super(TFNMaxPool2D, self).__getstate__()
    result['pool_width'] = self._pool_w
    result['pool_height'] = self._pool_h
    result['stride_width'] = self._stride_w
    result['stride_height'] = self._stride_h
    result['padding'] = self._padding
    result['data_format'] = self._data_format
    result['name'] = self._name
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __copy__(self):
    state = self.__getstate__()
    state['name'] = None
    result = TFNMaxPool2D(**state)
    return result

  @property
  def data_format(self):
    return self._data_format

  def outputsize(self, inputsizes: list):
    if len(inputsizes) == 0:
      raise TreeException('Error input list is empty!')
    b, h, w, _ = TFNodeUtil.get_bhwc(inputsizes[0], self._data_format)
    bhw = (b, h, w)
    c = 0
    for input_s in inputsizes:
      _b, _h, _w, _c = TFNodeUtil.get_bhwc(input_s, self._data_format)
      if (_b, _h, _w) != bhw:
        raise TreeException('All input nodes must have same batchsize, height and width!')
      if (_h is not None and _h <= 0) or \
              (_w is not None and _w <= 0) or \
              (_c is not None and _c <= 0):
        raise TreeException('Every input dimension must be greater than 0!\n'
                             'Found h:%s, w:%s, c:%s' % (_h, _w, _c))
      c += _c

    out_w = math.ceil((w if self._padding == TFNodeUtil.SAME else w - self._pool_w + 1) / self._stride_w)
    out_h = math.ceil((h if self._padding == TFNodeUtil.SAME else h - self._pool_h + 1) / self._stride_h)
    if self._data_format == TFNodeUtil.CHANNELS_LAST:
      return [b, out_h, out_w, c]
    else:
      return [b, c, out_h, out_w]

  def min_inputsize(self, outputsize):
    b, h, w, c = TFNodeUtil.get_bhwc(outputsize, self._data_format)
    if self._padding == TFNodeUtil.SAME:
      _h = (h - 1) * self._stride_h + 1
      _w = (w - 1) * self._stride_w + 1
    else:
      _h = (h - 1) * self._stride_h + self._pool_h
      _w = (w - 1) * self._stride_w + self._pool_w

    return (b, _h, _w, 1) if self._data_format == TFNodeUtil.CHANNELS_LAST else (b, 1, _h, _w)

  def max_inputsize(self, outputsize):
    b, h, w, c = TFNodeUtil.get_bhwc(outputsize, self._data_format)
    if self._padding == TFNodeUtil.SAME:
      _h = h * self._stride_h
      _w = w * self._stride_w
    else:
      _h = h * self._stride_h + self._pool_h - 1
      _w = w * self._stride_w + self._pool_w - 1
    return (b, _h, _w, 1) if self._data_format == TFNodeUtil.CHANNELS_LAST else (b, 1, _h, _w)

  def create_nn(self, inputs: list, **kwargs):
    with tf.variable_scope(name_or_scope=self._name, reuse=tf.AUTO_REUSE):
      if len(inputs) > 1:
        _input = tf.concat(values=inputs, axis=3 if self._data_format == TFNodeUtil.CHANNELS_LAST else 1)
      else:
        if len(inputs) == 0:
          raise TreeException('Error input list is empty!')
        _input = inputs[0]
      result = tf.layers.max_pooling2d(inputs=_input,
                                       pool_size=(self._pool_h, self._pool_w),
                                       strides=(self._stride_h, self._stride_w),
                                       padding=self._padding,
                                       data_format=self._data_format)
    return result

  @staticmethod
  def parameter_range(inputsize, outputsize, padding=None, data_format=TFNodeUtil.CHANNELS_LAST):
    def get_bhwc(size_):
      if data_format == TFNodeUtil.CHANNELS_LAST:
        return tuple(size_)
      else:
        return (size_[0], size_[2], size_[3], size_[1])

    b_i, h_i, w_i, c_i = get_bhwc(inputsize)
    b_o, h_o, w_o, c_o = get_bhwc(outputsize)

    def stride(in_, out_):
      if out_ == 1:
        return [in_]
      else:
        lower_limit = math.ceil(in_ / out_)
        upper_limit = math.ceil(in_ / (out_ - 1)) - 1
        if lower_limit == 0 or \
                math.ceil(in_ / lower_limit) < out_ or \
                upper_limit == 0 or \
                math.ceil(in_ / upper_limit) > out_:
          return None
        return list(range(lower_limit, upper_limit + 1))

    def filter(in_, out_):
      lower_limit = 1
      upper_limit = in_ - out_ + 1
      return list(range(lower_limit, upper_limit + 1))

    def with_padding(padding):
      if padding == TFNodeUtil.SAME:
        h_range = stride(h_i, h_o)
        w_range = stride(w_i, w_o)
        if h_range is None or w_range is None:
          return []

        return [{'padding': padding,
                 'stride_height': h,
                 'stride_width': w} for h in h_range for w in w_range]
      elif padding == TFNodeUtil.VALID:
        result = []
        for f_w in filter(w_i, w_o):
          for f_h in filter(h_i, h_o):
            h_range = stride(h_i - f_h + 1, h_o)
            w_range = stride(w_i - f_w + 1, w_o)
            if h_range is None or w_range is None:
              continue
            result += [{'padding': padding,
                        'pool_height': f_h,
                        'pool_width': f_w,
                        'stride_height': h,
                        'stride_width': w} for h in h_range for w in w_range]
        return result

    if padding == None:
      result = []
      for p in [TFNodeUtil.VALID, TFNodeUtil.SAME]:
        result += with_padding(p)
    else:
      result = with_padding(padding)
    return result

  pass


class TFInputNode(AbstractInputNode):
  Index = 0

  def __init__(self, **kwargs):
    super(TFInputNode, self).__init__(**kwargs)
    self._dtype = kwargs.get('dtype')
    self._shape = kwargs.get('shape')
    self._name = kwargs.get('name')
    self._data_format = kwargs.get('data_format')

    if self._data_format is None:
      self._data_format = TFNodeUtil.CHANNELS_LAST

    if not self._name:
      TFInputNode.Index += 1
      self._name = "Input_%02i" % TFInputNode.Index

    if self._dtype is None:
      raise TreeException('Data type must not be None!')
    if len(self._shape) < 4:
      raise TreeException('Shape length must be 4 but is %i' % len(self._shape))

  def __getstate__(self):
    result = super(TFInputNode, self).__getstate__()
    result['dtype'] = self._dtype
    result['shape'] = self._shape
    result['name'] = self._name
    result['data_format'] = self._data_format
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __str__(self):
    return "Input: %s, shape %s, dtype: %s" % \
           (self._name, self._shape.__str__(), self._dtype.__str__())

  def hash_without_name(self):
    if self._pseudo_hash is not None:
      return self._pseudo_hash
    result = 0
    for elem in [
      int.from_bytes(str(self._dtype.__class__).encode('utf-8'), byteorder='big') if self._dtype != None else 0,
      self._shape,
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
    ]:
      result = result * 31 + hash(elem)
    self._pseudo_hash = result
    return result

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    result = super(TFInputNode, self).__hash__()
    for elem in [
      int.from_bytes(str(self._dtype.__class__).encode('utf-8'), byteorder='big') if self._dtype != None else 0,
      self._shape,
      int.from_bytes(self._name.encode('utf-8'), byteorder='big'),
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
    ]:
      result = result * 31 + hash(elem)
    self._hash = result
    return result

  def similar(self, other):
    if not super(TFInputNode, self).similar(other) or \
            self._dtype != other._dtype or \
            self._shape != other._shape:
      return False
    return True

  def __eq__(self, other):
    if not super(TFInputNode, self).__eq__(other) or \
            self._dtype != other._dtype or \
            self._shape != other._shape or \
            self._name != other._name:
      return False
    return True

  def __copy__(self):
    state = self.__getstate__()
    # keep name
    # state['name'] = None
    result = TFInputNode(**state)
    return result

  @property
  def data_format(self):
    return self._data_format

  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, shape_):
    self._shape = shape_

  def outputsize(self, inputsizes: list):
    return self._shape

  def min_inputsize(self, outputsize):
    return outputsize

  def max_inputsize(self, outputsize):
    return outputsize

  def create_nn(self, inputs: list, **kwargs):
    if len(inputs) > 0:
      raise TreeException('Input nodes must not have an input itself!')
    try:
      result = tf.get_default_graph().get_tensor_by_name(self._name + ':0')
    except Exception as e:
      result = tf.placeholder(dtype=self._dtype, shape=self._shape, name=self._name)
    return result

  pass


class TFOutputNode(AbstractOutputNode):
  Index = 0

  def __init__(self, **kwargs):
    super(TFOutputNode, self).__init__(**kwargs)
    self._data_format = kwargs.get('data_format', TFNodeUtil.CHANNELS_LAST)
    self._name = kwargs.get('name')
    if not self._name:
      TFOutputNode.Index += 1
      self._name = "Output_%02i" % TFOutputNode.Index

  def __getstate__(self):
    result = super(TFOutputNode, self).__getstate__()
    result['data_format'] = self._data_format
    result['name'] = self._name
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __str__(self):
    return "Output: %s" % (self._name)

  def hash_without_name(self):
    if self._pseudo_hash is not None:
      return self._pseudo_hash
    result = 0
    for elem in [
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
      int.from_bytes(self._name.encode('utf-8'), byteorder='big'),
    ]:
      result = result * 31 + hash(elem)
    self._pseudo_hash = result
    return result

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    result = super(TFOutputNode, self).__hash__()
    for elem in [
      int.from_bytes(self._data_format.encode('utf-8'), byteorder='big'),
      int.from_bytes(self._name.encode('utf-8'), byteorder='big'),
    ]:
      result = result * 31 + hash(elem)
    self._hash = result
    return result

  def similar(self, other):
    if not super(TFOutputNode, self).similar(other) or \
            self._data_format != other._data_format:
      return False
    return True

  def __eq__(self, other):
    if not super(TFOutputNode, self).__eq__(other) or \
            self._data_format != other._data_format or \
            self._name != other._name:
      return False
    return True

  def __copy__(self):
    state = self.__getstate__()
    state['name'] = None
    result = TFOutputNode(**state)
    return result

  @property
  def data_format(self):
    return self._data_format

  def outputsize(self, inputsizes: list):
    if len(inputsizes) == 0:
      raise TreeException('Error input list is empty!')
    b, _, _, _ = TFNodeUtil.get_bhwc(inputsizes[0], self._data_format)
    w = 0
    for input_s in inputsizes:
      _b, _h, _w, _c = TFNodeUtil.get_bhwc(input_s, self._data_format)
      if _c is None or \
              _w is None or \
              _h is None:
        return [b, None]
      if (_h is not None and _h <= 0) or \
              (_w is not None and _w <= 0) or \
              (_c is not None and _c <= 0):
        raise TreeException('Every input dimension must be greater than 0!\n'
                             'Found h:%s, w:%s, c:%s' % (_h, _w, _c))
      w += _c * _w * _h
    return [b, w]

  def min_inputsize(self, outputsize):
    return [None, 1, 1, 1]

  def max_inputsize(self, outputsize):
    return [None, float('inf'), float('inf'), 1]

  def create_nn(self, inputs: list, **kwargs):
    with tf.variable_scope(name_or_scope=self._name, reuse=tf.AUTO_REUSE):
      if len(inputs) > 1:
        _input = [tf.layers.flatten(inputs=in_) for in_ in inputs]
        result = tf.concat(values=_input, axis=1)
      else:
        if len(inputs) == 0:
          raise TreeException('Error input list is empty!')
        result = tf.layers.flatten(inputs=inputs[0])
    return result

  pass
