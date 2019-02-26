class AbstractNode():
  def __init__(self, **kwargs):
    self._name = None
    self._hash = None
    self._pseudo_hash = None
    pass

  def __eq__(self, other):
    return (other is not None) and \
           (isinstance(self, other.__class__)) and \
           (isinstance(other, self.__class__))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __getstate__(self):
    return dict()

  def __hash__(self):
    return 0

  def outputsize(self, inputsizes: list):
    raise NotImplementedError('outputsize not implemented!')

  def min_inputsize(self, outputsize):
    raise NotImplementedError('min_inputsize not implemented!')

  def max_inputsize(self, outputsize):
    raise NotImplementedError('max_inputsize not implemented!')

  def create_nn(self, inputs: list, **kwargs):
    raise NotImplementedError('createGraph not implemented!')

  def hash_without_name(self):
    raise NotImplementedError('hash_without_name not implemented!')

  def similar(self, other):
    return (other is not None) and \
           (isinstance(self, other.__class__)) and \
           (isinstance(other, self.__class__))

  @staticmethod
  def parameter_range(inputsize, outputsize):
    raise NotImplementedError('parameter_range not implemented!')

  @property
  def name(self):
    return self._name

  pass


class AbstractInputNode(AbstractNode):
  def __init__(self, **kwargs):
    super(AbstractInputNode, self).__init__(**kwargs)

  pass


class AbstractOutputNode(AbstractNode):
  def __init__(self, **kwargs):
    super(AbstractOutputNode, self).__init__(**kwargs)

  pass