from abc import abstractmethod
from ea.dna import MNIST625DNA, MNIST625DNADual


class NEAIndividualException(Exception):
  pass


# <editor-fold desc="base GAIndividual">
class NEAIndividual(object):
  class_counter = 0

  DICT_DNA = 'dna'
  DICT_FITNESS = 'fitness'
  DICT_ID = 'id'
  DICT_META = 'meta_data'

  MUTATE = 'mutate'
  CROSSOVER = 'crossover'
  DISTANCE = 'distance'

  def __init__(self, **kwargs):
    super(NEAIndividual, self).__init__()
    self._fitness = kwargs.get(self.DICT_FITNESS, None)
    self._dna = kwargs.get(self.DICT_DNA, None)
    self._id = kwargs.get(self.DICT_ID)
    self._meta_data = kwargs.get(self.DICT_META, None)

  def __mod__(self, other):
    if self._dna is None:
      NEAIndividualException('ERROR: DNA is None!')
    pass

  def __pow__(self, other):
    if not isinstance(other, self.__class__):
      print(self.__class__, other.__class__)
      raise NEAIndividualException('ERROR: You can only crossbreed individuals of the same type!')
    if self._dna is None:
      NEAIndividualException('ERROR: self DNA is None!')
    if other._dna is None:
      NEAIndividualException('ERROR: other DNA is None!')
    pass

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      raise NEAIndividualException('ERROR: You can only calculate the distance of individuals with the same type!')
    if self._dna is None:
      NEAIndividualException('ERROR: self DNA is None!')
    if other._dna is None:
      NEAIndividualException('ERROR: other DNA is None!')
    return self._dna - other._dna

  def __getstate__(self):
    result = dict()
    result[self.DICT_FITNESS] = self._fitness
    result[self.DICT_DNA] = self._dna
    result[self.DICT_ID] = self._id
    result[self.DICT_META] = self._meta_data
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __eq__(self, other):
    if not isinstance(other, self.__class__) or \
            not isinstance(self, other.__class__) or \
            not self._dna == other._dna or \
            not self._id == other._id or \
            not self._meta_data == other._meta_data:
      return False
    return True

  def __hash__(self):
    result = hash(self._id)
    result = result * 31 + hash(self._dna)
    return result

  def identical_twin(self, other):
    if not isinstance(other, self.__class__):
      return False
    return self._dna == other._dna

  @property
  def fitness(self):
    """Individuals fitness.
    Calculates the fitness if it isn't done yet.
    :return: fitness
    :rtype: float
    """
    if self._fitness is None:
      self.evaluate()
    return self._fitness

  @property
  def f(self):
    """Individuals fitness.
    Doesn't calculate the fitness even if it hasn't been done yet. If so it returns -.1
    :return: fitness
    :rtype: float
    """
    return self._fitness if self._fitness is not None else -.1

  @f.setter
  def f(self, f):
    self._fitness = f

  @property
  def id(self):
    return self._id

  @id.setter
  def id(self, id):
    self._id = id

  @property
  def dna(self):
    return self._dna

  @abstractmethod
  def evaluate(self, **kwargs):
    pass

  @abstractmethod
  def test(self, **kwargs):
    pass

  @property
  def meta_data(self):
    return self._meta_data

  @meta_data.setter
  def meta_data(self, data):
    self._meta_data = data

  pass


# </editor-fold>


# <editor-fold desc="GAIndividual for the limited, 625 individual large search space">
class MNIST625Individual(NEAIndividual):
  def __init__(self, **kwargs):
    super(MNIST625Individual, self).__init__(**kwargs)
    self._evaluation_options = kwargs.get('evaluation_options', dict())
    if self._dna is None:
      self._dna = MNIST625DNA()

  def evaluate(self, **kwargs):
    self._fitness = self._evaluation_options.get(self._dna, 0)
    pass

  def evaluation_options(self, ev_opt):
    self._evaluation_options = ev_opt

  evaluation_options = property(None, evaluation_options)

  def __pow__(self, other):
    super(MNIST625Individual, self).__pow__(other)
    state = self.__getstate__()
    result = list()
    for dna in self._dna ** other._dna:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNIST625Individual(**state))
    return result

  def __mod__(self, probability):
    super(MNIST625Individual, self).__mod__(probability)
    state = self.__getstate__()
    result = list()
    for dna in self._dna % probability:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNIST625Individual(**state))
    return result

  def __getstate__(self):
    result = super(MNIST625Individual, self).__getstate__()
    result['evaluation_options'] = self._evaluation_options
    return result


# </editor-fold>

class MNIST625IndividualDual(MNIST625Individual):
  def __init__(self, **kwargs):
    super(MNIST625IndividualDual, self).__init__(**kwargs)
    self._evaluation_options = kwargs.get('evaluation_options', dict())
    if self._dna is None:
      self._dna = MNIST625DNADual()

  def __pow__(self, other):
    super(MNIST625Individual, self).__pow__(other)
    state = self.__getstate__()
    result = list()
    for dna in self._dna ** other._dna:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNIST625IndividualDual(**state))
    return result

  def __mod__(self, probability):
    super(MNIST625Individual, self).__mod__(probability)
    state = self.__getstate__()
    result = list()
    for dna in self._dna % probability:
      state[self.DICT_DNA] = dna
      state[self.DICT_FITNESS] = None
      state[self.DICT_ID] = None
      state[self.DICT_META] = None
      result.append(MNIST625IndividualDual(**state))
    return result

  pass
