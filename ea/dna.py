from enum import Enum
import numpy as np
import random


class DNAException(Exception):
  pass


class CrossoverType(Enum):
  N_POINT = 1
  N_POINT_WITH_IDENTITY = 2
  X_POINT = 3
  pass


class AbstractDNA():
  DICT_REPRESENTATION = 'representation'
  """Non functional DNA base class.

  Use this class as parent for your own dna representation.
  """

  def __init__(self, **kwargs):
    self._representation = kwargs.get(self.DICT_REPRESENTATION, None)
    self._crossover = None
    self._mutation = None
    self._distance = None
    pass

  def __hash__(self):
    return super(object, self).__hash__()

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    return self._representation == other._representation

  def __ne__(self, other):
    return not self.__eq__(other)

  def __getstate__(self):
    result = dict()
    result[self.DICT_REPRESENTATION] = self._representation
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __mod__(self, probability):
    """
    DNA Mutation
    :param probability: Mutation probability [0,1]
    :return: new DNA object
    """
    if self._mutation is None:
      raise DNAException('ERROR: You need to assign a mutation function to _mutation!')
    return self._mutation(probability)

  def __pow__(self, DNA):
    """
    DNA Crossover
    :param DNA: other DNA object
    :return: Complementary offsprings
    """
    if not isinstance(DNA, self.__class__):
      raise DNAException('ERROR: You can only crossbreed DNA of the same type!')
    if self._crossover is None:
      raise DNAException('ERROR: You need to assign a crossover function to _crossover!')
    return self._crossover(DNA)

  def __sub__(self, DNA):
    """
    DNA Distance
    :param DNA: other DNA object
    :return: Distance of DNA objects
    """
    if not isinstance(DNA, self.__class__):
      raise DNAException('ERROR: You can only calculate the distance of DNA objects with the same type!')
    if self._crossover is None:
      raise DNAException('ERROR: You need to assign a distance function to _distance!')
    return self._distance(DNA)

  @property
  def representation(self):
    return self._representation

  pass


class IndirectLinearDNA(AbstractDNA):
  DICT_CROSSOVER_TYPE = 'crossover_type'
  DICT_POINTS = 'points'

  def __init__(self, **kwargs):
    super(IndirectLinearDNA, self).__init__(**kwargs)
    self._crossoverType = kwargs.get(self.DICT_CROSSOVER_TYPE, CrossoverType.N_POINT)
    self._points = kwargs.get(self.DICT_POINTS, 2)

  def __getstate__(self):
    result = super(IndirectLinearDNA, self).__getstate__()
    result[self.DICT_CROSSOVER_TYPE] = self._crossoverType
    result[self.DICT_POINTS] = self._points
    return result

  """
  Crossover functions for list based representations
  """

  def _cross_x_point(self, other):
    __len = min(len(self._representation), len(other._representation))

    positions = range(1, __len)
    positions = random.sample(positions, min(self._points, len(positions)))
    positions.sort()

    tmp1, tmp2 = self._representation, other._representation
    descendant_a, descendant_b = [], []

    prev_p = 0
    for p in positions:
      descendant_a += tmp1[prev_p:p]
      descendant_b += tmp2[prev_p:p]
      prev_p = p
      tmp1, tmp2 = tmp2, tmp1

    descendant_a += tmp1[prev_p:len(tmp1)]
    descendant_b += tmp2[prev_p:len(tmp2)]

    if random.random() > 0.5:
      descendant_a, descendant_b = descendant_b, descendant_a

    return [self._new_DNA(descendant) for descendant in [descendant_a, descendant_b]]

  def _cross_n_point(self, other):
    __len = min(len(self._representation), len(other._representation))
    while True:
      crossing = np.random.randint(2, size=__len)
      __sum = np.sum(crossing)
      if 0 < __sum < __len:
        break

    descendant_a = [p1 if b else p2 for p1, p2, b in zip(self._representation, other._representation, crossing)]
    descendant_b = [p2 if b else p1 for p1, p2, b in zip(self._representation, other._representation, crossing)]

    tail_1, tail_2 = self._representation[__len:len(self._representation)], other._representation[
                                                                            __len:len(other._representation)]
    if not crossing[-1]:
      tail_1, tail_2 = tail_2, tail_1

    descendant_a += tail_1
    descendant_b += tail_2

    return [self._new_DNA(descendant) for descendant in [descendant_a, descendant_b]]

  def _cross_n_point_u_identity(self, other):
    __len = min(len(self._representation), len(other._representation))
    crossing = np.random.randint(2, size=__len)
    descendant_a = [p1 if b else p2 for p1, p2, b in zip(self._representation, other._representation, crossing)]
    descendant_b = [p2 if b else p1 for p1, p2, b in zip(self._representation, other._representation, crossing)]

    tail_1, tail_2 = self._representation[__len:len(self._representation)], other._representation[
                                                                            __len:len(other._representation)]
    if not crossing[-1]:
      tail_1, tail_2 = tail_2, tail_1

    descendant_a += tail_1
    descendant_b += tail_2

    return [self._new_DNA(descendant) for descendant in [descendant_a, descendant_b]]

  def _new_DNA(self, representation):
    state = self.__getstate__()
    state[self.DICT_REPRESENTATION] = representation
    new = self.__class__(**state)
    return new

  pass


class MNIST625DNA(IndirectLinearDNA):
  class DistanceType(Enum):
    SEARCH_SPACE = 1
    PARAMETER_SPACE = 2

  class MutationType(Enum):
    NEIGHBOR = 1
    RANDOM = 2
    EXCLUSIVE = 3

  __conv = [3, 5, 7, 9, 11]
  __pool = [1, 2, 3, 4, 5]
  _search_mat = [__conv, __pool, __conv, __pool]

  def __init__(self, **kwargs):
    super(MNIST625DNA, self).__init__(**kwargs)
    if self._representation is None:
      self._representation = self._random_representation()

    self.__representation_indexed = np.asarray(
      [__values.index(a) for a, __values in zip(self._representation, self._search_mat)])

    self._crossover = {CrossoverType.N_POINT: self._cross_n_point,
                       CrossoverType.N_POINT_WITH_IDENTITY: self._cross_n_point_u_identity,
                       CrossoverType.X_POINT: self._cross_x_point,
                       }.get(self._crossoverType)

    self._mutationType = kwargs.get('mutation_type', MNIST625DNA.MutationType.NEIGHBOR)
    self._mutation = {MNIST625DNA.MutationType.NEIGHBOR: self.__mut_neighbor,
                      MNIST625DNA.MutationType.RANDOM: self.__mut_random,
                      MNIST625DNA.MutationType.EXCLUSIVE: self.__mut_exclusive,
                      }.get(self._mutationType)

    self._distanceType = kwargs.get('distance_type', MNIST625DNA.DistanceType.PARAMETER_SPACE)
    self._distance = {MNIST625DNA.DistanceType.PARAMETER_SPACE: self._distance_euclidean,
                      MNIST625DNA.DistanceType.SEARCH_SPACE: self.__distance_search_space,
                      }.get(self._distanceType)

  def __getstate__(self):
    result = super(MNIST625DNA, self).__getstate__()
    result['mutation_type'] = self._mutationType
    result['distance_type'] = self._distanceType
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def __hash__(self):
    return hash(tuple(self._representation))

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    return self._representation == other._representation

  def __ne__(self, other):
    if not isinstance(other, self.__class__):
      return True
    return self._representation != other._representation

  def _random_representation(self):
    conv1 = np.random.choice(self.__conv, 1)[0]
    pool1 = np.random.choice(self.__pool, 1)[0]
    conv2 = np.random.choice(self.__conv, 1)[0]
    pool2 = np.random.choice(self.__pool, 1)[0]
    return [conv1, pool1, conv2, pool2]

  """
  Distance
  """

  def _distance_euclidean(self, other):
    if not isinstance(other, self.__class__):
      return float('inf')
    d = np.linalg.norm(np.asarray(self._representation) - np.asarray(other._representation))
    return d

  def __distance_search_space(self, other):
    if not isinstance(other, self.__class__):
      return float('inf')
    d = np.linalg.norm(self.__representation_indexed - other.__representation_indexed)
    return d

  """
  Mutation
  """

  def __mut_neighbor(self, prob):
    representation = [__v[(__v.index(a) + int(np.sign(p))) % len(__v)] if abs(p) < prob else a
                      for p, a, __v in zip(np.random.uniform(-1, 1, size=len(self._representation)),
                                           self._representation, self._search_mat)]
    return [self._new_DNA(representation)]

  def __mut_exclusive(self, prob):
    representation = [random.choice([cp for i, cp in enumerate(t) if i != t.index(a)])
                      if a_p < prob else a for a_p, a, t in
                      zip(np.random.uniform(0, 1, size=len(self._representation)), self._representation,
                          self._search_mat)]
    return [self._new_DNA(representation)]

  def __mut_random(self, prob):
    new_dna = [random.choice(__values) if p < prob else a for p, a, __values in
               zip(np.random.uniform(0, 1, size=len(self._representation)), self._representation, self._search_mat)]
    return [self._new_DNA(new_dna)]

  pass


class MNIST625DNADual(MNIST625DNA):
  def __init__(self, **kwargs):
    super(MNIST625DNA, self).__init__(**kwargs)
    self._crossover = self.__cross_dual
    self._mutation = self.__mut_dual_2
    self._distance = self._distance_euclidean

    self.__dna_one = kwargs.get('dna_one')
    self.__dna_two = kwargs.get('dna_two')

    if not self.DICT_REPRESENTATION in kwargs:
      if not self.__dna_one:
        self.__dna_one = self._random_representation()
      if not self.__dna_two:
        self.__dna_two = self._random_representation()
      self._representation = random.choice([self.__dna_one, self.__dna_two])
      self.__representation_indexed = np.asarray(
        [__values.index(a) for a, __values in zip(self._representation, self._search_mat)])
    else:
      if not self.__dna_one or not self.__dna_two:
        raise DNAException('This Should not happen!')

    pass

  def __getstate__(self):
    result = super(MNIST625DNA, self).__getstate__()
    result['dna_one'] = self.__dna_one
    result['dna_two'] = self.__dna_two
    return result

  def __mut_dual(self, prob):
    representation = [random.choice([__v[(__v.index(a) + int(np.sign(p))) % len(__v)],
                                     b]) if abs(p) < prob else a
                      for p, a, b, __v in zip(np.random.uniform(-1, 1, size=len(self._representation)),
                                              self._representation,
                                              self.__dna_one if self._representation !=
                                                                self.__dna_one else self.__dna_two,
                                              self._search_mat)]
    state = self.__getstate__()
    state['representation'] = representation
    state['dna_one'] = self.__dna_one if self._representation != self.__dna_one else representation
    state['dna_two'] = self.__dna_two if self._representation != self.__dna_two else representation
    return [MNIST625DNADual(**state)]

  def __mut_dual_2(self, prob):
    prob_list = np.random.uniform(0, 1, size=len(self._representation))
    is_one = self._representation == self.__dna_one
    representation = [b if abs(p) < prob else a
                      for p, a, b in zip(prob_list,
                                         self._representation,
                                         self.__dna_two if is_one else self.__dna_one, )]
    rep = [a if abs(p) < prob else b
           for p, a, b in zip(prob_list,
                              self._representation,
                              self.__dna_two if is_one else self.__dna_one, )]

    state = self.__getstate__()
    state['representation'] = representation
    state['dna_one'] = rep if is_one else representation
    state['dna_two'] = representation if is_one else rep
    return [MNIST625DNADual(**state)]

  def __cross_dual(self, other):
    self_a, self_b = random.choice([(self.__dna_one, self.__dna_two), (self.__dna_two, self.__dna_one)])
    other_a, other_b = random.choice([(other.__dna_one, other.__dna_two), (other.__dna_two, other.__dna_one)])
    state = self.__getstate__()
    state['representation'] = None
    state['dna_one'] = self_a
    state['dna_two'] = other_a
    ind_a = MNIST625DNADual(**state)
    state = self.__getstate__()
    state['representation'] = None
    state['dna_one'] = self_b
    state['dna_two'] = other_b
    ind_b = MNIST625DNADual(**state)
    return [ind_a, ind_b]

  pass
