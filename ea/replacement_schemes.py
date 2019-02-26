import numpy as np


class AbstractReplacementScheme():
  """
  Base Class For Replacement Schemes
  """

  def __init__(self, **kwargs):
    pass

  def __str__(self):
    try:
      return self.__doc__.splitlines()[1]
    except:
      return self.__doc__

  def new_generation(self, prev_gen, descendants):
    raise NotImplementedError()

  pass


class GenerationalReplacement(AbstractReplacementScheme):
  """
  Generational Replacement
  """

  def __init__(self, **kwargs):
    super(GenerationalReplacement, self).__init__(**kwargs)

  def __str__(self):
    return 'GR'

  def new_generation(self, prev_gen, descendants):
    return list(descendants)


class NElitism(AbstractReplacementScheme):
  """
  N Elitism
  """

  def __init__(self, **kwargs):
    super(NElitism, self).__init__()
    self.__n = kwargs.get('n', 1)

  def __str__(self):
    return "NE n=%i" % self.__n

  def new_generation(self, prev_gen, descendants):
    return descendants + sorted(prev_gen, key=lambda x: x.fitness, reverse=True)[:self.__n]


class NWeakElitism(AbstractReplacementScheme):
  """
  N Weak Elitism
  """

  def __init__(self, **kwargs):
    """
    :param n: Integers - Number of elite individuals
    :param p: Float - Mutation probability
    """
    super(NWeakElitism, self).__init__(**kwargs)
    self.__n = kwargs.get('n', 1)
    self.__p = kwargs.get('p', .1)

  def __str__(self):
    return 'NWE n=%i p=%01.2f' % (self.__n, self.__p)

  def new_generation(self, prev_gen, descendants):
    return descendants + [(ind % self.__p)[0] for ind in
                          sorted(prev_gen, key=lambda x: x.fitness, reverse=True)[:self.__n]]


class DeleteNLast(AbstractReplacementScheme):
  """
  Delete N Last
  """

  def __init__(self, **kwargs):
    super(DeleteNLast, self).__init__(**kwargs)
    self.__n = kwargs.get('n', 4)

  def __str__(self):
    return 'DNL n=%i' % self.__n

  def new_generation(self, prev_gen, descendants):
    return sorted(prev_gen, key=lambda x: x.fitness, reverse=True)[:-self.__n] + \
           np.random.choice(descendants, self.__n, replace=False).tolist()


class DeleteN(AbstractReplacementScheme):
  """
  Delete N
  """

  def __init__(self, **kwargs):
    super(DeleteN, self).__init__(**kwargs)
    self.__n = kwargs.get('n', 4)

  def __str__(self):
    return 'DN n=%i' % self.__n

  def new_generation(self, prev_gen, descendants):
    blacklist = np.random.choice(prev_gen, self.__n, replace=False)
    return [ind for ind in prev_gen if ind not in blacklist] + \
           np.random.choice(descendants, self.__n, replace=False).tolist()


class TournamentReplacement(AbstractReplacementScheme):
  """
  Tournament Replacement
  """

  def __init__(self, **kwargs):
    super(TournamentReplacement, self).__init__(**kwargs)
    raise Exception('Not Implemented!!')
    pass

  def new_generation(self, prev_gen, descendants):
    pass

  pass
