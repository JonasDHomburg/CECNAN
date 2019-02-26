class AbstractEvaluationHelper():
  def __init__(self, **kwargs):
    pass

  def evaluate_pool(self, pool):
    pass

  def __getstate__(self):
    return dict()

  pass


class MNIST625EH(AbstractEvaluationHelper):
  def __init__(self, **kwargs):
    super(MNIST625EH, self).__init__(**kwargs)
    self._fitness_lookup = kwargs.get('fitness_lookup', dict())

  def __getstate__(self):
    result = super(MNIST625EH, self).__getstate__()
    result['fitness_lookup'] = self._fitness_lookup
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  def evaluate_pool(self, pool):
    for ind in pool:
      if ind.f < 0:
        ind.f = self._fitness_lookup.get(ind.dna, -.1)
