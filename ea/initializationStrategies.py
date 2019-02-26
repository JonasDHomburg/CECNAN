class InitializationException(Exception):
  pass

class AbstractInitializationStrategy():
  def __init__(self, **kwargs):
    self._individuals_args = kwargs.get('individual_config', {})
    self._individual_class = kwargs.get('individual_class')
    self._generation_size = kwargs.get('size', 6)
    pass

  def new_generation(self):
    raise NotImplementedError()

  pass


class RandomGeneration(AbstractInitializationStrategy):
  def __init__(self, **kwargs):
    super(RandomGeneration, self).__init__(**kwargs)

  def new_generation(self):
    if self._individual_class is None:
      raise InitializationException('Individual class not defined!')
    return [self._individual_class(**self._individuals_args) for _ in range(self._generation_size)]
    pass

  pass


class MaximizeDiversity(AbstractInitializationStrategy):
  def __init__(self, **kwargs):
    super(MaximizeDiversity, self).__init__(**kwargs)
    self._poolsize = kwargs.get('pool_size', 3 * self._generation_size)

  def new_generation(self):
    if self._individual_class is None:
      raise InitializationException('Individual class not defined!')
    tmp_gen = [(self._individual_class(**self._individuals_args), 0)
               for _ in range(self._poolsize)]
    result = []
    for _ in range(self._generation_size):
      max_ind = max(tmp_gen, key=lambda x: x[1])
      ind = max_ind[0]
      result.append(ind)
      tmp_gen.remove(max_ind)
      tmp_gen = [(g[0], g[1] + (ind - g[0])) for g in tmp_gen]
    return result


class ConstantSeed(AbstractInitializationStrategy):
  def __init__(self, **kwargs):
    super(ConstantSeed, self).__init__(**kwargs)
    self._generation = kwargs.get('seed')
    if not self._generation:
      self._generation = self.__random_generation()

  def __random_generation(self):
    if self._individual_class is None:
      raise InitializationException('Individual class not defined!')
    return [self._individual_class(**self._individuals_args) for _ in range(self._generation_size)]

  def new_generation(self):
    return self._generation
