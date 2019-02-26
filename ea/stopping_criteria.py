from ea.GenerationManager import GenerationManager


class StoppingCriteriaException(Exception):
  pass


class AbstractStoppingCriteria(object):
  def __init__(self, **kwargs):
    self._generationManager = kwargs.get('generationManager')
    if not (self._generationManager, GenerationManager):
      raise StoppingCriteriaException('Generation manager is %s, but has to be of class %s' % (
        type(self._generationManager), GenerationManager.__class__))
    pass

  def converged(self):
    return True

  pass


class MNIST625OptimumInSelection(AbstractStoppingCriteria):
  def __init__(self, **kwargs):
    super(MNIST625OptimumInSelection, self).__init__(**kwargs)
    self.__optimal_dna = max(kwargs.get('fitness_dictionary').items(), key=lambda x: x[1])[0]
    pass

  def converged(self):
    return self.__optimal_dna in [ind.dna for ind in self._generationManager.prev_selection]

  pass


class SelectionStopping(AbstractStoppingCriteria):
  def __init__(self, **kwargs):
    super(SelectionStopping, self).__init__(**kwargs)
    self.__patience = kwargs.get('patience', 20)
    self.__patience_lvl = 0
    self.__last_optimum = -1

  def converged(self):
    curr_max = max(self._generationManager.prev_selection, key=lambda x: x.f)
    if curr_max.f > self.__last_optimum:
      self.__last_optimum = curr_max.f
      self.__patience_lvl = 0
    else:
      self.__patience_lvl += 1
    return self.__patience_lvl > self.__patience
    pass

  pass
