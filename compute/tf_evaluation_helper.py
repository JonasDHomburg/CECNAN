from compute.evaluation_helper import AbstractEvaluationHelper
import tensorflow as tf

class LocalEH(AbstractEvaluationHelper):
  def __init__(self, **kwargs):
    super(LocalEH, self).__init__(**kwargs)
    self._fitness_lookup = dict()

    self._tf_sess_conf = kwargs.get('tf_config')
    if self._tf_sess_conf is None:
      self._tf_sess_conf = tf.ConfigProto()
      self._tf_sess_conf.gpu_options.allow_growth = True

    self._ga_train = kwargs.get('train_data')
    self._ga_valid = kwargs.get('valid_data')
    self._ga_test = kwargs.get('test_data')
    self._batch_size = kwargs.get('batch_size', 32)

    self._log_folder = kwargs.get('log_path')

  def __getstate__(self):
    result = super(LocalEH, self).__getstate__()
    result['fitness_lookup'] = self._fitness_lookup
    result['tf_config'] = self._tf_sess_conf
    result['train_data'] = self._ga_train
    result['valid_data'] = self._ga_valid
    result['test_data'] = self._ga_test
    result['batch_size'] = self._batch_size
    result['log_path'] = self._log_folder

  def __setstate__(self, state):
    self.__init__(**state)
    self._fitness_lookup = state.get('fitness_lookup', dict())

  def evaluate_pool(self, pool):
    for ind in pool:
      val = self._fitness_lookup.get(ind.dna)
      if val is not None:
        ind.f, ind.meta_data = val
      else:
        eval_options = dict()
        eval_options['session_config'] = self._tf_sess_conf
        eval_options['X_train'] = self._ga_train['X']
        eval_options['Y_train'] = self._ga_train['Y']
        eval_options['X_valid'] = self._ga_valid['X']
        eval_options['Y_valid'] = self._ga_valid['Y']
        eval_options['X_test'] = self._ga_test['X']
        eval_options['Y_test'] = self._ga_test['Y']
        eval_options['batch_size'] = self._batch_size
        eval_options['log_path'] = self._log_folder
        ind.meta_data = ind.evaluate(**eval_options)
        self._fitness_lookup[ind.dna] = ind.f, ind.meta_data
  pass