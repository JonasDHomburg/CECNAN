from ea.selection_strategies import RankingSelectionV1
from ea.stopping_criteria import AbstractStoppingCriteria
from ea.GenerationManager import GenerationManager
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count
import random

class NEACoreException(Exception):
  pass


class AbstractNEACore(object):
  def __init__(self, **kwargs):
    super(AbstractNEACore, self).__init__()
    self._genManager = kwargs.get('generationManager')
    if self._genManager is None:
      self._genManager = GenerationManager()
    if not isinstance(self._genManager, GenerationManager):
      raise NEACoreException('Generation manager is %s but has to be of class: %s' % (
        type(self._genManager), GenerationManager.__class__))

    self._saturationStrategy = kwargs.get('saturationStrategy', AbstractStoppingCriteria())
    if not isinstance(self._saturationStrategy, AbstractStoppingCriteria):
      raise NEACoreException('Saturation strategy is %s but has to of class: %s' % (
        type(self._saturationStrategy), AbstractStoppingCriteria.__class__))

    self._parallel = kwargs.get('threads', 0)
    if self._parallel < 0:
      self._parallel = cpu_count()
    if self._parallel > 0:
      self._cpu_pool = Pool(self._parallel)

  @property
  def generation(self):
    return self._genManager.generation_idx

  @property
  def genManager(self):
    return self._genManager

  def initialize(self):
    raise NotImplementedError()

  def mutate(self):
    raise NotImplementedError()

  def crossover(self):
    raise NotImplementedError()

  def select(self):
    raise NotImplementedError()

  def pass_generation(self):
    raise NotImplementedError()

  def converged(self):
    raise NotImplementedError()

  def run(self):
    raise NotImplementedError()

  pass


class NEACore(AbstractNEACore):
  def __init__(self, **args):
    super(NEACore, self).__init__(**args)

    """
    config
    """
    # selection
    self.__selectionStrategy = args.get('selection', RankingSelectionV1())

    # crossover
    self.__crossover_size = args.get('crossover_size', 1000000)
    self.__crossover_pair_size = args.get('crossover_pair_size', 2)

    # mutation
    self.__mutation_size = args.get('mutation_size', 1000000)
    self.__mutation_per_ind = args.get('mutations_per_individual', 1)
    self.__mutation_rate = args.get('mutation_rate', 0.6)

    # generations
    self.__generation_limit = args.get('generation_limit', 2000)

    # notifications
    self.__t_notifications = args.get('telegram_notifications', False)
    self.__t_notification_level = args.get('t_notification_level', 0)
    self.__t_bot_token = args.get('telegram_token')
    if self.__t_notifications:
      if self.__t_bot_token is None:
        raise NEACoreException('Telegram token not defined!')
      try:
        from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
        from telegram import Bot
        self.__t_updater = Updater(token=self.__t_bot_token)
        self.__t_chat_id_list = args.get('telegram_ids', [])
        self.__bot = Bot(self.__t_bot_token)

        def register(self, bot, update, args):
          print('Tried with: %s | right: %s' % (' '.join(args), self.__hash__()))
          if ' '.join(args) == str(self.__hash__()):
            print('added: ', update.message.chat_id)
            self.__t_chat_id_list.append(update.message.chat_id)
            bot.send_message(chat_id=update.message.chat_id, text='Registered!')
          else:
            bot.send_message(chat_id=update.message.chat_id, text='Not running on that ID!')

        self.__t_updater.dispatcher.add_handler(
          CommandHandler('register', lambda bot, update, args: register(self, bot, update, args),
                         pass_args=True))
        self.__t_updater.start_polling()
        if self.__t_notification_level < 1:
          self.__t_notification_level = 1
        print('To get notifications register with: %s' % self.__hash__())
      except ImportError:
        self.__t_notifications = False
        self.__t_notification_level = 0
        print('Failed to load telegram! Notifications are now disabled!')

  # <editor-fold desc="Selection">
  def select(self):
    self._genManager.selection = self.__selectionStrategy.select(self._genManager.generation)
    pass

  # </editor-fold>

  # <editor-fold desc="Crossover">
  def crossover(self):
    if self.__crossover_size < 1:
      pass
    parents = self._genManager.cro_gen
    mating_pairs = [(parents[p1], parents[p2])
                    for p1 in range(len(parents)) for p2 in range(p1 + 1, len(parents))]
    if self._parallel < 1:
      tmp_list = []
      for p1, p2 in mating_pairs:
        descendants = p1 ** p2
        self._genManager.__pow__(([p1, p2], descendants))
        tmp_list.extend(random.sample(descendants, self.__crossover_pair_size))
    else:
      cro_size = self.__crossover_pair_size
      genMa = self._genManager

      def cro(p):
        descendants = p[0] ** p[1]
        genMa.__pow__(([p[0], p[1]], descendants))
        return random.sample(descendants, cro_size)

      tmp_list = self._cpu_pool.map(cro, mating_pairs)
      tmp_list = [item for sublist in tmp_list for item in sublist]
    self._genManager ** random.sample(tmp_list, min(len(tmp_list), self.__crossover_size))
    pass

  # </editor-fold>

  # <editor-fold desc="Mutation">
  def mutate(self):
    if self.__mutation_size < 1:
      pass
    prob = self.__mutation_rate() if callable(self.__mutation_rate) else self.__mutation_rate
    if self._parallel < 1:
      tmp_list = []
      for p in self._genManager.mut_gen:
        descendants = p % prob
        self._genManager.__mod__((p, descendants))
        tmp_list.extend(random.sample(descendants, min(len(descendants), self.__mutation_per_ind)))
    else:
      genMa = self._genManager
      mut_size = self.__mutation_per_ind

      def mut(source):
        descendants = source % prob
        genMa.__mod__((source, descendants))
        return random.sample(descendants, min(len(descendants), mut_size))

      tmp_list = self._cpu_pool.map(mut, self.genManager.mut_gen)
      tmp_list = [item for sublist in tmp_list for item in sublist]
    self._genManager % random.sample(tmp_list, min(len(tmp_list), self.__mutation_size))
    pass

  # </editor-fold>

  def pass_generation(self):
    self.crossover()
    self.mutate()
    self._genManager.__invert__()
    self.select()
    if self.__t_notifications and ((self.__t_notification_level > 1 and self._genManager.generation_idx % 10 == 0) or
            self.__t_notification_level > 2):
      max_f, min_f = max(self._genManager.selection, key=lambda x: x.f).f, \
                     min(self._genManager.selection, key=lambda x: x.f).f
      for id in self.__t_chat_id_list:
        try:
          self.__bot.send_message(chat_id=id, text='Generation %i with:\n'
                                                   '\t maximum fitness: %.5f\n'
                                                   '\t minimum fitness: %.5f\n'
                                                   % (self._genManager.generation_idx,
                                                      max_f,
                                                      min_f))
        except Exception:
          print('Failed to send telegram message!')
    self._genManager.__pos__()
    pass

  def converged(self):
    return self._saturationStrategy.converged()

  def __print_pool(self, pool):
    for p in pool:
      print(p.dna, p.fitness)

  def run(self):
    self.select()
    if self.__t_notifications and ((self.__t_notification_level > 1 and self._genManager.generation_idx % 10 == 0) or
            self.__t_notification_level > 2):
      max_f, min_f = max(self._genManager.selection, key=lambda x: x.f).f, \
                     min(self._genManager.selection, key=lambda x: x.f).f
      for id in self.__t_chat_id_list:
        try:
          self.__bot.send_message(chat_id=id, text='Generation %i with:\n'
                                                   '\t maximum fitness: %.5f\n'
                                                   '\t minimum fitness: %.5f\n'
                                                   % (self._genManager.generation_idx,
                                                      max_f,
                                                      min_f))
        except Exception:
          print('Failed to send telegram message!')
    self._genManager.__pos__()
    while not self.converged() and self._genManager.generation_idx < self.__generation_limit:
      self.pass_generation()

    if self.__t_notification_level > 0 and self.__t_notifications:
      max_f, min_f = max(self._genManager.prev_selection, key=lambda x: x.f).f, \
                     min(self._genManager.prev_selection, key=lambda x: x.f).f
      for id in self.__t_chat_id_list:
        try:
          self.__bot.send_message(chat_id=id, text="GA finished after %i generations!\n"
                                                   "\tThe final max fitness is: %.5f.\n"
                                                   "\tThe final min fitness is: %.5f\n"
                                                   % (self._genManager.generation_idx - 1,
                                                      max_f,
                                                      min_f))
        except Exception:
          print('Failed to send telegram message!')
      self.__t_updater.stop()
    return self._genManager

  def stop(self):
    self._genManager.stop()
