from ea.initializationStrategies import RandomGeneration
from ea.replacement_schemes import NElitism
from ea.dna import AbstractDNA
from ea.individuals import NEAIndividual, NEAIndividualException
import os.path
import pickle
import copy


class GenerationException(Exception):
  pass


class DatabaseConstants():
  PARAMETER = ''
  IGNORE = ''


class DBSqlite3(DatabaseConstants):
  PARAMETER = '?'
  IGNORE = 'OR IGNORE'


class DBMySql(DatabaseConstants):
  PARAMETER = '%s'
  IGNORE = 'IGNORE'


class PlaybackGenManager():
  _DICT_DES_IND = 'descendants'
  _DICT_SEL_IND = 'selected_individuals'
  _DICT_CRO_IND = 'crossover_descendants'
  _DICT_MUT_IND = 'mutation_descendants'
  _DICT_CRO_REL = 'crossover_relationship'
  _DICT_MUT_REL = 'mutation_relationship'
  _DICT_GEN_IND = 'generation'

  MEM_PICKLE = 'pickle'
  MEM_PICKLER = 'pickler'
  MEM_SQLITE3 = 'sqlite3'
  MEM_MySQL = 'MySql'
  # stats does not save the current state at all it only saves some statistics
  MEM_STATS = 'stats'

  _TABLE_DNA = '_dna'
  _TABLE_INDIVIDUALS = '_individuals'
  _TABLE_GENERATION_IDXS = '_generations_idx'

  _TABLE_DESCENDANTS = '_descendants_ind'
  _TABLE_SELECTION = '_selected_ind'
  _TABLE_GENERATION = '_generation_ind'
  _TABLE_CROSSOVER_IND = '_crossover_ind'
  _TABLE_MUTTAION_IND = '_mutation_ind'

  _TABLE_CROSSOVER_REL = '_crossover_rel'
  _TABLE_MUTATION_REL = '_mutation_rel'

  _TABLE_CLASSES = '_object_class'
  _TABLE_META_DATA = '_meta_data'
  _TABLE_CLASS_PARAMETER = '_class_parameter'

  def __init__(self, **kwargs):
    self._memory_type = kwargs.get('mem_type')
    if self._memory_type is None:
      self._memory_type = self.MEM_SQLITE3

    self._playback = False
    if 'filename' in kwargs:
      self._generation = -1
      self._gen_history = dict()
      self._playback = True
      self._prev_gen = self._empty_gen_dict()
      self._curr_gen = self._empty_gen_dict()
      if self._memory_type is self.MEM_PICKLER:
        f = open(kwargs.get('filename'), 'rb')
        up = pickle.Unpickler(f)
        while True:
          try:
            key, val = up.load()
            self._gen_history[key] = val
          except EOFError:
            break
        del up
        f.close()
      elif self._memory_type is self.MEM_PICKLE:
        f = open(kwargs.get('filename'), 'rb')
        while True:
          try:
            key, val = pickle.load(f)
            self._gen_history[key] = val
          except EOFError:
            break
        f.close()
      elif self._memory_type is self.MEM_SQLITE3:
        import sqlite3
        self._conn = sqlite3.connect(kwargs.get('filename'))
        self._db_constants = DBSqlite3
        self.__load_from_db(**kwargs)
      elif self._memory_type is self.MEM_MySQL:
        import mysql.connector
        from mysql.connector import errorcode
        user = kwargs.get('user', 'root')
        password = kwargs.get('password', '123456')
        database = kwargs.get('filename')
        host = kwargs.get('host', '127.0.0.1')
        try:
          self._conn = mysql.connector.connect(user=user, password=password, host=host, database=database)
        except mysql.connector.Error as err:
          if err.errno == errorcode.ER_BAD_DB_ERROR:
            raise GenerationException('Database does not exist!')
          raise err
        self._db_constants = DBMySql
        self.__load_from_db(**kwargs)
      elif self._memory_type is self.MEM_STATS:
        raise GenerationException('Stats do NOT support a generation replay!')
      else:
        raise GenerationException('Unsupported memory type!')
      self._playback = True
      if len(self._gen_history) < 1:
        NEAIndividualException('History is empty!')

  def __load_from_db(self, **kwargs):
    cursor = self._conn.cursor()
    self._table_prefix = kwargs.get('table_prefix', '')
    cursor.execute(
      "SELECT id FROM {}{} ORDER BY id ASC LIMIT 1;".format(self._table_prefix, self._TABLE_GENERATION_IDXS))
    fetched = cursor.fetchone()
    if fetched:
      self._generation = fetched[0] - 1
    else:
      raise GenerationException('Empty generation history!')

    class_dict, dna_dict, individual_dict, meta_dict, parameter_dict = dict(), dict(), dict(), dict(), dict()
    cursor.execute("SELECT id FROM {}{} ORDER BY id ASC".format(self._table_prefix, self._TABLE_GENERATION_IDXS))
    for _gen_idx in cursor.fetchall():
      gen_idx = _gen_idx[0]
      self._gen_history[gen_idx] = self._load_generation_from_db(class_dict=class_dict,
                                                                 dna_dict=dna_dict,
                                                                 individual_dict=individual_dict,
                                                                 meta_dict=meta_dict,
                                                                 parameter_dict=parameter_dict,
                                                                 gen_idx=gen_idx)
    cursor.close()

  def __pos__(self):
    tmp = self._gen_history.get(self._generation + 1)
    if tmp is None:
      raise StopIteration
    self._prev_gen, self._curr_gen = self._curr_gen, tmp
    self._generation += 1

  def __rshift__(self, other):
    if not self._playback:
      raise GenerationException("Not in playback mode!")
    tmp = self._gen_history.get(self._generation + other)
    if tmp is None:
      raise GenerationException("You reached the end of the history!")
    self._generation += other
    self._prev_gen, self._curr_gen = self._gen_history.get(self._generation - 1, self._empty_gen_dict()), tmp

  def __lshift__(self, other):
    if not self._playback:
      raise GenerationException("Not in playback mode!")
    tmp = self._gen_history.get(self._generation - other)
    if tmp is None:
      raise GenerationException("You reached the end of the history!")
    self._generation -= other
    self._prev_gen, self._curr_gen = self._gen_history.get(self._generation - 1, self._empty_gen_dict()), tmp

  def __iter__(self):
    return self

  def __next__(self):
    if self._playback == False:
      raise StopIteration
    self.__pos__()
    return self

  def _empty_gen_dict(self):
    result = {
      self._DICT_DES_IND: [],
      self._DICT_SEL_IND: [],
      self._DICT_CRO_IND: [],
      self._DICT_MUT_IND: [],
      self._DICT_GEN_IND: [],

      self._DICT_CRO_REL: [],
      self._DICT_MUT_REL: [],
    }
    return result

  def _load_class_from_db(self, class_idx):
    cursor = self._conn.cursor()
    cursor.execute("SELECT class FROM {}{} WHERE rowid={}".format
                   (self._table_prefix, self._TABLE_CLASSES, self._db_constants.PARAMETER), [class_idx])
    fetched = cursor.fetchone()
    if not fetched:
      raise GenerationException("Class with id %i not found in database!" % class_idx)
    result = pickle.loads(fetched[0])
    self._conn.commit()
    cursor.close()
    return result

  def _load_meta_from_db(self, meta_idx):
    cursor = self._conn.cursor()
    cursor.execute("SELECT meta_data FROM {}{} WHERE rowid={}".format
                   (self._table_prefix, self._TABLE_META_DATA, self._db_constants.PARAMETER), [meta_idx])
    fetched = cursor.fetchone()
    if not fetched:
      raise GenerationException("Meta data with id %i not found in database!" % meta_idx)
    result = pickle.loads(fetched[0])
    self._conn.commit()
    cursor.close()
    return result

  def _load_parameters_from_db(self, parameter_idx):
    cursor = self._conn.cursor()
    cursor.execute("SELECT parameters FROM {}{} WHERE rowid={}".format
                   (self._table_prefix, self._TABLE_CLASS_PARAMETER, self._db_constants.PARAMETER), [parameter_idx])
    fetched = cursor.fetchone()
    if not fetched:
      raise GenerationException("Parameters with id %i not found in database!" % parameter_idx)
    result = pickle.loads(fetched[0])
    self._conn.commit()
    cursor.close()
    return result

  def _load_dna_from_db(self, class_dict, parameter_dict, dna_idx):
    cursor = self._conn.cursor()
    cursor.execute("SELECT class, dna, parameters FROM {}{} WHERE rowid={}".format
                   (self._table_prefix, self._TABLE_DNA, self._db_constants.PARAMETER), [dna_idx])
    fetched = cursor.fetchone()
    if not fetched:
      raise GenerationException('DNA with id %i not found in database!' % dna_idx)
    _class = class_dict.get(fetched[0])
    if not _class:
      _class = self._load_class_from_db(class_idx=fetched[0])
      class_dict[fetched[0]] = _class
    _parameters = parameter_dict.get(fetched[2])
    if not _parameters:
      _parameters = self._load_parameters_from_db(parameter_idx=fetched[2])
      parameter_dict[fetched[2]] = _parameters
    _parameters.update({AbstractDNA.DICT_REPRESENTATION: pickle.loads(fetched[1])})
    result = _class(**_parameters)
    self._conn.commit()
    cursor.close()
    return result

  def _load_individual_from_db(self, class_dict, dna_dict, meta_dict, parameter_dict, individual_idx):
    cursor = self._conn.cursor()
    cursor.execute("SELECT class, dna, fitness, meta_data, parameters FROM {}{} WHERE rowid={}".format
                   (self._table_prefix, self._TABLE_INDIVIDUALS, self._db_constants.PARAMETER), [individual_idx])
    fetched = cursor.fetchone()
    if not fetched:
      raise GenerationException('Individual with id %i not found in database!' % individual_idx)
    _class = class_dict.get(fetched[0])
    if not _class:
      _class = self._load_class_from_db(class_idx=fetched[0])
      class_dict[fetched[0]] = _class
    _dna = dna_dict.get(fetched[1])
    if not _dna:
      _dna = self._load_dna_from_db(class_dict=class_dict, parameter_dict=parameter_dict, dna_idx=fetched[1])
      dna_dict[fetched[1]] = _dna
    _meta_data = meta_dict.get(fetched[3])
    if not _meta_data:
      _meta_data = self._load_meta_from_db(meta_idx=fetched[3])
      meta_dict[fetched[3]] = _meta_data
    _parameters = parameter_dict.get(fetched[4])
    if not _parameters:
      _parameters = self._load_parameters_from_db(parameter_idx=fetched[4])
      parameter_dict[fetched[4]] = _parameters
    _parameters.update({NEAIndividual.DICT_DNA: _dna, NEAIndividual.DICT_FITNESS: fetched[2],
                        NEAIndividual.DICT_ID: individual_idx, NEAIndividual.DICT_META: _meta_data})
    result = _class(**_parameters)
    self._conn.commit()
    cursor.close()
    return result

  def _load_generation_from_db(self, class_dict, dna_dict, individual_dict, meta_dict, parameter_dict, gen_idx):
    result = self._empty_gen_dict()
    cursor = self._conn.cursor()
    for key, table in [(self._DICT_DES_IND, self._TABLE_DESCENDANTS),
                       (self._DICT_SEL_IND, self._TABLE_SELECTION),
                       (self._DICT_GEN_IND, self._TABLE_GENERATION),
                       (self._DICT_CRO_IND, self._TABLE_CROSSOVER_IND),
                       (self._DICT_MUT_IND, self._TABLE_MUTTAION_IND)]:
      cursor.execute("SELECT individual FROM {}{} WHERE generation={}".format
                     (self._table_prefix, table, self._db_constants.PARAMETER), [gen_idx])
      for ind_id in cursor.fetchall():
        new_ind = individual_dict.get(ind_id[0])
        if not new_ind:
          new_ind = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                                  meta_dict=meta_dict, individual_idx=ind_id[0],
                                                  parameter_dict=parameter_dict)
          individual_dict[ind_id] = new_ind
        result[key].append(new_ind)

    cursor.execute("SELECT parent1, parent2, child FROM {}{} WHERE generation={}".format
                   (self._table_prefix, self._TABLE_CROSSOVER_REL, self._db_constants.PARAMETER),
                   [gen_idx])
    crossover_dict = dict()
    while True:
      table_row = cursor.fetchone()
      if not table_row:
        break
      p1 = individual_dict.get(table_row[0])
      if not p1:
        p1 = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                           meta_dict=meta_dict, individual_idx=table_row[0],
                                           parameter_dict=parameter_dict)
        individual_dict[table_row[0]] = p1
      p2 = individual_dict.get(table_row[1])
      if not p2:
        p2 = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                           meta_dict=meta_dict, individual_idx=table_row[1],
                                           parameter_dict=parameter_dict)
        individual_dict[table_row[1]] = p2
      parents = (p1, p2)
      child = individual_dict.get(table_row[2])
      if not child:
        child = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                              meta_dict=meta_dict, individual_idx=table_row[2],
                                              parameter_dict=parameter_dict)
        individual_dict[table_row[2]] = child
      crossover_dict[parents] = crossover_dict.get(parents, list()) + [child]
    result[self._DICT_CRO_REL] = [(list(key), val) for key, val in crossover_dict.items()]

    cursor.execute("SELECT parent, child FROM {}{} WHERE generation={}".format
                   (self._table_prefix, self._TABLE_MUTATION_REL, self._db_constants.PARAMETER), [gen_idx])
    mutation_dict = dict()
    while True:
      table_row = cursor.fetchone()
      if not table_row:
        break
      parent = individual_dict.get(table_row[0])
      if not parent:
        parent = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                               meta_dict=meta_dict, individual_idx=table_row[0],
                                               parameter_dict=parameter_dict)
        individual_dict[table_row[0]] = parent
      child = individual_dict.get(table_row[1])
      if not child:
        child = self._load_individual_from_db(class_dict=class_dict, dna_dict=dna_dict,
                                              meta_dict=meta_dict, individual_idx=table_row[1],
                                              parameter_dict=parameter_dict)
        individual_dict[table_row[1]] = child
      mutation_dict[parent] = mutation_dict.get(parent, list()) + [child]
    result[self._DICT_MUT_REL] = mutation_dict.items()

    self._conn.commit()
    cursor.close()
    return result

  @property
  def playback(self):
    return self._playback

  @property
  def descendants(self):
    return self._curr_gen[self._DICT_DES_IND]

  @property
  def prev_descendants(self):
    return self._prev_gen[self._DICT_DES_IND]

  @property
  def selection(self):
    return self._curr_gen[self._DICT_SEL_IND]

  @selection.setter
  def selection(self, selected):
    if isinstance(selected, list):
      self._curr_gen[self._DICT_SEL_IND] = selected

  @property
  def prev_selection(self):
    return self._prev_gen[self._DICT_SEL_IND]

  @property
  def mutation(self):
    return self._curr_gen[self._DICT_MUT_IND]

  @property
  def prev_mutation(self):
    return self._prev_gen[self._DICT_MUT_IND]

  @property
  def crossover(self):
    return self._curr_gen[self._DICT_CRO_IND]

  @property
  def prev_crossover(self):
    return self._prev_gen[self._DICT_CRO_IND]

  @property
  def mutation_ancestor(self):
    return self._curr_gen[self._DICT_MUT_REL]

  @property
  def crossover_ancestor(self):
    return self._curr_gen[self._DICT_CRO_REL]

  @property
  def generation(self):
    return self._curr_gen[self._DICT_GEN_IND]

  @property
  def prev_generation(self):
    return self._prev_gen[self._DICT_GEN_IND]

  @property
  def generation_idx(self):
    return self._generation

  @generation_idx.setter
  def generation_idx(self, other):
    if not self._playback:
      raise GenerationException("Not in playback mode!")
    tmp = self._gen_history.get(other)
    if tmp is None:
      raise GenerationException("You reached the end of the history!")
    self._generation = other
    self._prev_gen, self._curr_gen = self._gen_history.get(self._generation - 1, self._empty_gen_dict()), tmp

  pass


class GenerationManager(PlaybackGenManager):
  # <editor-fold desc="Constants">
  SELECTION = 0
  CROSSOVER = 1
  MUTATION = 2
  MUTATION_CROSSOVER = 3

  # </editor-fold>

  def __init__(self, **kwargs):
    super(GenerationManager, self).__init__(**kwargs)
    self._prev_gen = self._empty_gen_dict()
    self._curr_gen = self._empty_gen_dict()

    self._initStrat = kwargs.get('initializationStrategy')
    if self._initStrat is None:
      self._initStrat = RandomGeneration()
    gen = copy.deepcopy(self._initStrat.new_generation())
    for ind in gen:
      ind.id = None
    self._curr_gen[self._DICT_DES_IND] = gen
    self._curr_gen[self._DICT_GEN_IND] = self._curr_gen[self._DICT_DES_IND]

    self._generation = 0

    self.__replacement_scheme = kwargs.get('replacement', None)
    if self.__replacement_scheme is None:
      self.__replacement_scheme = NElitism()
    self.__gen_build = kwargs.get('generation', self.MUTATION)

    if self.__gen_build not in [self.CROSSOVER, self.MUTATION, self.MUTATION_CROSSOVER]:
      raise GenerationException("The new generation must depend on at least one of crossover and mutation!")

    self.__cro_base = kwargs.get('crossbreed', self.SELECTION)
    self.__mut_base = kwargs.get('mutate', self.CROSSOVER)

    if self.__mut_base is self.MUTATION:
      raise GenerationException("You cannot mutate nothing! Choose CROSSOVER or SELECTION!")
    if self.__cro_base is self.CROSSOVER:
      raise GenerationException("You cannot crossbreed nothing! Choose MUTATION or SELECTION!")
    if self.__mut_base is not self.SELECTION and self.__cro_base is not self.SELECTION:
      raise GenerationException("You build a circle! At least one of crossover and mutation must depend on SELECTION!")
    if self.__mut_base not in [self.SELECTION, self.CROSSOVER]:
      raise GenerationException("You can only mutate the selection or the crossover!")
    if self.__cro_base not in [self.SELECTION, self.MUTATION]:
      raise GenerationException("You can only crossbreed the selection or the mutation!")

    self.__log = kwargs.get('gen_log', None)
    self.__continue = kwargs.get('continue_log', False)

    self.__file = kwargs.get('file', None)
    if self.__file is not None and not isinstance(self.__file, str):
      self.__file = None
    if self.__file is not None:
      if not self.__continue:
        index = 0
        self.__file = self.__file.split('.')
        while True:
          _file = '.'.join(self.__file[:-1]) + '_%02i.' % index + self.__file[-1]
          if not os.path.isfile(_file):
            break
        self.__file = _file

    if self._memory_type is self.MEM_SQLITE3:
      import sqlite3
      self._to_binary = sqlite3.Binary
      self._conn = sqlite3.connect(self.__log)

      self._db_constants = DBSqlite3
      self.__setup_databases(**kwargs)
    elif self._memory_type is self.MEM_MySQL:
      import mysql.connector
      from mysql.connector import errorcode
      self._to_binary = lambda x: x

      self._user = kwargs.get('user', 'root')
      self._password = kwargs.get('password', '123456')
      self._host = kwargs.get('host', '127.0.0.1')
      self._conn = mysql.connector.connect(user=self._user, password=self._password, host=self._host)
      cursor = self._conn.cursor()
      cursor.execute("CREATE DATABASE IF NOT EXISTS %s DEFAULT CHARACTER SET 'utf8';" % (self.__log,))
      cursor.execute("USE %s" % (self.__log,))
      self._conn.commit()
      cursor.close()

      self._db_constants = DBMySql
      self.__setup_databases(**kwargs)
    elif self._memory_type is self.MEM_STATS:
      if self.__continue:
        raise GenerationException('Stats do NOT support to continue from a previous state!')
      self.dna_set = set()
      self.dna_evaluated_set = set()
      self.dna_generation_set = set()
      self.ind_idx = 0
      self.ind_evaluated_set = set()
      self.ind_generation_set = set()
    else:
      raise Exception('Unsupported memory type!')

    self._playback = False

    self.__gen_func = {self.CROSSOVER:
                         lambda: self._curr_gen[self._DICT_CRO_IND],
                       self.MUTATION:
                         lambda: self._curr_gen[self._DICT_MUT_IND],
                       self.MUTATION_CROSSOVER:
                         lambda: self._curr_gen[self._DICT_MUT_IND] + self._prev_gen[self._DICT_CRO_IND],
                       }.get(self.__gen_build)
    self.__cro_gen = {self.SELECTION:
                        lambda: self._prev_gen[self._DICT_SEL_IND],
                      self.MUTATION:
                        lambda: self._curr_gen[self._DICT_MUT_IND],
                      }.get(self.__cro_base)
    self.__mut_gen = {self.SELECTION:
                        lambda: self._prev_gen[self._DICT_SEL_IND],
                      self.CROSSOVER:
                        lambda: self._curr_gen[self._DICT_CRO_IND],
                      }.get(self.__mut_base)

  def __setup_databases(self, **kwargs):
    self._table_prefix = kwargs.get('table_prefix', '')
    cursor = self._conn.cursor()
    index = 0
    if not self.__continue:
      query = ''
      if self._memory_type is self.MEM_SQLITE3:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '"
      elif self._memory_type is self.MEM_MySQL:
        query = "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '"
      while True:
        table_name = self._table_prefix + '_%02i' % index
        cursor.execute(query + table_name + "%' LIMIT 2;")
        rows = cursor.fetchall()
        if len(rows) < 1:
          break
        index += 1
    self._table_prefix = self._table_prefix + '_%02i' % index

    autoincrement = ''
    if self._memory_type is self.MEM_SQLITE3:
      autoincrement = 'AUTOINCREMENT'
    elif self._memory_type is self.MEM_MySQL:
      autoincrement = 'AUTO_INCREMENT'

    """
    TABLE_CLASSES
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ class BLOB
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (rowid INTEGER PRIMARY KEY {}, class BLOB);".format
                   (self._table_prefix, self._TABLE_CLASSES, autoincrement))

    """
    TABLE_META_DATA
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ meta_data BLOB
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (rowid INTEGER PRIMARY KEY {}, meta_data BLOB);".format
                   (self._table_prefix, self._TABLE_META_DATA, autoincrement))


    """
    TABLE_CLASS_PARAMETER
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ parameters BLOB
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (rowid INTEGER PRIMARY KEY {}, parameters BLOB);".format
                   (self._table_prefix, self._TABLE_CLASS_PARAMETER, autoincrement))

    """
    TABLE_DNA
    ------------
    ~ rowid INTEGER PRIMARY KEY 
    ~ DNASequence BLOB UNIQUE
    ~ class INTEGER REFERENCES TABLE_CLASSES(rowid)
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (rowid INTEGER PRIMARY KEY {}, dna BLOB, class INTEGER, "
                   "parameters INTEGER, "
                   "FOREIGN KEY (class) REFERENCES {}{}(rowid),"
                   "FOREIGN KEY (parameters) REFERENCES {}{}(rowid));".format
                   (self._table_prefix, self._TABLE_DNA, autoincrement, self._table_prefix, self._TABLE_CLASSES,
                    self._table_prefix, self._TABLE_CLASS_PARAMETER))

    """
    TABLE_INDIVIDUALS
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ dna INTEGER REFERENCES TABLE_DNA(rowid)
    ~ class INTEGER REFERENCES TABLE_CLASSES(rowid)
    ~ meta_data INTEGER REFERENCES TABLE_META_DATA
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (rowid INTEGER PRIMARY KEY {}, "
                   "dna INTEGER, fitness REAL, class INTEGER, meta_data INTEGER, parameters INTEGER,"
                   "FOREIGN KEY (dna) REFERENCES {}{}(rowid),"
                   "FOREIGN KEY (class) REFERENCES {}{}(rowid),"
                   "FOREIGN KEY (meta_data) REFERENCES {}{}(rowid),"
                   "FOREIGN KEY (parameters) REFERENCES {}{}(rowid));".format
                   (self._table_prefix, self._TABLE_INDIVIDUALS, autoincrement,
                    self._table_prefix, self._TABLE_DNA,
                    self._table_prefix, self._TABLE_CLASSES,
                    self._table_prefix, self._TABLE_META_DATA,
                    self._table_prefix, self._TABLE_CLASS_PARAMETER))

    """
    TABLE_GENERATION_IDXS
    ------------
    ~ id INTEGER PRIMARY KEY
    """
    cursor.execute(
      "CREATE TABLE IF NOT EXISTS {}{} (id INTEGER PRIMARY KEY);".format
      (self._table_prefix, self._TABLE_GENERATION_IDXS))

    for _table in [self._TABLE_DESCENDANTS, self._TABLE_SELECTION, self._TABLE_GENERATION, self._TABLE_CROSSOVER_IND,
                   self._TABLE_MUTTAION_IND]:
      """
      TABLE_%s_IND
      ------------
      ~ individual INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
      ~ generation INTEGER REFERENCES TABLE_GENERATION_IDXS(id)
      ~ PRIMARY KEY (individual, generation)
              """
      cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (generation INTEGER, individual INTEGER,"
                     "FOREIGN KEY (generation) REFERENCES {}{}(id),"
                     "FOREIGN KEY (individual) REFERENCES {}{}(rowid){});".format
                     (self._table_prefix, _table, self._table_prefix, self._TABLE_GENERATION_IDXS,
                      self._table_prefix, self._TABLE_INDIVIDUALS,
                      ', PRIMARY KEY (generation, individual)' if _table in [self._TABLE_CROSSOVER_IND,
                                                                             self._TABLE_MUTTAION_IND] else ''))

    """
    TABLE_CROSSOVER_REL
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ generation INTEGER REFERENCES TABLE_GENERATION_IDXS(id)
    ~ parent1 INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
    ~ parent2 INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
    ~ child INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
    """
    cursor.execute(
      "CREATE TABLE IF NOT EXISTS {}{} (generation INTEGER, parent1 INTEGER, parent2 INTEGER, child INTEGER,"
      "FOREIGN KEY (generation) REFERENCES {}{}(id),"
      "FOREIGN KEY (parent1) REFERENCES {}{}(rowid),"
      "FOREIGN KEY (parent2) REFERENCES {}{}(rowid),"
      "FOREIGN KEY (child) REFERENCES {}{}(rowid));".format
      (self._table_prefix, self._TABLE_CROSSOVER_REL,
       self._table_prefix, self._TABLE_GENERATION_IDXS,
       self._table_prefix, self._TABLE_INDIVIDUALS,
       self._table_prefix, self._TABLE_INDIVIDUALS,
       self._table_prefix, self._TABLE_INDIVIDUALS))

    """
    TABLE_MUTATION_REL
    ------------
    ~ rowid INTEGER PRIMARY KEY
    ~ generation INTEGER REFERENCES TABLE_GENERATION_IDXS(id)
    ~ parent INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
    ~ child INTEGER REFERENCES TABLE_INDIVIDUALS(rowid)
    """
    cursor.execute("CREATE TABLE IF NOT EXISTS {}{} (generation INTEGER, parent INTEGER, child INTEGER,"
                   "FOREIGN KEY (generation) REFERENCES {}{}(id),"
                   "FOREIGN KEY (parent) REFERENCES {}{}(rowid),"
                   "FOREIGN KEY (child) REFERENCES {}{}(rowid));".format
                   (self._table_prefix, self._TABLE_MUTATION_REL,
                    self._table_prefix, self._TABLE_GENERATION_IDXS,
                    self._table_prefix, self._TABLE_INDIVIDUALS,
                    self._table_prefix, self._TABLE_INDIVIDUALS))
    try:
      if self.__continue:
        cursor.execute(
          "SELECT id FROM {}{} ORDER BY id DESC LIMIT 1".format(self._table_prefix, self._TABLE_GENERATION_IDXS))
        fetched = cursor.fetchone()
        if fetched:
          self._generation = fetched[0]
          class_dict, dna_dict, individual_dict, meta_dict, parameter_dict = dict(), dict(), dict(), dict(), dict()
          self._curr_gen = self._load_generation_from_db(class_dict=class_dict,
                                                         dna_dict=dna_dict,
                                                         individual_dict=individual_dict,
                                                         meta_dict=meta_dict,
                                                         parameter_dict=parameter_dict,
                                                         gen_idx=self._generation)
          if self._generation > 0:
            self._prev_gen = self._load_generation_from_db(class_dict=class_dict,
                                                           dna_dict=dna_dict,
                                                           individual_dict=individual_dict,
                                                           meta_dict=meta_dict,
                                                           parameter_dict=parameter_dict,
                                                           gen_idx=self._generation - 1)
          else:
            self._prev_gen = self._empty_gen_dict()
          cursor.execute("DELETE FROM {}{} WHERE generation=?;".format
                         (self._table_prefix, self._TABLE_SELECTION), [self._generation])
          print('Continue with generation %i' % self._generation)
        else:
          print('No generation found to continue with!')
          self.__continue = False
    except Exception as e:
      raise GenerationException('Failed to setup generation ids')
    self._conn.commit()
    cursor.close()
    pass

  def __mod__(self, mutationTuple):
    # """
    # Adds the relation between one GAIndividual and a list of GAIndividual.
    # :param mutationTuple: tuple of GAIndividual and list of GAIndividual
    # :return: None
    # """
    if isinstance(mutationTuple, tuple):
      self._curr_gen[self._DICT_MUT_REL].append(mutationTuple)
    if isinstance(mutationTuple, list):
      self._curr_gen[self._DICT_MUT_IND] += mutationTuple
    return None

  def __pow__(self, crossoverTuple):
    # """
    # Adds the relation between two GAIndividual and a list of GAIndividual.
    # :param crossoverTuple: tuple of list of GAIndividual and list of GAIndividual
    # :return: None
    # """
    if isinstance(crossoverTuple, tuple):
      self._curr_gen[self._DICT_CRO_REL].append(crossoverTuple)
    if isinstance(crossoverTuple, list):
      self._curr_gen[self._DICT_CRO_IND] = crossoverTuple
    return None

  def __pos__(self):
    if self._playback:
      super(GenerationManager, self).__pos__()
    else:
      if self.__log:
        if self._memory_type in [self.MEM_SQLITE3, self.MEM_MySQL]:
          cursor = self._conn.cursor()
          if self.__continue:
            for ind in self._curr_gen[self._DICT_SEL_IND]:
              cursor.execute(
                "INSERT INTO {}{} (generation, individual) VALUES ({},{});".format
                (self._table_prefix, self._TABLE_SELECTION,
                 self._db_constants.PARAMETER, self._db_constants.PARAMETER),
                [self._generation, ind.id])
            self.__continue = False
          else:

            cursor.execute("INSERT INTO {}{} (id) VALUES ({});"
                           .format(self._table_prefix, self._TABLE_GENERATION_IDXS, self._db_constants.PARAMETER),
                           [self._generation])
            for ind in self._curr_gen_ind():
              if ind.id is None:
                dna_class_binary = self._to_binary(pickle.dumps(ind.dna.__class__, pickle.HIGHEST_PROTOCOL))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE class={};".format
                                 (self._table_prefix, self._TABLE_CLASSES, self._db_constants.PARAMETER),
                                 [dna_class_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (class) VALUES ({});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_CLASSES,
                                  self._db_constants.PARAMETER),
                                 [dna_class_binary])
                dna_class_id = fetched[0]

                dna_parameters = ind.dna.__getstate__()
                dna_parameters.pop(AbstractDNA.DICT_REPRESENTATION, None)
                dna_parameters_binary = self._to_binary(pickle.dumps(dna_parameters))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE parameters={};".format
                                 (self._table_prefix, self._TABLE_CLASS_PARAMETER, self._db_constants.PARAMETER),
                                 [dna_parameters_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (parameters) VALUES ({});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_CLASS_PARAMETER,
                                  self._db_constants.PARAMETER),
                                 [dna_parameters_binary])
                dna_parameters_id = fetched[0]

                dna_binary = self._to_binary(pickle.dumps(ind.dna.representation, pickle.HIGHEST_PROTOCOL))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE dna={};".format
                                 (self._table_prefix, self._TABLE_DNA,
                                  self._db_constants.PARAMETER, self._db_constants.PARAMETER),
                                 [dna_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (dna, class, parameters) VALUES ({},{},{});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_DNA,
                                  self._db_constants.PARAMETER, self._db_constants.PARAMETER,
                                  self._db_constants.PARAMETER),
                                 [dna_binary, dna_class_id, dna_parameters_id])
                dna_id = fetched[0]

                ind_class_binary = self._to_binary(pickle.dumps(ind.__class__, pickle.HIGHEST_PROTOCOL))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE class={};".format
                                 (self._table_prefix, self._TABLE_CLASSES, self._db_constants.PARAMETER),
                                 [ind_class_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (class) VALUES ({});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_CLASSES,
                                  self._db_constants.PARAMETER),
                                 [ind_class_binary])
                ind_class_id = fetched[0]

                meta_data_binary = self._to_binary(pickle.dumps(ind.meta_data, pickle.HIGHEST_PROTOCOL))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE meta_data={};".format
                                 (self._table_prefix, self._TABLE_META_DATA, self._db_constants.PARAMETER),
                                 [meta_data_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (meta_data) VALUES ({});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_META_DATA,
                                  self._db_constants.PARAMETER), [meta_data_binary])
                meta_data_id = fetched[0]

                ind_parameters = ind.__getstate__()
                ind_parameters.pop(NEAIndividual.DICT_META, None)
                ind_parameters.pop(NEAIndividual.DICT_DNA, None)
                ind_parameters.pop(NEAIndividual.DICT_ID, None)
                ind_parameters.pop(NEAIndividual.DICT_FITNESS, None)
                ind_parameters_binary = self._to_binary(pickle.dumps(ind_parameters))
                while True:
                  cursor.execute("SELECT rowid FROM {}{} WHERE parameters={};".format
                                 (self._table_prefix, self._TABLE_CLASS_PARAMETER, self._db_constants.PARAMETER),
                                 [ind_parameters_binary])
                  fetched = cursor.fetchone()
                  if fetched:
                    break
                  cursor.execute("INSERT {} INTO {}{} (parameters) VALUES ({});".format
                                 (self._db_constants.IGNORE, self._table_prefix, self._TABLE_CLASS_PARAMETER,
                                  self._db_constants.PARAMETER),
                                 [ind_parameters_binary])
                ind_parameters_id = fetched[0]

                cursor.execute(
                  "INSERT INTO {}{} (dna,fitness,class,meta_data,parameters) VALUES ({},{},{},{},{});".format
                  (self._table_prefix, self._TABLE_INDIVIDUALS,
                   self._db_constants.PARAMETER, self._db_constants.PARAMETER, self._db_constants.PARAMETER,
                   self._db_constants.PARAMETER, self._db_constants.PARAMETER),
                  [dna_id, ind.f, ind_class_id, meta_data_id,ind_parameters_id])
                ind.id = cursor.lastrowid

            for key, table in [(self._DICT_DES_IND, self._TABLE_DESCENDANTS),
                               (self._DICT_SEL_IND, self._TABLE_SELECTION),
                               (self._DICT_GEN_IND, self._TABLE_GENERATION),
                               (self._DICT_CRO_IND, self._TABLE_CROSSOVER_IND),
                               (self._DICT_MUT_IND, self._TABLE_MUTTAION_IND)]:
              for ind in self._curr_gen[key]:
                cursor.execute(
                  "INSERT INTO {}{} (generation, individual) VALUES ({},{});".format
                  (self._table_prefix, table, self._db_constants.PARAMETER, self._db_constants.PARAMETER),
                  [self._generation, ind.id])

            for parents, children in self._curr_gen[self._DICT_CRO_REL]:
              for child in children:
                cursor.execute("INSERT INTO {}{} (generation, parent1, parent2, child) VALUES ({},{},{},{});".format
                               (self._table_prefix, self._TABLE_CROSSOVER_REL,
                                self._db_constants.PARAMETER, self._db_constants.PARAMETER,
                                self._db_constants.PARAMETER, self._db_constants.PARAMETER),
                               [self._generation] + [ind.id for ind in parents] + [child.id])
            for parent, children in self._curr_gen[self._DICT_MUT_REL]:
              for child in children:
                cursor.execute("INSERT INTO {}{} (generation, parent, child) VALUES ({},{},{});".format
                               (self._table_prefix, self._TABLE_MUTATION_REL,
                                self._db_constants.PARAMETER, self._db_constants.PARAMETER,
                                self._db_constants.PARAMETER),
                               [self._generation, parent.id, child.id])
          self._conn.commit()
          cursor.close()
        elif self._memory_type is self.MEM_STATS:
          for ind in self._curr_gen_ind():
            if not ind.id:
              self.ind_idx += 1
              ind.id = self.ind_idx
              self.dna_set.add(ind.dna)
            if ind.f >= 0:
              self.dna_evaluated_set.add(ind.dna)
              self.ind_evaluated_set.add(ind.id)
          for ind in self.generation:
            self.ind_generation_set.add(ind.id)
            self.dna_generation_set.add(ind.dna)

      self._prev_gen = self._curr_gen
      self._curr_gen = self._empty_gen_dict()
      self._generation += 1

  def __getstate__(self):
    result = dict()
    # init function
    result['initializationStrategy'] = self._initStrat
    result['replacement'] = self.__replacement_scheme
    result['generation'] = self.__gen_build
    result['crossbreed'] = self.__cro_base
    result['mutate'] = self.__mut_base
    result['gen_log'] = self.__log
    result['continue_log'] = self.__continue
    result['mem_type'] = self._memory_type
    result['file'] = self.__file
    if self._memory_type in [self.MEM_SQLITE3, self.MEM_MySQL]:
      result['table_prefix'] = self._table_prefix
    if self._memory_type is self.MEM_MySQL:
      result['user'] = self._user
      result['password'] = self._password
      result['host'] = self._host
    # manually assigned
    result['curr_gen'] = self._curr_gen
    result['prev_gen'] = self._prev_gen
    result['gen_idx'] = self._generation
    return result

  def __setstate__(self, state):
    self.__init__(**state)
    self._curr_gen = state.get('curr_gen')
    self._prev_gen = state.get('prev_gen')
    self._generation = state.get('gen_idx')

  def __iter__(self):
    if self.__log:
      state = self.__getstate__()
      state['filename'] = self.__log
      return PlaybackGenManager(**state)
    else:
      return None

  def __invert__(self):
    tmp = self.__gen_func()
    self._curr_gen[self._DICT_DES_IND] = tmp
    self._curr_gen[self._DICT_GEN_IND] = \
      self.__replacement_scheme.new_generation(self._prev_gen[self._DICT_GEN_IND], tmp)
    if self.__file:
      with open(self.__file, 'wb') as f:
        pickle.dump(self, f, -1)

  def __del__(self):
    if hasattr(self, '_conn'):
      try:
        self._conn.close()
      except Exception:
        pass

  def _curr_gen_ind(self):
    for ind in self._curr_gen[self._DICT_GEN_IND]:
      yield ind

    for ind in self._curr_gen[self._DICT_DES_IND]:
      yield ind

    first, second = self._DICT_CRO_REL, self._DICT_MUT_REL
    if self.__mut_base is self.SELECTION:
      first, second = second, first
    for key in [first, second]:
      for parents, childs in self._curr_gen[key]:
        if isinstance(parents, list):
          for ind in parents:
            yield ind
        else:
          yield parents
        for ind in childs:
          yield ind

  @staticmethod
  def restore(filename):
    with open(filename, 'rb') as f:
      return pickle.load(f)

  @property
  def mut_gen(self):
    return self.__mut_gen()

  @property
  def cro_gen(self):
    return self.__cro_gen()

  @property
  def log_file(self):
    return self.__log

  @property
  def table_prefix(self):
    if self._memory_type is self.MEM_SQLITE3:
      return self._table_prefix
    else:
      return self.__log

  def file(self):
    return self.__file

  def stop(self):
    if hasattr(self, '_conn'):
      try:
        self._conn.close()
      except Exception:
        pass
    if hasattr(self, 'dna_set'):
      try:
        del self.dna_set
      except Exception:
        pass
    if hasattr(self, 'ind_evaluated_set'):
      try:
        del self.ind_evaluated_set
      except Exception:
        pass
    if hasattr(self, 'dna_evaluated_set'):
      try:
        del self.dna_evaluated_set
      except Exception:
        pass
    if hasattr(self, 'dna_generation_set'):
      try:
        del self.dna_generation_set
      except Exception:
        pass
    if hasattr(self, 'ind_generation_set'):
      try:
        del self.ind_generation_set
      except Exception:
        pass

  pass
