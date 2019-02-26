import paho.mqtt.client as mqtt
import pickle
from ea.individuals import NEAIndividual
import tensorflow as tf
from multiprocessing import Process, Pipe, Manager, Lock, Queue
import time
from compute.evaluation_helper import AbstractEvaluationHelper
from queue import Empty
import json


class NEAServerException(Exception):
  pass


class ServerClient(AbstractEvaluationHelper):
  def __init__(self, **kwargs):
    super(ServerClient, self).__init__(**kwargs)
    self._echo_counter = 0
    self._echo_interval = kwargs.get('echo_interval', 10)
    self._echo_topic_send = kwargs.get('echo_topic_send', 'EA_echo')
    self._echo_topic_receive = kwargs.get('echo_topic_receive', 'EA_Servers')
    self._echo_sleep = kwargs.get('echo_wait', 10)
    self._ping_interval = kwargs.get('ping_interval', 60)
    self._server_list = list()

    self._fitness_lookup = dict()
    self._fitness_topic = kwargs.get('fitness_topic', 'EA_Fitness')

    self._t_token = kwargs.get('telegram_token')
    self._t_chat_ids = kwargs.get('telegram_ids', [])
    if self._t_token is not None:
      from telegram import Bot
      self._t_bot = Bot(self._t_token)
    else:
      self._t_bot = None

    self._server = kwargs.get('server', ("iot.eclipse.org", 1883, 60))
    self._client = mqtt.Client()
    self._client.on_connect = self.__client_on_connect
    self._client.on_message = self.__client_on_message
    self._client.connected_flag = False
    _host, _port, _keepalive = self._server
    self._client.connect(host=_host, port=_port, keepalive=_keepalive)
    self._client.loop_start()
    while not self._client.connected_flag:
      time.sleep(.1)

  def __del__(self):
    self._client.loop_stop()

  def __getstate__(self):
    result = super(ServerClient, self).__getstate__()
    result['echo_interval'] = self._echo_interval
    result['echo_topic_send'] = self._echo_topic_send
    result['echo_topic_receive'] = self._echo_topic_receive
    result['echo_wait'] = self._echo_sleep
    result['fitness_lookup'] = self._fitness_lookup
    result['fitness_topic'] = self._fitness_topic
    result['server'] = self._server
    return result

  def __setstate__(self, state):
    self.__init__(**state)
    self._fitness_lookup = state.get('fitness_lookup', dict())

  def __client_on_connect(self, client, userdata, flags, rc):
    if rc==0:
      client.connected_flag = True
      client.subscribe(self._echo_topic_receive)
      print('Connected with result code: ' + str(rc))
    else:
      print('Connection refused!')

  def __client_on_message(self, client, userdata, msg):
    try:
      payload = json.loads(msg.payload.decode('utf-8'))
      topic = payload.get('topic')
      if topic is not None:
        self._server_list.append(topic)
    except Exception:
        pass

  def __echo(self):
    self._server_list.clear()
    self._client.publish(self._echo_topic_send, json.dumps({
      'topic': self._echo_topic_receive}))
    time.sleep(self._echo_sleep)
    message = 'Echo compute server. Found %i servers:' % len(self._server_list)
    message += '\n\t' + '\n\t'.join(self._server_list)
    if self._t_bot:
      for id in self._t_chat_ids:
        try:
          self._t_bot.send_message(id, message)
        except Exception:
          print('Failed to send telegram message!')
    pass

  def __server_communication(self, lock_, ind_list_, ind_on_server, topic_send, topic_receive, send_end):
    value = None
    current_id = None
    current_alive = True

    def connect(client, userdata, flags, rc):
      if rc == 0:
        client.connected_flag = True
        client.subscribe(topic_receive)
      else:
        print('Connected with result code: ' + str(rc))

    def message(client, userdata, msg):
      if msg.topic == topic_receive:
        try:
          payload = json.loads(msg.payload.decode('utf-8'))
          nonlocal value
          nonlocal current_alive
          nonlocal current_id
          id_ = payload.get('id', None)
          if current_id == id_:
            value_ = payload.get('result', None)
            if value_ is not None:
              value = pickle.loads(bytes.fromhex(value_))
            current_alive = payload.get('alive', True)
        except Exception:
          pass

    _client = mqtt.Client()
    _client.connected_flag = False
    _client.on_connect = connect
    _client.on_message = message
    _host, _port, _keepalive = self._server
    _client.connect(host=_host, port=_port, keepalive=_keepalive)
    _client.loop_start()
    while not _client.connected_flag:
      time.sleep(.1)
    keep_running = True
    while (len(ind_list_) > 0 or len(ind_on_server) > 0) and keep_running:
      if len(ind_list_) == 0:
        time.sleep(self._ping_interval / 2)
        continue
      with lock_:
        ind = ind_list_.pop(0)
        ind_on_server.append(ind)
      current_alive = True
      current_id = hash(ind)
      _client.publish(topic_send, json.dumps({'topic': topic_receive,
                                              'workload': True,
                                              'id': current_id,
                                              'ind': pickle.dumps(ind, -1).hex()}))
      ping_count = 0
      while (value is None):
        time.sleep(1)
        ping_count += 1
        if ping_count % 60 == 0:
          ping_count = 0
          if not current_alive:
            self._server_list.remove(topic_send)
            with lock_:
              ind_list_.append(ind)
            keep_running = False
            break
          _client.publish(topic_send, json.dumps({'topic': topic_receive,
                                                  'ping': True,
                                                  'id': current_id}))

      if value is not None and value[0] >= 0 and ind in ind_on_server:
        send_end.put((ind.dna, value))
        with lock_:
          try:
            ind_on_server.remove(ind)
          except Exception:
            pass
        value = None
    _client.loop_stop()
    _client.disconnect()
    pass

  def evaluate_pool(self, pool):
    if self._echo_counter == 0:
      self.__echo()

    m = Manager()
    ind_list = m.list()
    ind_on_server = m.list()
    dna_list = list()
    for ind in pool:
      if ind.f < 0:
        lookup_val = self._fitness_lookup.get(ind.dna)
        if lookup_val is None and ind.dna not in dna_list:
          ind_list.append(ind)
          dna_list.append(ind.dna)
        else:
          ind.f = lookup_val
    pool_size = min(len(self._server_list), len(ind_list))

    lock = Lock()
    queue = Queue()
    for idx, t_send in zip(range(pool_size), self._server_list):
      t_receive = self._fitness_topic + '/client_%i' % idx
      p = Process(target=self.__server_communication, args=(lock, ind_list, ind_on_server, t_send, t_receive, queue))
      p.daemon = True
      p.start()
    while (len(ind_list) > 0) or (not queue.empty()) or (len(ind_on_server) > 0):
      try:
        key, val = queue.get()
        self._fitness_lookup[key] = val
      except Empty:
        continue
      time.sleep(.5)
    for ind in pool:
      if ind.f < 0:
        ind.f, ind.meta_data = self._fitness_lookup.get(ind.dna)

    self._echo_counter = (self._echo_counter + 1) % self._echo_interval
    pass


class BaseServer():
  def __init__(self, **kwargs):
    self._server = kwargs.get('server', ("iot.eclipse.org", 1883, 60))

    self._topic_echo = kwargs.get('echo', 'EA_echo')
    self._topic_subscribe = kwargs.get('data_topic', 'EA_Individual')

    self._client = None

    self._eval = None
    self._pipe_out, self._pipe_in = Pipe(False)
    self._alive_processes = dict()
    pass

  def _on_connect(self, client, userdata, flags, rc):
    if rc==0:
      client.subscribe(self._topic_echo)
      client.subscribe(self._topic_subscribe)
      print('Connected with result code %s' % (str(rc)))
    else:
      print('Connection refused!')

  def _on_message(self, client, userdata, msg):
    try:
      payload = json.loads(msg.payload.decode('utf-8'))
      if msg.topic == self._topic_echo:
        if payload.get('topic') is not None:
          client.publish(payload.get('topic'), json.dumps({'topic': self._topic_subscribe}))
        self._alive_processes = dict([(id_, p) for id_, p in self._alive_processes.items() if p.is_alive()])

      elif msg.topic == self._topic_subscribe:
        id_ = payload.get('id')
        if payload.get('workload', False) and id_ is not None:
          p = Process(target=self._eval, args=(self._pipe_in, payload))
          p.daemon = True
          p.start()
          self._alive_processes[id_] = p
        if payload.get('ping', False) and id_ is not None:
          p = self._alive_processes.get(id_)
          if p is not None and p.is_alive():
            alive = True
          else:
            alive = False
            self._alive_processes.pop(id_, None)
          client.publish(payload.get('topic'), json.dumps({'id': id_,
                                                           'alive': alive}))
          pass

    except Exception:
      pass
    pass

  def start(self):
    if self._client is not None:
      raise NEAServerException('Already started!')
    self._client = mqtt.Client()
    self._client.on_connect = self._on_connect
    self._client.on_message = self._on_message
    self._client.connect_flag = False

    host, port, keepalive = self._server
    print(host, port, keepalive)
    self._client.connect(host=host, port=port, keepalive=keepalive)
    self._client.loop_start()
    while True:
      try:
        topic, payload = self._pipe_out.recv()
        self._client.publish(topic, payload)
        time.sleep(1)
      except Exception as e:
        break
    self._client.loop_stop()
    self._client.disconnect()
    pass

  pass


class TensorflowServer(BaseServer):
  def __init__(self, **kwargs):
    super(TensorflowServer, self).__init__(**kwargs)
    self._eval = self.__evaluate
    self._tf_sess_conf = kwargs.get('tf_config')
    if self._tf_sess_conf is None:
      self._tf_sess_conf = tf.ConfigProto()
      self._tf_sess_conf.gpu_options.allow_growth = True

    self._ga_train = kwargs.get('train_data')
    self._ga_valid = kwargs.get('valid_data')
    self._ga_test = kwargs.get('test_data')
    self._batch_size = kwargs.get('batch_size', 32)

    self._log_folder = kwargs.get('log_path')
    pass

  def __evaluate(self, pipe_in, msg):
    try:
      reply = msg.get('topic', '')
      individual = pickle.loads(bytes.fromhex(msg.get('ind')))

      if not isinstance(individual, NEAIndividual):
        pipe_in.send((reply, (-.1, None)))

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

      meta_data = individual.evaluate(**eval_options)
      pipe_in.send((reply, json.dumps({'id': msg.get('id'),
                                       'result': pickle.dumps((individual.f, meta_data), -1).hex()})))
    except Exception as e:
      print('Evaluation Failed!!')
      print('Error:')
      print(e)
      pipe_in.send((reply, json.dumps({'id': msg.get('id'),
                                       'result': pickle.dumps((-1, None), -1).hex()})))

    pass

  pass
