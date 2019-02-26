from ea.dna import IndirectLinearDNA, AbstractDNA, CrossoverType, DNAException
import numpy as np
import random


class MNISTSmallDNA(IndirectLinearDNA):
  __conv = [3, 5, 7, 9, 11]
  __pool = [1, 2, 3, 4, 5]
  # True == conv stride size instead of max pool
  __stride_vs_pool = [True, False]
  __norm_min = np.asarray(
    [min(__stride_vs_pool), min(__conv), min(__pool), min(__stride_vs_pool), min(__conv), min(__pool)])
  __norm_max = np.asarray(
    [max(__stride_vs_pool), max(__conv), max(__pool), max(__stride_vs_pool), max(__conv), max(__pool)]) - __norm_min
  __search_mat = [__stride_vs_pool, __conv, __pool, __stride_vs_pool, __conv, __pool]

  def __init__(self, **kwargs):
    super(MNISTSmallDNA, self).__init__(**kwargs)
    if self._representation is None:
      self._representation = self.__random_representation()

    self.__representation_normed = (np.asarray(self._representation) - self.__norm_min) / self.__norm_max
    self.__representation_indexed = np.asarray(
      [__values.index(a) for a, __values in zip(self._representation, self.__search_mat)])
    self.__len = len(self._representation)

    self._crossover = {CrossoverType.N_POINT: self._cross_n_point,
                       CrossoverType.N_POINT_WITH_IDENTITY: self._cross_n_point_u_identity,
                       CrossoverType.X_POINT: self._cross_x_point,
                       }.get(self._crossoverType)

    self._mutation = self.__mutation
    self.__mutation_scale = kwargs.get('mutation_scale', 1.7)

    self._distance = self.__distance
    pass

  def __getstate__(self):
    result = super(MNISTSmallDNA, self).__getstate__()
    result['mutation_scale'] = self.__mutation_scale
    return result

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

  def __random_representation(self):
    st_pl1 = np.random.choice(self.__stride_vs_pool)
    conv1 = np.random.choice(self.__conv, 1)[0]
    pool1 = np.random.choice(self.__pool, 1)[0]
    st_pl2 = np.random.choice(self.__stride_vs_pool)
    conv2 = np.random.choice(self.__conv, 1)[0]
    pool2 = np.random.choice(self.__pool, 1)[0]
    return [st_pl1, conv1, pool1, st_pl2, conv2, pool2]

  def __distance(self, other):
    if not isinstance(other, self.__class__):
      return float('inf')
    return np.linalg.norm(self.__representation_normed - other.__representation_normed)

  def __mutation(self, prob):
    representation = [__v[int(min(max(0, round(i)), len(__v) - 1))] if p < prob else a
                      for p, i, a, __v in zip(np.random.uniform(0, 1, size=self.__len),
                                              np.random.normal(loc=self.__representation_indexed,
                                                               scale=self.__mutation_scale,
                                                               size=self.__len),
                                              self._representation,
                                              self.__search_mat)]
    return [self._new_DNA(representation)]

  pass


from ea.representations.tf_nodes import tf, TFOutputNode, TFNMaxPool2D, TFNConv2D, TFInputNode, TFNodeUtil
from ea.representations.k_ary_tree import KAryTree


class TFKTreeDNA(AbstractDNA):
  def __init__(self, **kwargs):
    super(TFKTreeDNA, self).__init__(**kwargs)
    self._crossover = self.__crossover
    self._mutation = self.__mutation
    self._distance = self.__distance

    self._shape = kwargs.get('shape')
    self._dtype = kwargs.get('dtype')
    self._data_format = kwargs.get('data_format')
    self._name = kwargs.get('input_name')
    self._save_mutation = kwargs.get('save_mutation', True)
    self._hier_mut = kwargs.get('hier_mut', True)

    if self._shape is None:
      self._shape = (None, 28, 28, 1)
    if self._dtype is None:
      self._dtype = tf.float32
    if self._data_format is None:
      self._data_format = TFNodeUtil.CHANNELS_LAST
    if self._name is None:
      self._name = 'X'
    if not isinstance(self._save_mutation, bool):
      self._save_mutation = True

    if not self._representation:
      self._representation = self.__random_representation()
    pass

  def __eq__(self, other):
    result = super(TFKTreeDNA, self).__eq__(other)
    return result

  def __hash__(self):
    return hash(self._representation)

  def __getstate__(self):
    result = super(TFKTreeDNA, self).__getstate__()
    result['shape'] = self._shape
    result['dtype'] = self._dtype
    result['data_format'] = self._data_format
    result['input_name'] = self._name
    result['save_mutation'] = self._save_mutation
    result['hier_mut'] = self._hier_mut
    return result

  @property
  def hier_mut(self):
    return self._hier_mut

  @hier_mut.setter
  def hier_mut(self, hier_mut_):
    self._hier_mut = hier_mut_

  def __random_representation(self):
    input_graph = KAryTree(node=TFInputNode(dtype=self._dtype, shape=self._shape, name=self._name), inputs=list())
    result = KAryTree(node=TFOutputNode(), inputs=list())
    stack = list([result])
    dna_max_in = np.asarray(TFNodeUtil.get_bhwc(self._shape, self._data_format)[1:3])
    while len(stack) > 0:
      graph = stack.pop()
      if random.random() < 1 / (graph.depth() * 2 + len(graph.children) * 2 + 1):
        stack.append(graph)
      layer_class = random.choice([TFNMaxPool2D, TFNConv2D])
      min_in, max_in = TFNodeUtil.get_bhwc(graph.min_inputsize(), graph.node.data_format), \
                       TFNodeUtil.get_bhwc(graph.max_inputsize(), graph.node.data_format)
      b, _, _, c = min_in
      min_in = np.asarray(min_in[1:3])
      max_in = np.minimum(np.asarray(max_in[1:3]),
                          np.asarray(dna_max_in))
      outputsize = (np.random.beta(1, 3, 2) * (max_in - min_in) + min_in).round()
      inputsize = (np.random.beta(1, 3, 2) * (dna_max_in - outputsize) + outputsize).round()

      outputsize = TFNodeUtil.from_bhwc(b, int(outputsize[0]), int(outputsize[1]), c, graph.node.data_format)
      inputsize = TFNodeUtil.from_bhwc(b, int(inputsize[0]), int(inputsize[1]), c, graph.node.data_format)

      add_to_stack = True
      if random.random() > (1 / (graph.depth() + 1) + .2):
        inputsize = self._shape
        add_to_stack = False

      configs = layer_class.parameter_range(inputsize,
                                            outputsize,
                                            data_format=input_graph.node.data_format)
      if len(configs) > 0:
        config = random.choice(configs)
        if layer_class == TFNConv2D:
          config['activation'] = tf.nn.relu
        new_graph = KAryTree(node=layer_class(**config), inputs=list())
        graph.add_child(new_graph)
        if add_to_stack:
          stack.append(new_graph)
        else:
          new_graph.add_child(input_graph.__copy__())
      else:
        raise DNAException('Failed to create GraphDNA! This should not happen!')
    result.outputsize()
    return result

  def _new_ind(self, representation):
    state = self.__getstate__()
    state[self.DICT_REPRESENTATION] = representation
    return TFKTreeDNA(**state)

  def __crossover(self, other):
    result = list()
    for source_tree, crossing_base in [(other._representation, self._representation),
                                       (self._representation, other._representation),
                                       ]:
      crossing_result = crossing_base.small_copy_replace()
      stack = list()
      for subtree in crossing_base.children:
        stack.append((crossing_result, subtree, source_tree, crossing_base))
      while len(stack) > 0:
        new_tree, crossing_point, tree_pool, other_tree = stack.pop()
        replaced = False
        new_subtree = None
        replacing_tree = None
        if random.random() <= .5:
          referrence_out = np.asarray(TFNodeUtil.get_bhwc(crossing_point.parent.min_inputsize(origin=crossing_point),
                                                          crossing_point.parent.node.data_format)[1:3])
          referrence_out = np.maximum(referrence_out,
                                      np.asarray(TFNodeUtil.get_bhwc(new_tree.min_inputsize(),
                                                                     new_tree.node.data_format)[1:3]))
          possible_new_trees = [subtree for subtree in tree_pool.trees
                                if not isinstance(subtree.node, TFOutputNode)
                                and not isinstance(subtree.node, TFInputNode)
                                and all([i >= j for i, j in zip(TFNodeUtil.get_bhwc(subtree.min_inputsize(),
                                                                                    subtree.node.data_format)[1:3],
                                                                referrence_out)])]
          if len(possible_new_trees) > 0:
            previous_out = np.asarray(TFNodeUtil.get_bhwc(crossing_point.outputsize(),
                                                          crossing_point.node.data_format)[1:3])
            prob = np.asarray([1 / (np.linalg.norm(
              np.asarray(TFNodeUtil.get_bhwc(subtree.outputsize(),
                                             subtree.node.data_format)[1:3]) - previous_out)
                                    ** 2 + .1) for subtree in possible_new_trees])
            replacing_tree = np.random.choice(possible_new_trees, replace=False, p=prob / prob.sum())

            inputsize = TFNodeUtil.get_bhwc(replacing_tree.min_inputsize(), replacing_tree.node.data_format)

            min_out = np.asarray(
              TFNodeUtil.get_bhwc(crossing_point.parent.min_inputsize(origin=crossing_point),
                                  crossing_point.parent.node.data_format)[1:3])
            min_out = np.maximum(min_out, np.asarray(TFNodeUtil.get_bhwc(new_tree.min_inputsize(),
                                                                         new_tree.node.data_format)[1:3]))

            max_out = np.asarray(
              TFNodeUtil.get_bhwc(crossing_point.parent.max_inputsize(origin=crossing_point),
                                  crossing_point.parent.node.data_format)[1:3])
            max_out = np.minimum(max_out, np.asarray(TFNodeUtil.get_bhwc(new_tree.max_inputsize(),
                                                                         new_tree.node.data_format)[1:3]))
            max_out = np.minimum(max_out, np.asarray(inputsize[1:3]))
            outputsize = (np.random.beta(2, 2, 2) * (max_out - min_out) + min_out).round()
            outputsize = (inputsize[0], int(outputsize[0]), int(outputsize[1]), inputsize[2])

            distance = np.linalg.norm(previous_out - np.asarray(
              TFNodeUtil.get_bhwc(replacing_tree.outputsize(), replacing_tree.node.data_format)[1:3]))
            if random.random() < (distance + .001) ** (-.3):
              # modify replacing subgraph if necessary
              if all([i <= j <= k for i, j, k in zip(
                      min_out,
                      TFNodeUtil.get_bhwc(replacing_tree.outputsize(), replacing_tree.node.data_format)[1:3],
                      max_out)]):
                # no modifications necessary
                new_subtree = replacing_tree.small_copy_replace()
                if len(replacing_tree.children) > 1:
                  new_subtree.min_in_override = replacing_tree.min_inputsize()
                  new_subtree.max_in_override = replacing_tree.max_inputsize()
                tree_pool, other_tree = other_tree, crossing_point
                # tree_pool, other_tree = other_tree, tree_pool
                replaced = True
              else:
                # try to modify
                node_class = replacing_tree.node.__class__
                parameters = node_class.parameter_range(inputsize=inputsize, outputsize=outputsize)

                if len(parameters) > 0:
                  node_state = replacing_tree.node.__getstate__()
                  node_state['name'] = None
                  for key, value in random.choice(parameters).items():
                    node_state[key] = value
                  new_subtree = replacing_tree.small_copy_replace(node=node_class(**node_state))
                  if len(replacing_tree.children) > 1:
                    new_subtree.min_in_override = replacing_tree.min_inputsize()
                    new_subtree.max_in_override = replacing_tree.max_inputsize()
                  tree_pool, other_tree = other_tree, crossing_point
                  # tree_pool, other_tree = other_tree, tree_pool
                  replaced = True
            else:
              # insert intermediate node
              node_class = random.choice([TFNConv2D, TFNMaxPool2D])
              parameters = node_class.parameter_range(inputsize=replacing_tree.outputsize(),
                                                      outputsize=outputsize)
              if len(parameters) > 0:
                config = random.choice(parameters)
                if node_class == TFNConv2D:
                  config['activation'] = tf.nn.relu
                intermediate_tree = replacing_tree.small_copy_replace(node=node_class(**config))
                new_tree.add_child(intermediate_tree)
                new_tree = intermediate_tree
                new_subtree = replacing_tree.small_copy_replace()
                if len(replacing_tree.children) > 1:
                  new_subtree.min_in_override = replacing_tree.min_inputsize()
                  new_subtree.max_in_override = replacing_tree.max_inputsize()
                tree_pool, other_tree = other_tree, crossing_point
                # tree_pool, other_tree = other_tree, tree_pool
                replaced = True

        if not replaced:
          # copy original tree if not replaced
          replacing_tree = crossing_point
          new_subtree = replacing_tree.small_copy_replace()
          if len(replacing_tree.children) > 1:
            new_subtree.min_in_override = replacing_tree.min_inputsize()
            new_subtree.max_in_override = replacing_tree.max_inputsize()
        new_tree.add_child(new_subtree)
        for subtree in replacing_tree.children:
          stack.append((new_subtree, subtree, tree_pool, other_tree))
      crossing_result.outputsize()
      for subTree in crossing_result.trees:
        subTree.min_in_override = None
        subTree.max_in_override = None
      crossing_result.outputsize()
      result.append(self._new_ind(crossing_result))
    return result

  def __mutation(self, prob):
    def delete_branch(parent, child, stack):
      if len(child.parent.children) > 1:
        return True
      return False

    def add_node(parent, child, stack):
      inputsize = TFNodeUtil.get_bhwc(child.outputsize(), child.node.data_format)
      in_np = np.asarray(inputsize[1:3])
      min_outputsize = np.asarray(
        TFNodeUtil.get_bhwc(child.parent.min_inputsize(origin=child), child.parent.node.data_format)[1:3])
      max_outputsize = np.asarray(
        TFNodeUtil.get_bhwc(child.parent.max_inputsize(origin=child), child.parent.node.data_format)[1:3])
      max_outputsize = np.minimum(max_outputsize, in_np)

      if self._save_mutation:
        # prevents mutation errors but makes mutations dependent
        max_outputsize = np.minimum(max_outputsize, np.asarray(TFNodeUtil.get_bhwc(parent.max_inputsize(),
                                                                                   parent.node.data_format)[1:3]))
        min_outputsize = np.maximum(min_outputsize, np.asarray(TFNodeUtil.get_bhwc(parent.min_inputsize(),
                                                                                   parent.node.data_format)[1:3]))

      # outputsize = np.random.uniform(min_outputsize, max_outputsize).round()
      outputsize = (np.random.beta(2, 2, 2) * (max_outputsize - min_outputsize) + min_outputsize).round()
      outputsize = (inputsize[0], int(outputsize[0]), int(outputsize[1]), inputsize[2])

      node_class = random.choice([TFNConv2D, TFNMaxPool2D])
      parameters = node_class.parameter_range(inputsize=inputsize, outputsize=outputsize)
      if len(parameters) <= 0:
        raise DNAException('Failed to mutate DNA!')
      config = random.choice(parameters)
      if node_class == TFNConv2D:
        config['activation'] = tf.nn.relu
      extra_node = child.small_copy_replace(node=node_class(**config))
      new_node = child.small_copy_replace()
      for c in child.children:
        stack.append((new_node, c))
      parent.add_child(extra_node)
      extra_node.add_child(new_node)
      if len(child.children) > 1:
        new_node.min_in_override = child.min_inputsize()
        new_node.max_in_override = child.max_inputsize()
      return True

    def change_node(parent, child, stack):
      if isinstance(child.node, TFInputNode):
        return False
      # min_inputsize() equals max_inputsize() as long as the mutated graph is valid and each subgraph has itself
      # at least one subgraph except for input subgraphs
      inputsize = TFNodeUtil.get_bhwc(child.min_inputsize(), child.node.data_format)
      min_out = np.asarray(TFNodeUtil.get_bhwc(child.parent.min_inputsize(origin=child),
                                               child.parent.node.data_format)[1:3])
      max_out = np.asarray(TFNodeUtil.get_bhwc(child.parent.max_inputsize(origin=child),
                                               child.parent.node.data_format)[1:3])
      max_out = np.minimum(max_out, np.asarray(inputsize[1:3]))

      if self._save_mutation:
        # prevents mutation errors but makes mutations dependent
        max_out = np.minimum(max_out, np.asarray(TFNodeUtil.get_bhwc(parent.max_inputsize(),
                                                                     parent.node.data_format)[1:3]))
        min_out = np.maximum(min_out, np.asarray(TFNodeUtil.get_bhwc(parent.min_inputsize(),
                                                                     parent.node.data_format)[1:3]))

      outputsize = (np.random.beta(2, 2, 2) * (max_out - min_out) + min_out).round()
      outputsize = (None, int(outputsize[0]), int(outputsize[1]), None)

      if random.random() < .5:
        # keep node type
        node_state = child.node.__getstate__()
        node_state['name'] = None
        node_class = child.node.__class__
        parameters = node_class.parameter_range(inputsize=inputsize, outputsize=outputsize)

        # probability to change padding .5
        padding = random.choice([TFNodeUtil.CHANNELS_LAST, TFNodeUtil.CHANNELS_FIRST])
        filtered_parameters = [param_set for param_set in parameters if param_set.get('padding') == padding]
        if len(filtered_parameters) <= 0:
          filtered_parameters = parameters

        if len(filtered_parameters) == 0:
          raise DNAException('Failed to mutate node!')

        for key, value in random.choice(filtered_parameters).items():
          node_state[key] = value

        if node_class == TFNConv2D:
          filter_n = node_state.get('filters')
          _min = max(filter_n - 5, 1)
          _max = filter_n + 5
          node_state['filters'] = round(np.random.beta(2, 2) * (_max - _min) + _min)

        new_node = child.small_copy_replace(node=node_class(**node_state))
        if len(child.children) > 1:
          new_node.min_in_override = child.min_inputsize()
          new_node.max_in_override = child.max_inputsize()
        for c in child.children:
          stack.append((new_node, c))
        parent.add_child(new_node)
        return True
      else:
        # change node type
        node_classes = [TFNConv2D, TFNMaxPool2D]
        node_classes.remove(child.node.__class__)
        node_class = random.choice(node_classes)
        parameters = node_class.parameter_range(inputsize=inputsize, outputsize=outputsize)
        if len(parameters) == 0:
          raise DNAException('Failed to mutate node!')
        parameter = random.choice(parameters)
        if node_class == TFNConv2D:
          parameter['activation'] = tf.nn.relu
        new_node = child.small_copy_replace(node=node_class(**parameter))
        if len(child.children) > 1:
          new_node.min_in_override = child.min_inputsize()
          new_node.max_in_override = child.max_inputsize()
        for c in child.children:
          stack.append((new_node, c))
        parent.add_child(new_node)
        return True

    def duplicate_branch(parent, child, stack):
      if isinstance(child.node, TFInputNode):
        return False
      new_node = child.small_copy_replace()
      for c in child.children:
        stack.append((new_node, c))
      if len(child.children) > 1:
        new_node.min_in_override = child.min_inputsize()
        new_node.max_in_override = child.max_inputsize()
      parent.add_child(new_node)
      parent.add_child(child.__copy__())
      return True

    def adding_ops():
      return [add_node, duplicate_branch]

    def modifying_ops():
      return [change_node]

    def removing_ops():
      return [delete_branch]

    tries = 0
    while True:
      tries += 1
      result = self._representation.small_copy_replace()

      stack = list()
      for child in self._representation.children:
        stack.append((result, child))

      try:
        while len(stack) > 0:
          parent, current_subgraph = stack.pop()
          if random.random() <= prob:
            success = False
            if self._hier_mut:
              mutation_operations = [adding_ops(), modifying_ops(), removing_ops()]
            else:
              mutation_operations = [add_node, duplicate_branch, change_node, delete_branch]
            while not success:
              if len(mutation_operations) < 1:
                raise DNAException('Failed to mutate DNA!')
              if self._hier_mut:
                op_list = random.choice(mutation_operations)
                mutation_op = random.choice(op_list)
              else:
                mutation_op = random.choice(mutation_operations)
              success = mutation_op(parent, current_subgraph, stack)
              if self._hier_mut:
                op_list.remove(mutation_op)
                if len(op_list) < 1:
                  mutation_operations.remove(op_list)
              else:
                mutation_operations.remove(mutation_op)
          else:
            new_subgraph = current_subgraph.small_copy_replace()
            if len(current_subgraph.children) > 1:
              new_subgraph.min_in_override = current_subgraph.min_inputsize()
              new_subgraph.max_in_override = current_subgraph.max_inputsize()
            for child in current_subgraph.children:
              stack.append((new_subgraph, child))
            parent.add_child(new_subgraph)
        result.outputsize()
        for subTree in result.trees:
          subTree.min_in_override = None
          subTree.max_in_override = None
        result.outputsize()
        break
      except Exception as e:
        if tries > 40:
          raise DNAException('Mutation failed! Tries exceeded!')
    result = self._new_ind(result)
    return [result]

  def __distance(self, other):
    if not isinstance(other, self.__class__):
      return float('inf')
    return self.representation - other.representation

  pass
