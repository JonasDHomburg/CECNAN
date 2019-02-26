from ea.representations.nodes import AbstractNode, AbstractInputNode, AbstractOutputNode

class TreeException(Exception):
  pass

class Edge():
  def __init__(self, parent, child):
    self._parent = parent
    self._child = child
    self._hash = None

  def __eq__(self, other):
    if (other is None) or \
            (not isinstance(self, other.__class__)) or \
            (not isinstance(other, self.__class__)):
      return False
    return self._parent.similar(other._parent) and self._child.similar(other._child)

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    result = hash(self._parent.hash_without_name() * self._child.hash_without_name())
    self._hash = result
    return result


class KAryTree():
  def __init__(self, **kwargs):
    self._node = kwargs.get('node')
    self._output = kwargs.get('output')
    if not isinstance(self._node, AbstractNode):
      raise TreeException("'node' type must be 'Node' but is: %s" % (type(self._node)))

    self._inputs = kwargs.get('inputs')
    if not (isinstance(self._inputs, list) or isinstance(self._inputs, AbstractNode)):
      raise TreeException("'inputs' type must be 'list' or 'NTree' but is: %s" % (type(self._inputs)))
    if isinstance(self._inputs, list):
      if not all(isinstance(x, KAryTree) for x in self._inputs):
        raise TreeException("Type of all inputs in list must be 'NTree'")
    if isinstance(self._inputs, KAryTree):
      self._inputs = list(self._inputs)
    for _in in self._inputs:
      _in._output = self

    self._min_in_override = None
    self._max_in_override = None
    self._hash = None

  def __getstate__(self):
    result = dict()
    result['node'] = self._node
    result['inputs'] = list(self._inputs)
    return result

  def __setstate__(self, state):
    self.__init__(**state)

  @property
  def min_in_override(self):
    return self._min_in_override

  @property
  def max_in_override(self):
    return self._max_in_override

  @min_in_override.setter
  def min_in_override(self, value):
    self._min_in_override = value

  @max_in_override.setter
  def max_in_override(self, value):
    self._max_in_override = value

  def __copy__(self):
    result = KAryTree(node=self._node.__copy__(),
                      inputs=list(),
                      )
    for child in [g.__copy__() for g in self._inputs]:
      result.add_child(child)
    return result

  def small_copy_replace(self, node=None):
    if isinstance(node, AbstractNode):
      return KAryTree(node=node,
                      inputs=list())
    else:
      return KAryTree(node=self._node.__copy__(),
                      inputs=list())

  def __hash__(self):
    if self._hash is not None:
      return self._hash
    stack = []
    stack.append((self, list(), list(self._inputs)))
    result = None
    while len(stack) > 0:
      current_tree, child_hashes, sub_trees = stack[-1]
      if result is not None:
        child_hashes.append(result)
        result = None
      if len(sub_trees) <= 0:
        result = hash(current_tree._node)
        fac = 1
        for child_hash in child_hashes:
          fac = fac * child_hash
        result = hash(result * 31 + fac)
        current_tree._hash = result
        stack.pop()
        continue
      subtree = sub_trees.pop()
      stack.append((subtree, list(), list(subtree._inputs)))
    return self._hash

  def __eq__(self, other):
    if (other is None) or \
            (not isinstance(self, other.__class__)) or \
            (not isinstance(other, self.__class__)) or \
            self._node != other._node or \
            len(self._inputs) != len(other._inputs):
      return False
    a, b = set(self.children), set(other.children)
    for child in a:
      if child not in b:
        return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def add_child(self, child):
    if isinstance(child, KAryTree):
      if child not in self._inputs and \
              child.parent is None:
        self._inputs.append(child)
        child._output = self

  def remove_child(self, child):
    if isinstance(child, KAryTree):
      self._inputs.remove(child)
      child._output = None

  @property
  def children(self):
    return self._inputs

  @property
  def parent(self):
    return self._output

  def depth(self):
    result = 0
    current = self
    while current._output is not None:
      current = current._output
      result += 1
    return result

  def child_depth_width(self):
    depth = 0
    width = 0
    stack = [(0, self)]
    while len(stack) > 0:
      d, current = stack.pop()
      w = len(current._inputs)
      for _in in current._inputs:
        stack.append((d + 1, _in))
      if d > depth:
        depth = d
      if w > width:
        width = w
    return depth, width

  @property
  def node(self):
    return self._node

  @property
  def trees(self):
    result = list()
    for g in self._inputs:
      result += g.trees
    result.append(self)
    return result

  @property
  def nodes(self):
    result = list()
    stack = [self]
    while len(stack) > 0:
      current = stack.pop()
      result.append((current._node, current.depth()))
      for _in in current._inputs:
        stack.append(_in)
    return result

  @property
  def edges(self):
    result = list()
    stack = [self]
    while len(stack) > 0:
      current = stack.pop()
      for t in current._inputs:
        result.append(Edge(current._node, t._node))
        stack.append(t)
    return result

  def outputsize(self):
    stack = []
    stack.append((self, list(), list(self._inputs)))
    result = None
    while len(stack) > 0:
      current_tree, input_sizes, sub_trees = stack[-1]
      if result is not None:
        input_sizes.append(result)
        result = None
      if len(sub_trees) <= 0:
        result = current_tree._node.outputsize(input_sizes)
        stack.pop()
        continue
      subtree = sub_trees.pop()
      stack.append((subtree, list(), list(subtree._inputs)))
    return result

  def min_inputsize(self, origin=None):
    stack = []
    stack.append((self, origin))
    result = None
    while result == None:
      current_tree, current_origin = stack.pop()
      if current_tree._min_in_override != None:
        result = current_tree._min_in_override
        break
      if isinstance(current_tree._node, AbstractOutputNode) or \
              isinstance(current_tree._node, AbstractInputNode):
        result = current_tree._node.min_inputsize(None)
        break
      tmp_tree = None
      for tree in current_tree._inputs:
        if tree != current_origin:
          tmp_tree = tree
          break
      if tmp_tree is not None:
        result = tmp_tree.outputsize()
        break
      stack.append(current_tree)
      stack.append((current_tree._output, current_tree))

    while len(stack) > 0:
      current_tree = stack.pop()
      result = current_tree._node.min_inputsize(result)
    return result

  def max_inputsize(self, origin=None):
    stack = []
    stack.append((self, origin))
    result = None
    while result == None:
      current_tree, current_origin = stack.pop()
      if current_tree._max_in_override != None:
        result = current_tree._max_in_override
        break
      if isinstance(current_tree._node, AbstractOutputNode) or \
              isinstance(current_tree._node, AbstractInputNode):
        result = current_tree._node.max_inputsize(None)
        break
      tmp_tree = None
      for tree in current_tree._inputs:
        if tree != current_origin:
          tmp_tree = tree
          break
      if tmp_tree is not None:
        result = tmp_tree.outputsize()
        break
      stack.append(current_tree)
      stack.append((current_tree._output, current_tree))

    while len(stack) > 0:
      current_tree = stack.pop()
      result = current_tree._node.max_inputsize(result)
    return result

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      return float('inf')
    edges_s, edges_o = self.edges, other.edges
    edges_s, edges_o = set(edges_s), set(edges_o)
    intersection_len = 2 * len(edges_s.intersection(edges_o))
    return len(edges_s) + len(edges_o) - intersection_len

  def create_nn(self, **kwargs):
    stack = []
    stack.append((self, list(), list(self._inputs)))
    result = None
    while len(stack) > 0:
      current_tree, created_graphs, sub_trees = stack[-1]
      if result is not None:
        created_graphs.append(result)
        result = None
      if len(sub_trees) <= 0:
        result = current_tree._node.create_nn(created_graphs, **kwargs)
        stack.pop()
        continue
      subtree = sub_trees.pop()
      stack.append((subtree, list(), list(subtree._inputs)))
    return result

  pass
