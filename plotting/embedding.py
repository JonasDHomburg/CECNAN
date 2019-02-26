from ea.GenerationManager import *
from ea.dna import *
from sklearn import manifold
import pickle
from multiprocessing import Pool


def calc(_in):
  return _in[0] - _in[1]

def embed_list(dnas: "list of baseDNA", n_components: int = 2, pool_s=8):
  p = Pool(pool_s)
  distance = p.map(calc,[(d0, d1) for d0 in dnas for d1 in dnas])
  distance = np.asarray(distance)
  distance = distance.reshape((len(dnas), len(dnas)))
  # horizontal, vertical = np.meshgrid(dnas, dnas)
  # distance = horizontal - vertical
  tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, n_iter=3000, metric='precomputed')
  return tsne.fit_transform(distance)


def mnist625_embedding(filename, predictions):
  conv_filter = [3, 5, 7, 9, 11]
  pool_filter = [1, 2, 3, 4, 5]
  dnas = list()
  for conv1 in conv_filter:
    for pool1 in pool_filter:
      for conv2 in conv_filter:
        for pool2 in pool_filter:
          dnas.append(MNIST625DNA(representation=[conv1, pool1, conv2, pool2]))
  Y = embed_list(dnas)
  fitness_table = dict()
  tmp_ = pickle.load(open(predictions, 'rb'))
  for key, value in tmp_.items():
    pred = value['prediction']
    labels = value['labels']
    w_acc = [p_[l_] if p_[l_] > .5 else 0 for p_, l_ in zip(pred, labels)]
    # w_acc = [p_[l_] for p_, l_ in zip(pred, labels)]
    # w_acc = [np.argmax(p_) == l_ for p_, l_ in zip(pred, labels)]
    w_acc = sum(w_acc) / len(w_acc)
    fitness_table[key] = w_acc

  fitness = [fitness_table[dna] for dna in dnas]
  result = dict()
  for dna, y, f in zip(dnas, Y, fitness):
    result[dna] = (y, f)
  pickle.dump(result, open(filename, 'wb'), -1)
  pass


def embed_generation(generationManager: PlaybackGenManager, file: str):
  dna_fitness = dict()
  for state in generationManager:
    for ind in state.descendants:
      dna_fitness[ind.dna] = ind.f
    for ind in state.mutation:
      dna_fitness[ind.dna] = ind.f
    for ind in state.crossover:
      dna_fitness[ind.dna] = ind.f
    for ind in state.generation:
      dna_fitness[ind.dna] = ind.f

  dnas = list(dna_fitness.keys())
  coordinates = embed_list(dnas)
  result = dict()
  for dna, y, f in zip(dnas, coordinates, [dna_fitness[ind] for ind in dnas]):
    result[dna] = (y, f)
  if file is not None:
    pickle.dump(result, open(file, 'wb'), -1)
  return result


def embed_set_of_generations(generationManagers: list, file: str):
  dna_fitness = dict()
  for generationManager in generationManagers:
    for state in generationManager:
      for ind in state.descendants:
        if dna_fitness.get(ind.dna) is None:
          dna_fitness[ind.dna] = ind.f
      for ind in state.mutation:
        if dna_fitness.get(ind.dna) is None:
          dna_fitness[ind.dna] = ind.f
      for ind in state.crossover:
        if dna_fitness.get(ind.dna) is None:
          dna_fitness[ind.dna] = ind.f
      for ind in state.generation:
        if dna_fitness.get(ind.dna) is None:
          dna_fitness[ind.dna] = ind.f

  dnas = list(dna_fitness.keys())
  coordinates = embed_list(dnas)
  result = dict()
  for dna, y, f in zip(dnas, coordinates, [dna_fitness[ind] for ind in dnas]):
    result[dna] = (y, f)
  pickle.dump(result, open(file, 'wb'), -1)
  pass
