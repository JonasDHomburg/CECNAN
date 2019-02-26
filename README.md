# Constraint Exploration of Convolutional Network Architectures with Neuroevolution

## Abstract
The effort spent on adapting existing networks to new applications has motivated the automated architecture search. Network structures discovered with evolutionary or other search algorithms have surpassed hand-crafted image classifiers in terms of accuracy. However, these approaches do not constrain certain characteristics like network size or numbers of parameters, which leads to unnecessary computational effort. Thus, this work shows that generational and evolutionary algorithms can be used for a constrained exploration of convolutional network architectures to create various networks which represent a trade-off in applied constraints.

##### Example:
```python
# import tensorflow as tf
# depends on tensorflow for network evaluation
from ea.initializationStrategies import RandomGeneration
from ea.tf_individuals import MNISTNTreeIndividual
from ea.replacement_schemes import NElitism
from ea.GenerationManager import GenerationManager
from ea.selection_strategies import RankingSelectionV1 # exponential ranking
from ea.stopping_criteria import SelectionStopping
from ea.nea_core import NEACore

import pickle

def run():
    initStrat = RandomGeneration(size=10,
                                 individual_class=MNIST,
                                 individual_config={'parameters':True,
                                                    'parameters_minimum':10**5,
                                                    'parameters_maximum':10**7})
    replacementScheme = NElitism(n=2)
    genMan = GenerationManager(initializationStrategy=initStrat,
                               replacement=replacementScheme,
                               mem_type=GenerationManager.MEM_SQLITE3,
                               # sqlite3 or mySql should be used for a complete logfile, to get only some statistics: MEM_STATS
                               gen_log='/path/to/log/file.db3'
                               )

    eh = LocalEH(
      train_data=pickle.load(open('/path/to/mnist_ga_train.p', 'rb')),
      valid_data=pickle.load(open('/path/to/mnist_ga_valid.p', 'rb')),
      test_data=pickle.load(open('/path/to/mnist_ga_test.p', 'rb')),
      # log_path='/path/to/training/log/files' # optional
    )

    selectStrat = RankingSelectionV1(limit=4,
                                     evaluation_helper=eh)
    saturationStrat = SelectionStopping(
                 # stops after 20 generations without fitness improvement
                 patience=20,
                 generationManager=genMan)
    core = NEACore(
        generationManager=genMan,
        saturationStrategy=saturationStrat,
        selection=selectStrat,

        crossover_size=10, # 2 new networks per crossing pair creates 12 new networks per generation -> uniform selection of 10
        mutation_rate=.4
    )
    core.run()
    pass

from ea.GenerationManager import GenerationManager, PlaybackGenManager
from plotting.plot_ea import generation_history

def plot():
    genMan = PlaybackGenManager(filename='/path/to/log/file.db3',
                                mem_type=GenerationManager.MEM_SQLITE3,
                                table_prefix='_00' # automatically incremented number if the same file is used
                                )
    generation_history(genManager=genMan,
                       filename='/path/to/plot', # format: pdf
                       gen_per_plot=10, # one file for 10 generations
                       # using a color map with
                       NUM_COLORS=15 # colors
                       )
    pass

if __name__ == '__main__':
    run()
```
