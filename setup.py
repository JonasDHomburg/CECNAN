from setuptools import setup
setup(name='ea',
      version='0.1',
      description='Collection of evolutions algorithms and plotting functions.',
      url='https://git.ks.techfak.uni-bielefeld.de/projects/ma-jhomburg.git',
      author='Jonas Dominik Homburg',
      author_email='jhomburg@techfak.de',
      license='MIT',
      packages=[
        'ea',
        'ea.representations',
        'plotting',
        'compute',
      ],
      # packages = find_packages(),
      install_requires=[
        'numpy',
        'matplotlib',
        'bokeh',
        'sklearn',
        'mayavi',
        'pathos',
        'paho-mqtt',
      ],
      extras_require={
        'telegram': ['python-telegram-bot'],
        'tensorflow-gpu': ['tensorflow-gpu'],
        'tensorfow': ['tensorflow'],
      },
      zip_safe=False)


