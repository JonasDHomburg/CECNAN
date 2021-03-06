from setuptools import setup
setup(name='ea',
      version='1.0',
      description='Constraint Exploration of Convolutional Network Architectures with Neuroevolution.',
      url='https://github.com/JonasDHomburg/CECNAN.git',
      author='Jonas Dominik Homburg',
      author_email='jonasdhomburg@gmail.com',
      license='GPL-2.0',
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
