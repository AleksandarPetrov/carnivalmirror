from setuptools import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

version='0.5.0'

setup(name='carnivalmirror',
      version=version,
      description='A tool to handle and simulate pinhole camera miscalibrations',
      url='https://github.com/AleksandarPetrov/carnivalmirror',
      author='Aleksandar Petrov',
      author_email='alpetrov@ethz.ch',
      license='MIT',
      packages=['carnivalmirror'],
      zip_safe=False,
      install_requires=['numpy>=1.14.0', 'opencv-python>=4.1.0.0', 'matplotlib>=2.2.2'],
      package_data={
            'carnivalmirror': ['test_image_*.png'],
      },
      test_suite='carnivalmirror.tests',
      command_options={
        'build_sphinx': {
            'project': 'CarnivalMirror',
            'copyright': ('setup.py', '2019, Aleksandar Petrov'),
            'author': ('setup.py', 'Aleksandar Petrov'),
            'version': ('setup.py', version),
            'release': ('setup.py', version),
            'source_dir': ('setup.py', 'docs')}}
      )
