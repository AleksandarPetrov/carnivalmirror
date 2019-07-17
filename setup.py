from setuptools import setup

version='0.5.3'

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
      test_suite='carnivalmirror.tests'
      )
