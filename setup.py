import sys
major, minor, micro, releaselevel, serial = sys.version_info
if not (major == 2 and minor >= 5):
	print("Python >=2.5 is required to use this module.")
	sys.exit(1)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path

setup_dir = os.path.split(os.path.abspath(__file__))[0]
DOCUMENTATION = open(os.path.join(setup_dir, 'README.rst')).read()

tigger_path = os.path.join(setup_dir, 'tigger', '__init__.py')
globals_dict = {}
execfile(tigger_path, globals_dict)
VERSION = '.'.join([str(x) for x in globals_dict['VERSION']])

dependencies = ['mako', 'numpy']

setup(
	name='tigger',
	packages=['tigger'],
	provides=['tigger'],
	requires=dependencies,
	install_requires=dependencies,
	package_data={'tigger': ['*.mako']},
	version=VERSION,
	author='Bogdan Opanchuk',
	author_email='bogdan@opanchuk.net',
	url='http://github.com/Manticore/pyfft',
	description='GPGPU algorithms for PyCuda and PyOpenCL',
	long_description=DOCUMENTATION,
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2',
		'Topic :: Scientific/Engineering :: Mathematics'
	]
)
