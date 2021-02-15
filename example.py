"""
	This script is supplied as an example of how to use utils.py
"""

import json
from utils import *

_this_dir, _this_file = os.path.split(os.path.abspath(__file__))

a = {
	"key1": list(range(10)),
	"key2": list(range(10,20))
}

def main(**kwargs):
	return build_pipe([
			_func1,
			_func2,
			build_branches([
					_branch1,
					_branch2
			]),
			_print
	], decorators=[timing, escape])(a.update(kwargs) or a)


def _func1(a):
	return a.update({'key1': [x%2 for x in a['key1']]}) or a


def _func2(a):
	return a.update({'key3': list(zip(a['key1'], a['key2'])) }) or a


def _branch1(a):
	with open('tempfile.json', 'w') as f:
		json.dump(a,f)

	del a['key3']

	return a


def _branch2(a):
	return a.update({"key4": list(range(20,30))}) or a


def _print(a):
	print(a)
	return a

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--key1', nargs='*', default=argparse.SUPPRESS)
	parser.add_argument('--key2',nargs='*', default=argparse.SUPPRESS)

	args = parser.parse_args()

	main(**vars(args))

	print('Done')
