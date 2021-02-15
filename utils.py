"""
  Helper functions used across scripts.
"""

import os, sys, inspect, pickle, shutil, traceback
import multiprocessing as mp
from functools import wraps, reduce
from timeit import default_timer

_this_dir, _this_file = os.path.split(os.path.abspath(__file__))


class PipelineIO(object):
    def __init__(self, pickle_filename):
        self.pickle_filename = pickle_filename

    def __enter__(self):
        with open(self.pickle_filename, 'r') as f:
            self.a = pickle.load(f)
        return self.a

    def __exit__(self):
        with open(self.pickle_filename, 'w') as f:
            pickle.dump(self.a, f)


def build_pipe(fs, decorators=None):
    """
      builds and returns a data pipeline from a list of functions (fs)
      only supports functions with single positional arguments
    """
    if decorators:
        fs = reduce(lambda a, dec: map(dec, a), decorators, fs)
    return lambda x: reduce(lambda a, f: f(a), fs, x)


def build_branches(fs):
    """
        Executes all functions against the same input and merges results.
        All results must be dicts and no two branches can update the same key.
    """
    q = mp.Queue()

    def _merge_branches(branch_results):
        for k in set.union(*list(map(set, branch_results))):
            branch_values = []
            for b in branch_results:
                elem = b.get(k)
                if (elem not in branch_values) and (elem is not None):
                    branch_values.append(elem)
            assert len(branch_values) <= 1, f"multiple branches updated the same key, {k}, in the input"
        return reduce(lambda a, d: a.update(d) or a, branch_results, {})

    def _execute_branches(x):
        print(f"Executing {[f.__name__ for f in fs]} in parallel...")
        procs = [mp.Process(target=queue_results, args=(q, f, x)) for f in fs]
        for p in procs: p.start()
        for p in procs: p.join()
        return _merge_branches([q.get() for p in procs])

    return _execute_branches


def queue_results(q, f, x):
    """
        Places results of f(x) on the multiprocessing queue, q.
    """
    q.put(f(x))


def escape(f):
    """
      fall-through decorator for pipelines
      only supports functions with single positional arguments
    """

    @wraps(f)
    def wrapper(args):
        if globals().get('pipeline_error', False):
            return args
        else:
            try:
                return f(args)
            except:
                globals().update({'pipeline_error': f'Error occurred during execution of function {f.__name__}'})
                traceback.print_exc()
                return args

    return wrapper


def args_type_checking(f):
    """
      typechecking decorator, requires decorated function to leverage type hints from typing module
      ONLY works with positional arguments
    """
    signature = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        for arg, param in zip(locals()['args'], signature.parameters.values()):
            if param.annotation != inspect.Parameter.empty:
                if type(param.annotation) == list:
                    assert type(
                            arg) in param.annotation, f"Argument {param.name, type(arg)} does not have the expected type: {param.annotation}"
                else:
                    assert type(
                            arg) == param.annotation, f"Argument {param.name, type(arg)} does not have the expected type: {param.annotation}"
        return f(*args, **kwargs)

    return wrapper


def timing(f):
    """
      Timing decorator
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = default_timer()
        result = f(*args, **kwargs)
        t = default_timer() - ts
        print(f'Function "{f.__name__}" Elapsed Time: {t}')
        return result

    return wrapper


def time_step(step_max):
    """
      Timing decorator for each step in a training procedure
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            ts = default_timer()
            result = f(*args, **kwargs)
            t = default_timer() - ts
            print(f'{result}, Step Time: {t}, Time Remaining: {t * (step_max - wrapper.calls)}')
            return result

        wrapper.calls = 0

        return wrapper

    return decorator


def workdir(d):
    """
        Decorator for changing the working directory and path temporarily
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            entry_workdir = sys.path[0]
            os.chdir(d)
            sys.path[0] = d
            print(f"cwd: {os.getcwd()}")
            print(f"sys.path: {sys.path}")
            result = f(*args, **kwargs)
            os.chdir(entry_workdir)
            sys.path[0] = entry_workdir
            return result

        return wrapper

    return decorator


def multiworker(f):
    """
      Multiworker training strategy: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#overview
    """
    import tensorflow as tf
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    @wraps(f)
    def wrapper(*args, **kwargs):
        with strategy.scope():
            print('Compiling model for distributed training...')
            print(f'Cluster targeting -- {strategy.cluster_resolver.cluster_spec()}')
            result = f(*args, **kwargs)
        return result

    return wrapper


def download_binary_file(remote_loc, destination_loc):
    """
      Supports rapid download of large binary files.
    """
    import requests
    with requests.get(remote_loc, stream=True) as r:
        with open(destination_loc, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
