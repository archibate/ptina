import sys
import importlib
import time


found_map = set()
depth = 0


class MetaPathFinder:
    def find_module(self, fullname, path=None):
        if not (fullname == 'tina' or fullname.startswith('tina.')):
            return
        if fullname in found_map:
            return
        found_map.add(fullname)

        return MetaPathLoader()


def get_tick():
    curr_time = time.time()
    if not hasattr(get_tick, 't0'):
        get_tick.t0 = curr_time
    return curr_time - get_tick.t0


class MetaPathLoader:
    def load_module(self, fullname):
        global depth
        print('[{:.3f}] {} {}'.format(get_tick(), depth * '|' + '+', fullname))
        depth += 1
        module = importlib.import_module(fullname)
        depth -= 1
        print('[{:.3f}] {}'.format(get_tick(), depth * '|' + '^'))
        sys.modules[fullname] = module
        return module


sys.meta_path.insert(0, MetaPathFinder())


if __name__ == '__main__':
    import runpy
    runpy.run_path(sys.argv[1])
