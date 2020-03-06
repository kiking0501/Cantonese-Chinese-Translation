import pickle
import os


def save_cache(objs, cache_path):
    pickle.dump(objs, open(cache_path, 'wb'))


def load_cache(cache_path):
    return pickle.load(open(cache_path, 'rb'))


def CACHE(name, dir_path):
    cache_path = os.path.join(dir_path, name)

    def _INNER_CACHE(func):
        def _INNER2_CACHE(*args, **kwargs):
            if os.path.exists(cache_path):
                return load_cache(cache_path)
            return_objs = func(*args, **kwargs)
            save_cache(return_objs, cache_path)
            return return_objs
        return _INNER2_CACHE
    return _INNER_CACHE
