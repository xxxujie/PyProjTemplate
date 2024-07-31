from functools import wraps


def singleton(cls):
    instance = {}

    @wraps(cls)
    def _singleton(*args, **kargs):
        # 如果没有 cls 这个类，则创建，并且将这个 cls 所创建的实例保存在一个字典中
        if cls not in instance:
            instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return _singleton


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):

        @wraps(self._cls)
        def wrapped_function():
            if self._cls not in self._instance:
                self._instance[self._cls] = self._cls(*args, **kwargs)
            return self._instance[self._cls]

        return wrapped_function
