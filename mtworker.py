import functools
import threading
import queue
import time


class DaemonThread(threading.Thread):
    def __init__(self, func, *args):
        super().__init__(daemon=True)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


class DaemonWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.daemon = DaemonThread(self.daemon_main)
        self.daemon.start()

    def daemon_main(self):
        while True:
            func = self.queue.get()
            func.retval = func()
            self.queue.task_done()

    def launch(self, func):
        self.queue.put(func)
        self.queue.join()
        return func.retval


class DaemonModule:
    def __init__(self, getmodule):
        self.worker = DaemonWorker()

        @self.worker.launch
        def _():
            self.module = getmodule()

    def __getattr__(self, name):
        func = getattr(self.module, name)
        if not callable(func):
            return func

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            @self.worker.launch
            def retval():
                return func(*args, **kwargs)

            return retval

        return wrapped
