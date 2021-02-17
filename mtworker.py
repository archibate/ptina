import threading
import queue
import time


q = queue.Queue()


class DaemonThread(threading.Thread):
    def __init__(self, func, *args):
        super().__init__(daemon=True)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


@DaemonThread
def func():
    while True:
        func = q.get()
        func()
        q.task_done()


func.start()

@q.put
def _():
    print('What the fuck')

q.join()
