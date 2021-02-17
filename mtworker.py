from threading import Thread
import time


t = Thread(target=self.__main, daemon=True)


@MTWorker
def worker():
    def hello():
        pass

    hello.x = 1
    return hello.x


time.sleep(1)
print(worker.x)
