try:
    from queue import Queue
except Exception as e:
    from Queue import Queue

from threading import Thread
import time

class TaskQueue(Queue):

    def __init__(self, maxsize=0, num_workers=1, verbosity=0):
        Queue.__init__(self, maxsize=maxsize)
        self.num_workers = num_workers
        self.verbosity = verbosity
        self.threads = []

        self.start_workers()

    def add_task(self, task, *args, **kwargs):

        if self.num_workers > 0:
            args = args or ()
            kwargs = kwargs or {}
            self.put((task, args, kwargs))

            if (self.verbosity > 0 and self.qsize() == self.num_workers) or self.verbosity > 1: 
                print('Added task, len = ', self.qsize())
        else:
            task(*args, **kwargs)  

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            # t.daemon = True
            t.start()

            self.threads.append(t)

    def worker(self):
        while True:
            item = self.get()
            if item is None:
                break

            fn, args, kwargs = item
            fn(*args, **kwargs)  
            self.task_done()

    def stop_(self):

        self.join()

        if self.num_workers > 1:
            for i in range(self.num_workers):
                self.put(None)
            for t in self.threads:
                t.join()



