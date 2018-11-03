try:
    from queue import Queue
except Exception as e:
    from Queue import Queue

from threading import Thread

class TaskQueue(Queue):

    def __init__(self, maxsize=0, num_workers=1, debug=False):
        Queue.__init__(self, maxsize=maxsize)
        self.num_workers = num_workers
        self.debug = debug
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

        if self.debug: 
            print('Added task, len = ', self.qsize())
            
    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            tupl = self.get()
            item, args, kwargs = self.get()
            item(*args, **kwargs)  
            self.task_done()



