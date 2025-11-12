import threading

class WatchThread :
    def __init__(self, target, name) :
        self.target = target
        self.name = name
        self.thread = threading.Thread(target=self.target, name=self.name)
        self.thread.start()
        self.logger = None
    
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('watch thread class object activate')

    def restart_if_out(self) :
        if self.thread.is_alive():
            return 1
        print(f'[ERROR] {self.thread.name} is out')
        self.logger.error(f'[ERROR] {self.thread.name} is out')
        self.thread = threading.Thread(target=self.target, name=self.name)
        self.thread.start()
        print(f'{self.thread.name} restart!')
        self.logger.info(f'{self.thread.name} restart!')
        return 0
    
    def thread_end(self) :
        self.thread.join()

    def __del__(self) :
        print('\nwatchThread class desturctor')
        if self.thread :
            self.thread.join()