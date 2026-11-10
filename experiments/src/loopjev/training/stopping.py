import signal


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for name in (signal.SIGTERM, signal.SIGINT):
            self.previous[name] = signal.signal(name, self.handle)
        return self

    def handle(self, number, frame):
        self.requested = True
        print('Stop requested; saving after the current optimizer update', flush=True)

    def __exit__(self, exception_type, exception, traceback):
        for name, handler in self.previous.items():
            signal.signal(name, handler)
