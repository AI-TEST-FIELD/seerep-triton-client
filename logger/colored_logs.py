import logging
from colorlog import ColoredFormatter

class Client_logger():
    def __init__(self, name='client', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = ColoredFormatter(
            "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red',
            }
        )
        self.ch = logging.StreamHandler()
        self.ch.setLevel(level)
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.propagate = False

    def get_logger(self):
        return self.logger