import logging


class LoggerSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=LoggerSingleton):
    def __init__(self, log_console=None, log_file=None, level=logging.INFO):
        self.log_console = log_console
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)

        self.logger.setLevel(level)
        # TODO Adicionar o path
        self.log_file = "path"

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                "%d-%m-%Y %H:%M:%S",
            )

            if log_file:
                file_handler = logging.FileHandler(self.log_file, mode="a")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            if log_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_critical(self, message):
        self.logger.critical(message)
