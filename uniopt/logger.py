import logging
from typing import Any, override


class LoggerSingleton(type):
    _instance = None  # pyright: ignore [reportUnannotatedClassAttribute]

    @override
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Logger(metaclass=LoggerSingleton):
    def __init__(self, log_console: bool = True, log_file: bool = False, level: int = logging.INFO):
        self.log_console: bool = log_console
        self.log_file: bool = log_file
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.logger.setLevel(level)
        # TODO Adicionar o path
        self.log_path: str = "path"

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                "%d-%m-%Y %H:%M:%S",
            )

            if log_file:
                file_handler = logging.FileHandler(self.log_path, mode="a")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            if log_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

    def log_debug(self, message: str):
        self.logger.debug(message)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def log_critical(self, message: str):
        self.logger.critical(message)
