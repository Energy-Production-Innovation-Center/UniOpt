from uniopt.logger import Logger


class StoppingCriteria:
    def __init__(self):
        self.logger = Logger()

    def run(self) -> bool:
        self._logger.log(name_file=__name__, texto=f"Running {__name__}")

        return False
