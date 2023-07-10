import logging
from typing import Optional, Union

import netsquid as ns


class SimTimeFilter(logging.Filter):
    def filter(self, record):
        record.simtime = ns.sim_time()
        return True


class LogManager:
    STACK_LOGGER = "Stack"
    _STACK_LOGGER_HAS_BEEN_SETUP = False
    TASK_LOGGER = "Scheduler"
    _TASK_LOGGER_HAS_BEEN_SETUP = False

    @classmethod
    def _setup_stack_logger(cls) -> None:
        logger = logging.getLogger(cls.STACK_LOGGER)
        formatter = logging.Formatter(
            "%(levelname)s:%(simtime)s ns:%(name)s:%(message)s"
        )
        syslog = logging.StreamHandler()
        syslog.setFormatter(formatter)
        syslog.addFilter(SimTimeFilter())
        logger.addHandler(syslog)
        logger.propagate = False
        cls._STACK_LOGGER_HAS_BEEN_SETUP = True

    @classmethod
    def _setup_task_logger(cls) -> None:
        logger = logging.getLogger(cls.TASK_LOGGER)
        formatter = logging.Formatter(
            "%(levelname)s:%(simtime)s ns:%(name)s:%(message)s"
        )
        syslog = logging.StreamHandler()
        syslog.setFormatter(formatter)
        syslog.addFilter(SimTimeFilter())
        logger.addHandler(syslog)
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)
        cls._TASK_LOGGER_HAS_BEEN_SETUP = True

    @classmethod
    def get_stack_logger(cls, sub_logger: Optional[str] = None) -> logging.Logger:
        if not cls._STACK_LOGGER_HAS_BEEN_SETUP:
            cls._setup_stack_logger()
        logger = logging.getLogger(cls.STACK_LOGGER)
        if sub_logger is None:
            return logger
        else:
            return logger.getChild(sub_logger)

    @classmethod
    def get_task_logger(cls, sub_logger: Optional[str] = None) -> logging.Logger:
        if not cls._TASK_LOGGER_HAS_BEEN_SETUP:
            cls._setup_task_logger()
        logger = logging.getLogger(cls.TASK_LOGGER)
        if sub_logger is None:
            return logger
        else:
            return logger.getChild(sub_logger)

    @classmethod
    def set_log_level(cls, level: Union[int, str]) -> None:
        logger = cls.get_stack_logger()
        logger.setLevel(level)

    @classmethod
    def get_log_level(cls) -> int:
        return cls.get_stack_logger().level

    @classmethod
    def log_to_file(cls, path: str) -> None:
        file_handler = logging.FileHandler(path, mode="w")
        formatter = logging.Formatter(
            "%(levelname)s:%(simtime)s ns:%(name)s:%(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SimTimeFilter())
        cls.get_stack_logger().addHandler(file_handler)

    @classmethod
    def log_tasks_to_file(cls, path: str) -> None:
        file_handler = logging.FileHandler(path, mode="w")
        formatter = logging.Formatter(
            "%(levelname)s:%(simtime)s ns:%(name)s:%(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(SimTimeFilter())
        cls.get_task_logger().addHandler(file_handler)

    @classmethod
    def enable_task_logger(cls, enable: bool) -> None:
        if enable:
            cls.get_task_logger().setLevel(logging.INFO)
        else:
            cls.get_task_logger().setLevel(logging.CRITICAL + 1)
