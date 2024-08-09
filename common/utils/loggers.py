import os
import logging
import logging.config

from . import globalenv
from .config_loader import load_config, find_config_path


class _Logger:
    def __init__(self, name, log_dir=globalenv.LOG_DIR):
        # 保证目录存在并能够读写
        if not os.path.exists(log_dir):
            os.mkdir(log_dir, mode=0o766)
        os.chmod(log_dir, 0o766)

        self._log_dir = log_dir
        self._name = name
        self.logger = self._init_logger()

    def _init_logger(self):
        config = load_config(find_config_path("logger_conf.yaml"))
        fname = config["handlers"]["info_file"]["filename"]
        config["handlers"]["info_file"]["filename"] = os.path.join(self._log_dir, fname)
        fname = config["handlers"]["error_file"]["filename"]
        config["handlers"]["error_file"]["filename"] = os.path.join(self._log_dir, fname)
        logging.config.dictConfig(config)
        logger = logging.getLogger(self._name)
        return logger


def get_logger(name="default"):
    logger = _Logger(name).logger
    return logger
