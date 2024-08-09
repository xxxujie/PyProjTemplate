import argparse

from xu_pytools import configs, loggers
from xu_pytools.settings import tool_settings


def main():
    logger.info("run main")
    logger.info(configs.sample_config.USER_FULL_NAME)
    logger.info(configs.sample_config.TYPE)


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    args = parser.parse_args()
    if args.config_dir != "":
        tool_settings.CONFIG_DIRS.insert(0, args.config_dir)
    logger = loggers.get_logger()
    main()
