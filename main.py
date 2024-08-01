import argparse
import settings

from common.utils import configs, loggers


def main(args):
    if args.config_dir != "":
        settings.CONFIG_DIRS.insert(0, args.config_dir)
    print(configs.sample_config.USER_FULL_NAME)
    print(configs.sample_config.TYPE)
    logger = loggers.get_logger()
    logger.error("test")


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./resources/configs")

    args = parser.parse_args()

    main(args)
