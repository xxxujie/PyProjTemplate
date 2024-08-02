# 程序入口
# ! 不要在主函数入口之前添加任何执行代码
import argparse
import settings

from common.utils import configs, loggers


def main():
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    args = parser.parse_args()
    if args.config_dir != "":
        settings.USER_CONFIG_DIRS.insert(0, args.config_dir)


if __name__ == "__main__":
    main()

    # 测试代码
    print(configs.sample_config.USER_ID)
    print(configs.sample_config.USER_FULL_NAME)
    print(configs.sample_config.TYPE)
    logger = loggers.get_logger()
    logger.info("run main")
