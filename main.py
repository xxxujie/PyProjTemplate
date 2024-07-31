import argparse

from common.utils.configs import sample_config


def main(args):
    # train_config = configs.TrainConfig(f"{args.config_dir}/train_conf.yaml")
    print(sample_config.id)
    print(sample_config.name)


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./resources/configs")

    args = parser.parse_args()

    main(args)
