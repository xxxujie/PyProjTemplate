import os
import yaml
import json

from settings import CONFIG_DIRS

# import settings


class _ConfigHandler:
    def __init__(self, config_dir):
        self.reload(config_dir)

    def reload(self, config_dir):
        loader = self._load_configs(config_dir)
        # 根据配置文件自动添加属性
        for name, config in loader:
            setattr(self, name, config)

    def _load_configs(self, config_dir: str):
        """加载配置文件"""
        for fname in os.listdir(config_dir):
            with open(os.path.join(config_dir, fname), "r", encoding="utf-8") as f:
                ext = os.path.splitext(fname)[1]
                if ext == ".yaml":
                    config = yaml.load(f, yaml.FullLoader)
                elif ext == ".json":
                    config = json.load(f)
                else:
                    raise ValueError("不支持的配置文件格式：" + ext)
                yield os.path.splitext(fname)[0], dict(config)


def _load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        ext = os.path.splitext(config_path)[1]
        if ext == ".yaml":
            config = yaml.load(f, yaml.FullLoader)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError("不支持的配置文件格式：" + ext)
        return config


class BaseConfig:
    def __init__(self, config_path: str):
        self._config = _load_config(config_path)

    def get_property(self, name):
        return self._config.get(name)


# with open("./configs/train_conf.yaml", "r", encoding="utf-8") as f:
#     print(f.read())
# config_helper = _ConfigHelper(settings.CONFIG_DIR)
