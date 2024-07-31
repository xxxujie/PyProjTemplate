import os
import yaml
import json

from common.utils.singleton import Singleton, singleton
from settings import CONFIG_DIRS
import settings


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


def _find_config_path(file_name: str):
    """按照 settings.CONFIG_DIRS 列表中的目录顺序查找名为 file_name（需要带扩展名）的配置文件"""

    for config_dir in settings.CONFIG_DIRS:
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 或者命令行参数")


class _BaseConfig:
    """配置基类，不同的配置继承于它并实现对应属性，具体见 _SampleConfig"""

    def __init__(self, config_path: str):
        self._config = _load_config(config_path)

    def _get_property(self, key):
        return self._config.get(key)


class _SampleConfig(_BaseConfig):
    """一个配置类的例子，继承于 _BaseConfig，通过 self._get_property(key) 方法获取对应值，以实现属性"""

    @property
    def id(self):
        return self._get_property("id")

    @property
    def name(self):
        return self._get_property("name")


# 留给外部调用的单例，初始化需要指定对应配置文件的地址
sample_config = _SampleConfig(_find_config_path("sample_conf.yaml"))
