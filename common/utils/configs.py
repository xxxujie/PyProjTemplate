import os
import yaml
import json
import settings


class _ConfigHandler:
    """(Deprecated)
    读取所有配置文件，自动添加字典类型的配置实例作为属性
    """

    def __init__(self, config_dir):
        self.reload(config_dir)

    def reload(self, config_dir):
        loader = self._load_configs(config_dir)
        # 根据配置文件自动添加字典属性
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
        # print(f"Finding {file_name} in {config_dir}")
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 或者命令行参数")


class _BaseConfig:
    """配置类的基类，实际上是一个字典的包装类，具体见 _SampleConfig"""

    def __init__(self, config_path: str):
        self._config = _load_config(config_path)

    def get(self, key):
        return self._config.get(key)


class _SampleConfig(_BaseConfig):
    """一个配置类的例子，继承于 _BaseConfig，具体实现各项属性"""

    @property
    def USER_ID(self):
        return self.get("user").get("id")

    @property
    def USER_FIRST_NAME(self):
        return self.get("user").get("name").get("first_name")

    @property
    def USER_LAST_NAME(self):
        return self.get("user").get("name").get("last_name")

    @property
    def USER_FULL_NAME(self):
        return self.USER_FIRST_NAME + " " + self.USER_LAST_NAME

    @property
    def TYPE(self):
        type = self.get("type")
        if not hasattr(self, "_type"):
            match type:
                case "TYPE1":
                    tmp = 1
                case "TYPE2":
                    tmp = 2
                case "TYPE3":
                    tmp = 3
                case _:
                    raise ValueError("Invalid value of 'type' in config file")
            setattr(self, "_type", tmp)

        return getattr(self, "_type")


# 留给外部调用的单例，初始化需要指定对应配置文件的地址
sample_config = _SampleConfig(_find_config_path("sample_conf.yaml"))
