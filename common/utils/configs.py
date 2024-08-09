from .config_loader import load_config, find_config_path


class _Config:
    """所有配置类的基类，实际上是一个字典的包装类，具体见 _SampleConfig"""

    def __init__(self, config_fname: str):
        self._config = load_config(find_config_path(config_fname))

    def _get(self, *args):
        item = ""
        value = self._config
        for key in args:
            item += key
            if type(value) is not dict:
                raise ValueError(f"尝试从配置项 {item} 访问不存在的键 {key}")
            value = value.get(key)
            if value is None:
                break
            item += "."

        return value


class _SampleConfig(_Config):
    """一个配置类的例子，继承于 _BaseConfig，具体实现各项属性"""

    @property
    def USER_ID(self):
        return self._get("user", "id")
        return self._get("user", "id")

    @property
    def USER_FIRST_NAME(self):
        return self._get("user", "name", "first_name")
        return self._get("user", "name", "first_name")

    @property
    def USER_LAST_NAME(self):
        return self._get("user", "name", "last_name")
        return self._get("user", "name", "last_name")

    @property
    def USER_FULL_NAME(self):
        if self.USER_FIRST_NAME is None:
            return self.USER_LAST_NAME
        elif self.USER_LAST_NAME is None:
            return self.USER_FIRST_NAME
        else:
            return self.USER_FIRST_NAME + " " + self.USER_LAST_NAME
        if self.USER_FIRST_NAME is None:
            return self.USER_LAST_NAME
        elif self.USER_LAST_NAME is None:
            return self.USER_FIRST_NAME
        else:
            return self.USER_FIRST_NAME + " " + self.USER_LAST_NAME

    @property
    def TYPE(self):
        type = self._get("type")
        type = self._get("type")
        if not hasattr(self, "_type"):
            if type == "TYPE1":
                tmp = 1
            elif type == "TYPE2":
                tmp = 2
            elif type == "TYPE3":
                tmp = 3
            else:
                raise ValueError("Invalid value of 'type' in config file")
            setattr(self, "_type", tmp)

        return getattr(self, "_type")


# 留给外部调用的单例，初始化需要指定对应配置文件带扩展的全名
sample_config = _SampleConfig("sample_conf.yaml")
