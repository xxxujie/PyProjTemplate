from os.path import dirname, abspath

# root directory
APP_DIR = dirname(dirname(dirname(abspath(__file__))))

# where user's configuration files locate (finding by order)
CONFIG_DIRS = [
    APP_DIR,
    f"{APP_DIR}/common/configs",
    f"{APP_DIR}/common/configs/default",
]

# where logs locate
LOG_DIR = f"{APP_DIR}/logs"

# where data locate
DATA_DIR = f"{APP_DIR}/resources/data"
