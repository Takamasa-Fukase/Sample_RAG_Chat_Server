import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

def _getenv(key: str):
    env = os.getenv(key)
    if env is None:
        return env

    # 空文字列の場合はNoneを返す
    if env.strip() == "":
        os.environ.pop(key)
        return None

    return env

class Env:
    APP_URL = _getenv("APP_URL")
    OPENAI_API_KEY = _getenv("OPENAI_API_KEY")
    SERPER_API_KEY = _getenv("SERPER_API_KEY")
    SERPER_API_KEY = _getenv("SERPER_API_KEY")
    DATABASE_HOST = _getenv("DATABASE_HOST")
    DATABASE_PORT = _getenv("DATABASE_PORT")
    DATABASE_NAME = _getenv("DATABASE_NAME")
    DATABASE_USER = _getenv("DATABASE_USER")
    DATABASE_PASSWORD = _getenv("DATABASE_PASSWORD")