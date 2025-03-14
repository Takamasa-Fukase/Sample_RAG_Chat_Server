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
    OPENAI_API_KEY = _getenv("OPENAI_API_KEY")
    SERPER_API_KEY = _getenv("SERPER_API_KEY")