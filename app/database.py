from env import Env
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base

_database_url = URL.create(
    drivername='mysql+mysqldb',
    host=Env.DATABASE_HOST,
    database=Env.DATABASE_NAME,
    port=Env.DATABASE_PORT,
    username=Env.DATABASE_USER,
    password=Env.DATABASE_PASSWORD,
)
_engine = create_engine(_database_url, echo=True)
_session = sessionmaker(autocommit=False, autoflush=True, bind=_engine)
Base = declarative_base()


def get_db():
    db = _session()
    try:
        yield db
    finally:
        db.close()