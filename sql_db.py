import os
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

def db_engine() :
    db_user = os.getenv('IMPRESSO_MYSQL_USER')
    db_host = os.getenv('IMPRESSO_MYSQL_HOST')
    db_name = os.getenv('IMPRESSO_MYSQL_DB')
    db_password = os.getenv('IMPRESSO_MYSQL_PWD')
    db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=utf8'
    engine = create_engine(db_url, poolclass=NullPool)
    return engine
