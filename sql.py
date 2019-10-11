import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool


def db_engine() :
    db_user = os.getenv('IMPRESSO_MYSQL_USER')
    db_host = os.getenv('IMPRESSO_MYSQL_HOST')
    db_name = os.getenv('IMPRESSO_MYSQL_DB')
    db_password = os.getenv('IMPRESSO_MYSQL_PWD')
    db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=utf8'
    engine = create_engine(db_url, poolclass=NullPool)
    # Debug in case environment variables cannot be found
    #print(os.environ)
    #print(db_user)
    return engine


def read_table(table_name: str, eng: ):
    return pd.read_sql('SELECT * FROM {};'.format(table_name), eng)


def export_table_csv(table, path):
    #use example : export_table_csv(newspapers_df, r'../local-data/newspapers.csv')
    table.to_csv(path)


def import_table_csv(path):
    # recover table that has been exported using function above : export_table_csv
    # use example : recover = import_table_csv('../local-data/newspapers.csv')
    return pd.read_csv(path, index_col=0)
