import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import pymysql; pymysql.install_as_MySQLdb()


def db_engine() -> sqlalchemy.engine.base.Engine:
    """ Get the SQL engine, based on environment variables.
        :return: SQL engine corresponding to the parameters given as environment variables (in the .bash_profile file).
        """
    db_user = os.getenv('IMPRESSO_MYSQL_USER')
    db_host = os.getenv('IMPRESSO_MYSQL_HOST')
    db_name = os.getenv('IMPRESSO_MYSQL_DB')
    db_password = os.getenv('IMPRESSO_MYSQL_PWD')
    db_url = f'mysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=UTF8MB4'
    engine = create_engine(db_url, poolclass=NullPool)

    return engine


def read_table(table_name: str, eng: sqlalchemy.engine.base.Engine) -> pd.core.frame.DataFrame:
    """ Load a whole table from SQL.
        :param table_name: Name of the table we want to get.
        :param eng: SQL engine.
        :return: Pandas data frame corresponding to the SQL table.
        """
    return pd.read_sql('SELECT * FROM {};'.format(table_name), eng)


def export_table_csv(table: pd.core.frame.DataFrame, path: str) -> None:
    """ Export pandas data frame to a csv file.
        Can be useful if you want to save a copy of the data locally.
        :param table: Table which we want to export.
        :param path: Path to which we want to save the file (e.g. r'../local-data/newspapers.csv').
        :return: Nothing
        """
    table.to_csv(path)


def import_table_csv(path: str) -> pd.core.frame.DataFrame:
    """ Load table that has been exported using function export_table_csv.
        :param path: path to the csv file (e.g. '../local-data/newspapers.csv').
        :return: Pandas data frame corresponding to the csv file.
        """
    return pd.read_csv(path, index_col=0)

