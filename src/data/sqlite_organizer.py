import sqlite3
import os
import pandas as pd
from viz import dataframe_calculators as dfc


def create_connection():
    """ create a connection to a SQLite database"""
    directory = '../data/interim/sqlite'
    if not os.path.exists(directory):
        os.makedirs(directory)
    db_file = directory + '/geos5'
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return(conn)


def add_dataframe(frame, var, date, conn, dtheta):

    cursor = conn.cursor()
    cursor.execute(
        "select count(name) from sqlite_master where  type='table' and name=?", var)
    if cursor.fetchone()[0] == 1:
        action = "exists"
    else:
        action = "replace"
    df = pd.DataFrame().stack(frame).rename_axis(
        ['y', 'x']).reset_index(name=var.lower())
    df['datetime'] = pd.Timestamp(date)
    df.set_index('datetime', inplace=True)
    df = dfc.latlon_converter(df, dtheta)
    df.to_sql(var, conn, if_exists=action)
