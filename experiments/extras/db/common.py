import datetime
import os
from typing import Optional

from dotenv import load_dotenv
from peewee import DateTimeField, Model, SqliteDatabase

load_dotenv()

DEFAULT_DATABASE_URL = os.environ.get("DATABASE_URL", "")
db: Optional[SqliteDatabase] = None


def create_global_data_store(database_path: str = DEFAULT_DATABASE_URL):
    global db
    if not database_path:
        raise ValueError("Database path must be provided.")

    if not db:
        db = SqliteDatabase(database_path)

    return db


def get_global_database(database_path: str = DEFAULT_DATABASE_URL):
    if not db:
        return create_global_data_store(database_path)

    return db


class SharedModel(Model):
    class Meta:
        database = get_global_database()
        table_name: str

    created_at = DateTimeField(default=datetime.datetime.now)
