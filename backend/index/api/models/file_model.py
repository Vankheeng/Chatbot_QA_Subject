from __future__ import annotations

from mongoengine import connect
from mongoengine import DictField
from mongoengine import Document
from mongoengine import IntField
from mongoengine import StringField
from mongoengine import ListField

# Kết nối hai database
connect(alias='chunk_db', db='chunk_db', host='mongodb://localhost:27018/chunk_db')


class Chunk(Document):
    header = StringField(required=True)
    topic = StringField(required=True)
    chunk = StringField(required=True)
    page = IntField(required=True)
    sub_chunks = ListField(StringField(), required=True)

    meta = {'db_alias': 'chunk_db'}


Chunk.objects().first()
