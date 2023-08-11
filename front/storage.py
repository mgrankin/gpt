dbhost = '192.168.1.4'
THREADS_NUM = 10
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
engine = create_engine(f'postgresql+psycopg2://postgres:rlp4ZKc6oC0OzgK1FSsJ@{dbhost}:5535', pool_size=THREADS_NUM+2)
session_maker = sessionmaker(bind=engine, autocommit=False)
connection = engine.connect()
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert, ARRAY, REAL
from sqlalchemy import Table, Column, LargeBinary, DateTime, Integer, String, MetaData, ForeignKey, update, func, delete, select, text, Index
import pandas as pd

metadata = MetaData()

logs = Table('logs', metadata,
     Column('created', DateTime, nullable=False, server_default=func.current_timestamp()),
     Column('ip', String, ),
     Column('origin', String, ),
     Column('agent', String, ),
     Column('fs', String, ),
     Column('ff', String, ),
     Column('session', String, ),
     Column('type', Integer, ),
     Column('hash', String, ),
     Column('log', String, ),
     Column('model', String, ),
)

ban = Table('ban', metadata,
     Column('created', DateTime, nullable=False, server_default=func.current_timestamp()),
     Column('ip', String, primary_key=True),
     Column('ban_type', Integer, nullable=False, server_default="1"),
     Column('note', String, ),
)
#ban.drop(engine)
#connection.execute(delete(ban))

metadata.create_all(engine)
Index("log_created_ix", logs.c.created).create(connection, checkfirst=True)

def log(model,request, type, hash, log):
    sessionid = request.cookies.get('sessionid')
    insert_stmt = insert(logs).values(ip=request.headers.get('X-Real-IP'), origin=request.headers.get('Origin'),
                                      agent=request.headers.get('User-Agent'), fs=request.headers.get('X-Forwarded-Server'),
                                      ff=request.headers.get('X-Forwarded-For'),
                                      session=f'{sessionid}', type=type, hash=hash, 
                                      log=str(log), model=model)
    connection.execute(insert_stmt)

def get_ban_ip(ip):
    ban_type = pd.read_sql_query(text("select ban_type from ban where ip = :ip"), connection, params={"ip":ip})
    if ban_type.empty: return 0
    return int(ban_type.iloc[0]['ban_type'])

def get_ban(request):
    ip = request.headers.get('X-Real-IP')
    return get_ban_ip(ip)