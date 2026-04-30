from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# GANTI sesuai MySQL kamu
DATABASE_URL = "mysql+pymysql://root:12345@localhost/sisfota"

engine = create_engine(
    DATABASE_URL
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()