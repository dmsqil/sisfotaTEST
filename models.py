from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from database import Base

class KlasifikasiTA(Base):
    __tablename__ = "klasifikasi_ta"

    id = Column(Integer, primary_key=True, index=True)
    judul = Column(String(255))
    abstrak = Column(Text)
    hasil = Column(String(150))
    created_at = Column(DateTime, default=datetime.utcnow)