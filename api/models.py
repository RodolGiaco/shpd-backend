from sqlalchemy import Column, String, Integer
from api.database import Base

class Paciente(Base):
    __tablename__ = "pacientes"

    id = Column(String, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    edad = Column(Integer)
    sexo = Column(String)
    diagnostico = Column(String)
