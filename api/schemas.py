from pydantic import BaseModel

class PacienteOut(BaseModel):
    nombre: str
    edad: int
    sexo: str
    diagnostico: str

    class Config:
        orm_mode = True
