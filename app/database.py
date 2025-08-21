from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, DateTime, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import MetaData, inspect
from sqlalchemy.sql import func
import os
from datetime import date

# SQLite database URL
DATABASE_URL = "sqlite:///./advisory_firms.db"

# Create SQLite engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define models
class AdvisoryFirm(Base):
    __tablename__ = "advisory_firms"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    founded_year = Column(Integer)
    headquarters = Column(String)
    employee_count = Column(Integer)
    revenue_millions = Column(Float)
    services = Column(Text)
    website = Column(String)
    created_at = Column(DateTime, default=func.now())
    
    clients = relationship("Client", back_populates="firm")
    consultants = relationship("Consultant", back_populates="firm")

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    industry = Column(String)
    firm_id = Column(Integer, ForeignKey("advisory_firms.id"))
    project_type = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)
    value_millions = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    firm = relationship("AdvisoryFirm", back_populates="clients")

class Consultant(Base):
    __tablename__ = "consultants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    firm_id = Column(Integer, ForeignKey("advisory_firms.id"))
    specialization = Column(String)
    experience_years = Column(Integer)
    education = Column(String)
    certifications = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    firm = relationship("AdvisoryFirm", back_populates="consultants")

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_database_schema():
    metadata = MetaData()
    metadata.reflect(bind=engine)
    
    schema_info = {}
    for table_name in metadata.tables:
        table = metadata.tables[table_name]
        columns = []
        for column in table.columns:
            columns.append({
                "name": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
                "primary_key": column.primary_key
            })
        
        schema_info[table_name] = {
            "columns": columns,
            "foreign_keys": [
                {
                    "column": fk.parent.name,
                    "references": f"{fk.column.table.name}.{fk.column.name}"
                }
                for fk in table.foreign_keys
            ]
        }
    
    return schema_info

def init_database():
    create_tables()
    
    db = SessionLocal()
    
    if db.query(AdvisoryFirm).first():
        db.close()
        return
    
    firms = [
        AdvisoryFirm(
            name="McKinsey & Company",
            industry="Management Consulting",
            founded_year=1926,
            headquarters="New York, NY",
            employee_count=30000,
            revenue_millions=12000.0,
            services="Strategy, Operations, Digital Transformation",
            website="https://www.mckinsey.com"
        ),
        AdvisoryFirm(
            name="Bain & Company",
            industry="Management Consulting",
            founded_year=1973,
            headquarters="Boston, MA",
            employee_count=12000,
            revenue_millions=5800.0,
            services="Strategy, M&A, Performance Improvement",
            website="https://www.bain.com"
        )
    ]
    
    db.add_all(firms)
    db.commit()
    
    clients = [
        Client(name="TechCorp Inc", industry="Technology", firm_id=1, project_type="Digital Transformation", start_date=date(2024, 1, 15), end_date=date(2024, 6, 30), value_millions=5.2),
        Client(name="Global Retail", industry="Retail", firm_id=1, project_type="Strategy", start_date=date(2024, 2, 1), end_date=date(2024, 4, 30), value_millions=3.8)
    ]
    
    db.add_all(clients)
    db.commit()
    
    consultants = [
        Consultant(name="Dr. Sarah Johnson", firm_id=1, specialization="Strategy", experience_years=8, education="PhD Harvard", certifications="PMP, Six Sigma"),
        Consultant(name="Michael Chen", firm_id=1, specialization="Operations", experience_years=6, education="MBA Stanford", certifications="Lean, Agile")
    ]
    
    db.add_all(consultants)
    db.commit()
    
    db.close()
