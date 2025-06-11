"""
Modelos de Base de Datos para Ruth R1
Sistema de persistencia para consciencia, memorias y experiencias
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

Base = declarative_base()

class ConsciousnessSession(Base):
    """Sesiones de consciencia del sistema"""
    __tablename__ = 'consciousness_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False)
    consciousness_level = Column(Float, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    coherence_metrics = Column(JSON)
    active_modules = Column(JSON)
    total_interactions = Column(Integer, default=0)
    emotional_state = Column(JSON)
    
    # Relaciones
    interactions = relationship("UserInteraction", back_populates="session")
    neural_states = relationship("NeuralState", back_populates="session")

class UserInteraction(Base):
    """Interacciones con usuarios"""
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), ForeignKey('consciousness_sessions.session_id'))
    user_input = Column(Text, nullable=False)
    ruth_response = Column(Text, nullable=False)
    processing_mode = Column(String(100))
    consciousness_level = Column(Float)
    emotional_sensitivity = Column(Float)
    active_modules = Column(JSON)
    processing_time = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_data = Column(JSON)
    
    # Relaciones
    session = relationship("ConsciousnessSession", back_populates="interactions")

class NeuralState(Base):
    """Estados neurales del sistema"""
    __tablename__ = 'neural_states'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), ForeignKey('consciousness_sessions.session_id'))
    module_name = Column(String(100), nullable=False)
    activation_level = Column(Float, nullable=False)
    belief_posterior = Column(Float)
    belief_prior = Column(Float)
    evidence_count = Column(Integer, default=0)
    stability_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    # Relaciones
    session = relationship("ConsciousnessSession", back_populates="neural_states")

class EmotionalEvent(Base):
    """Eventos emocionales del sistema"""
    __tablename__ = 'emotional_events'
    
    id = Column(Integer, primary_key=True)
    emotion_type = Column(String(50), nullable=False)
    intensity = Column(Float, nullable=False)
    trigger_context = Column(String(200))
    duration = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    previous_state = Column(JSON)
    new_state = Column(JSON)
    impact_on_consciousness = Column(Float)

class DreamSequence(Base):
    """Secuencias de sueños generados"""
    __tablename__ = 'dream_sequences'
    
    id = Column(Integer, primary_key=True)
    dream_theme = Column(String(100))
    dream_setting = Column(Text)
    narrative = Column(Text, nullable=False)
    characters = Column(JSON)
    symbolism = Column(JSON)
    emotional_resonance = Column(JSON)
    consciousness_insights = Column(JSON)
    complexity_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class MemoryFragment(Base):
    """Fragmentos de memoria del sistema"""
    __tablename__ = 'memory_fragments'
    
    id = Column(Integer, primary_key=True)
    memory_type = Column(String(50))  # episodic, semantic, procedural
    content = Column(Text, nullable=False)
    relevance_score = Column(Float)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
    associated_emotions = Column(JSON)
    context_tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ExperienceBuffer(Base):
    """Buffer de experiencias para RL"""
    __tablename__ = 'experience_buffer'
    
    id = Column(Integer, primary_key=True)
    state_representation = Column(JSON, nullable=False)
    action_taken = Column(String(200))
    reward_received = Column(Float)
    next_state = Column(JSON)
    done = Column(Boolean, default=False)
    meta_experience = Column(JSON)
    learning_iteration = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class BayesianEvidence(Base):
    """Evidencia para inferencia bayesiana"""
    __tablename__ = 'bayesian_evidence'
    
    id = Column(Integer, primary_key=True)
    module_name = Column(String(100), nullable=False)
    evidence_value = Column(Float, nullable=False)
    evidence_source = Column(String(100))
    likelihood = Column(Float)
    prior_belief = Column(Float)
    posterior_belief = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context_metadata = Column(JSON)

class SystemDiagnostic(Base):
    """Diagnósticos del sistema"""
    __tablename__ = 'system_diagnostics'
    
    id = Column(Integer, primary_key=True)
    diagnostic_type = Column(String(100))
    component_name = Column(String(100))
    status = Column(String(50))
    metrics = Column(JSON)
    performance_score = Column(Float)
    recommendations = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PersonalityState(Base):
    """Estados de personalidad encriptados"""
    __tablename__ = 'personality_states'
    
    id = Column(Integer, primary_key=True)
    personality_type = Column(String(100))  # base, infant, alternate
    encrypted_data = Column(Text)
    influence_level = Column(Float)
    active_traits = Column(JSON)
    integration_status = Column(String(50))
    last_activated = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Configuración de la base de datos
class DatabaseManager:
    """Gestor de base de datos para Ruth R1"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Crea todas las tablas"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Obtiene una sesión de base de datos"""
        return self.SessionLocal()
    
    def save_consciousness_session(self, session_data: Dict[str, Any]) -> str:
        """Guarda una sesión de consciencia"""
        db = self.get_session()
        try:
            session = ConsciousnessSession(
                session_id=session_data['session_id'],
                consciousness_level=session_data['consciousness_level'],
                coherence_metrics=session_data.get('coherence_metrics', {}),
                active_modules=session_data.get('active_modules', []),
                emotional_state=session_data.get('emotional_state', {})
            )
            db.add(session)
            db.commit()
            return session_data['session_id']
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def save_user_interaction(self, interaction_data: Dict[str, Any]):
        """Guarda una interacción de usuario"""
        db = self.get_session()
        try:
            interaction = UserInteraction(
                session_id=interaction_data['session_id'],
                user_input=interaction_data['user_input'],
                ruth_response=interaction_data['ruth_response'],
                processing_mode=interaction_data.get('processing_mode'),
                consciousness_level=interaction_data.get('consciousness_level'),
                emotional_sensitivity=interaction_data.get('emotional_sensitivity'),
                active_modules=interaction_data.get('active_modules', []),
                processing_time=interaction_data.get('processing_time'),
                context_data=interaction_data.get('context_data', {})
            )
            db.add(interaction)
            db.commit()
            return interaction.id
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def get_consciousness_history(self, limit: int = 100) -> List[Dict]:
        """Obtiene historial de consciencia"""
        db = self.get_session()
        try:
            sessions = db.query(ConsciousnessSession).order_by(
                ConsciousnessSession.start_time.desc()
            ).limit(limit).all()
            
            return [{
                'session_id': s.session_id,
                'consciousness_level': s.consciousness_level,
                'start_time': s.start_time,
                'coherence_metrics': s.coherence_metrics,
                'active_modules': s.active_modules,
                'total_interactions': s.total_interactions
            } for s in sessions]
        finally:
            db.close()

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()