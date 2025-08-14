from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    api_key = Column(String, unique=True, nullable=True)
    subscription_tier = Column(String, default="free")
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


class ReputationScore(Base):
    __tablename__ = "reputation_scores"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, index=True)
    username = Column(String, index=True)
    score = Column(Float)
    metrics = Column(JSON)
    alerts = Column(JSON)
    timeframe = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


class MonitoringConfig(Base):
    __tablename__ = "monitoring_configs"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, index=True)
    username = Column(String, index=True)
    alert_thresholds = Column(JSON)
    monitoring_interval = Column(Integer, default=300)
    alert_channels = Column(JSON)
    keywords = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


class ResponseTemplate(Base):
    __tablename__ = "response_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    content = Column(String)
    category = Column(String, index=True)
    sentiment = Column(String)
    variables = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)


class ResponseRule(Base):
    __tablename__ = "response_rules"

    id = Column(Integer, primary_key=True, index=True)
    trigger_type = Column(String, index=True)
    trigger_value = Column(String)
    template_id = Column(Integer, ForeignKey("response_templates.id"))
    priority = Column(Integer, default=1)
    conditions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    template = relationship("ResponseTemplate", back_populates="rules")


ResponseTemplate.rules = relationship(
    "ResponseRule", back_populates="template")


class AutoResponse(Base):
    __tablename__ = "auto_responses"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, index=True)
    comment_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    original_text = Column(String)
    response_text = Column(String)
    template_id = Column(
        Integer,
        ForeignKey("response_templates.id"),
        nullable=True)
    sentiment_score = Column(Float)
    status = Column(String, default="pending")
    response_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    template = relationship("ResponseTemplate")
