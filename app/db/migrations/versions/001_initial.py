"""Initial database migration.

Revision ID: 001
Revises:
Create Date: 2024-03-19 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("api_key", sa.String(), nullable=True),
        sa.Column("subscription_tier", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(
        op.f("ix_users_username"), "users", ["username"], unique=True
    )

    # Create reputation_scores table
    op.create_table(
        "reputation_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("metrics", sqlite.JSON(), nullable=True),
        sa.Column("alerts", sqlite.JSON(), nullable=True),
        sa.Column("timeframe", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_reputation_scores_platform"),
        "reputation_scores",
        ["platform"],
        unique=False,
    )
    op.create_index(
        op.f("ix_reputation_scores_username"),
        "reputation_scores",
        ["username"],
        unique=False,
    )

    # Create monitoring_configs table
    op.create_table(
        "monitoring_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("alert_thresholds", sqlite.JSON(), nullable=True),
        sa.Column("monitoring_interval", sa.Integer(), nullable=False),
        sa.Column("alert_channels", sqlite.JSON(), nullable=True),
        sa.Column("keywords", sqlite.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_monitoring_configs_platform"),
        "monitoring_configs",
        ["platform"],
        unique=False,
    )
    op.create_index(
        op.f("ix_monitoring_configs_username"),
        "monitoring_configs",
        ["username"],
        unique=False,
    )

    # Create response_templates table
    op.create_table(
        "response_templates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("sentiment", sa.String(), nullable=False),
        sa.Column("variables", sqlite.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_response_templates_category"),
        "response_templates",
        ["category"],
        unique=False,
    )

    # Create response_rules table
    op.create_table(
        "response_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("trigger_type", sa.String(), nullable=False),
        sa.Column("trigger_value", sa.String(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("conditions", sqlite.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["template_id"],
            ["response_templates.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_response_rules_trigger_type"),
        "response_rules",
        ["trigger_type"],
        unique=False,
    )

    # Create auto_responses table
    op.create_table(
        "auto_responses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("comment_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("original_text", sa.String(), nullable=False),
        sa.Column("response_text", sa.String(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=True),
        sa.Column("sentiment_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("response_time", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["template_id"],
            ["response_templates.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_auto_responses_comment_id"),
        "auto_responses",
        ["comment_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_auto_responses_platform"),
        "auto_responses",
        ["platform"],
        unique=False,
    )
    op.create_index(
        op.f("ix_auto_responses_user_id"),
        "auto_responses",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("auto_responses")
    op.drop_table("response_rules")
    op.drop_table("response_templates")
    op.drop_table("monitoring_configs")
    op.drop_table("reputation_scores")
    op.drop_table("users")
