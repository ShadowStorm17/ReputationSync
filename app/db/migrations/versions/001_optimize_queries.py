"""Optimize database queries and add indexes.

Revision ID: 001
Revises:
Create Date: 2024-03-09 10:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add indexes for reputation scores
    op.create_index(
        "ix_reputation_scores_platform_username",
        "reputation_scores",
        ["platform", "username"],
    )
    op.create_index(
        "ix_reputation_scores_score", "reputation_scores", ["score"]
    )
    op.create_index(
        "ix_reputation_scores_created_at", "reputation_scores", ["created_at"]
    )

    # Add indexes for user analytics
    op.create_index(
        "ix_user_analytics_user_id_platform",
        "user_analytics",
        ["user_id", "platform"],
    )
    op.create_index(
        "ix_user_analytics_timestamp", "user_analytics", ["timestamp"]
    )

    # Add indexes for sentiment analysis
    op.create_index(
        "ix_sentiment_analysis_entity_id_platform",
        "sentiment_analysis",
        ["entity_id", "platform"],
    )
    op.create_index(
        "ix_sentiment_analysis_timestamp", "sentiment_analysis", ["timestamp"]
    )

    # Add indexes for engagement metrics
    op.create_index(
        "ix_engagement_metrics_entity_id_platform",
        "engagement_metrics",
        ["entity_id", "platform"],
    )
    op.create_index(
        "ix_engagement_metrics_timestamp", "engagement_metrics", ["timestamp"]
    )

    # Add indexes for monitoring alerts
    op.create_index(
        "ix_monitoring_alerts_entity_id_platform",
        "monitoring_alerts",
        ["entity_id", "platform"],
    )
    op.create_index(
        "ix_monitoring_alerts_created_at", "monitoring_alerts", ["created_at"]
    )
    op.create_index(
        "ix_monitoring_alerts_status", "monitoring_alerts", ["status"]
    )


def downgrade():
    # Remove indexes
    op.drop_index("ix_reputation_scores_platform_username")
    op.drop_index("ix_reputation_scores_score")
    op.drop_index("ix_reputation_scores_created_at")
    op.drop_index("ix_user_analytics_user_id_platform")
    op.drop_index("ix_user_analytics_timestamp")
    op.drop_index("ix_sentiment_analysis_entity_id_platform")
    op.drop_index("ix_sentiment_analysis_timestamp")
    op.drop_index("ix_engagement_metrics_entity_id_platform")
    op.drop_index("ix_engagement_metrics_timestamp")
    op.drop_index("ix_monitoring_alerts_entity_id_platform")
    op.drop_index("ix_monitoring_alerts_created_at")
    op.drop_index("ix_monitoring_alerts_status")
