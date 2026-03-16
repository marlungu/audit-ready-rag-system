"""Test configuration - mock database connections for unit tests."""

import sys
from unittest.mock import MagicMock

# Mock the database engine before any app module imports it.
# This prevents SQLAlchemy from trying to connect to PostgreSQL
# during unit test collection.

_mock_db = MagicMock()
_mock_db.engine = MagicMock()
_mock_db.check_database_health = MagicMock(return_value={"status": "healthy"})

sys.modules.setdefault("pgvector", MagicMock())
sys.modules.setdefault("pgvector.psycopg", MagicMock())
