"""
Knowledge Graph Configuration

PATTERN: Configuration management with validation
WHY: Centralized settings for Neo4j connectivity
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class KnowledgeGraphConfig:
    """
    Configuration for Neo4j knowledge graph

    Supports both embedded and server deployments:
    - Embedded: neo4j-community for Railway/local deployment
    - Server: Bolt protocol for remote Neo4j instances

    Environment Variables:
    - NEO4J_URI: Connection URI (default: bolt://localhost:7687)
    - NEO4J_USER: Username (default: neo4j)
    - NEO4J_PASSWORD: Password (required for server mode)
    - NEO4J_DATABASE: Database name (default: neo4j)
    - NEO4J_EMBEDDED: Use embedded mode (default: true)
    - NEO4J_DATA_PATH: Path for embedded database (default: ./data/neo4j)
    """

    # Connection settings
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: Optional[str] = os.getenv("NEO4J_PASSWORD")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Embedded mode settings
    embedded: bool = os.getenv("NEO4J_EMBEDDED", "true").lower() == "true"
    data_path: str = os.getenv("NEO4J_DATA_PATH", "./data/neo4j")

    # Connection pool settings
    max_connection_pool_size: int = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))
    connection_timeout: float = float(os.getenv("NEO4J_CONNECT_TIMEOUT", "30.0"))
    max_transaction_retry_time: float = float(os.getenv("NEO4J_RETRY_TIME", "30.0"))

    # Query settings
    default_query_timeout: float = float(os.getenv("NEO4J_QUERY_TIMEOUT", "5.0"))

    # Graph algorithm settings
    max_relationship_depth: int = int(os.getenv("GRAPH_MAX_DEPTH", "3"))
    min_relationship_strength: float = float(os.getenv("GRAPH_MIN_STRENGTH", "0.3"))

    # Concept extraction settings
    min_concept_frequency: int = int(os.getenv("CONCEPT_MIN_FREQUENCY", "2"))
    concept_similarity_threshold: float = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", "0.7"))

    def __post_init__(self):
        """Validate configuration"""
        if self.embedded:
            # Ensure data path exists for embedded mode
            Path(self.data_path).mkdir(parents=True, exist_ok=True)
        else:
            # Validate server connection settings
            if not self.password:
                raise ValueError("NEO4J_PASSWORD required for server mode")

    @property
    def connection_config(self) -> dict:
        """Get connection configuration for Neo4j driver"""
        return {
            "max_connection_pool_size": self.max_connection_pool_size,
            "connection_timeout": self.connection_timeout,
            "max_transaction_retry_time": self.max_transaction_retry_time,
        }

    def get_embedded_config(self) -> dict:
        """Get configuration for embedded Neo4j"""
        if not self.embedded:
            return {}

        return {
            "dbms.directories.data": self.data_path,
            "dbms.connector.bolt.enabled": "true",
            "dbms.connector.bolt.listen_address": "localhost:7687",
            "dbms.memory.heap.initial_size": "256m",
            "dbms.memory.heap.max_size": "512m",
            "dbms.memory.pagecache.size": "128m",
        }
