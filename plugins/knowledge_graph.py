"""
Knowledge graph plugin.
Builds and maintains a knowledge graph of entities and relationships.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
import networkx as nx
from datetime import datetime, timezone
from app.core.plugins.base import KnowledgeGraphPlugin, PluginType, PluginMetadata
from app.core.error_handling import ReputationError, ErrorSeverity, ErrorCategory

class EntityData(TypedDict):
    """Type definition for entity data."""
    id: str
    type: str
    attributes: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class RelationshipData(TypedDict):
    """Type definition for relationship data."""
    source_id: str
    target_id: str
    type: str
    attributes: Dict[str, Any]
    created_at: datetime

class ReputationKnowledgeGraph(KnowledgeGraphPlugin):
    """Plugin for building and maintaining a knowledge graph."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge graph plugin."""
        super().__init__(config)
        self.graph = nx.DiGraph()
        self.entity_types: Dict[str, Dict[str, Any]] = {}
        self.relationship_types: Dict[str, Dict[str, Any]] = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="reputation_knowledge_graph",
            version="1.0.0",
            description="Knowledge graph plugin for reputation monitoring",
            author="Reputation Sync Team",
            type=PluginType.KNOWLEDGE_GRAPH,
            config_schema={
                "type": "object",
                "properties": {
                    "max_entities": {
                        "type": "integer",
                        "description": "Maximum number of entities",
                        "minimum": 1,
                        "default": 10000
                    },
                    "max_relationships": {
                        "type": "integer",
                        "description": "Maximum number of relationships",
                        "minimum": 1,
                        "default": 50000
                    },
                    "entity_types": {
                        "type": "object",
                        "description": "Entity type definitions",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "attributes": {
                                    "type": "object",
                                    "description": "Required attributes for entity type"
                                }
                            }
                        }
                    },
                    "relationship_types": {
                        "type": "object",
                        "description": "Relationship type definitions",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "source_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Valid source entity types"
                                },
                                "target_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Valid target entity types"
                                },
                                "attributes": {
                                    "type": "object",
                                    "description": "Required attributes for relationship type"
                                }
                            }
                        }
                    }
                }
            }
        )
    
    async def initialize(self) -> bool:
        """Initialize plugin."""
        try:
            # Load entity and relationship type definitions
            self.entity_types = self.config.get("entity_types", {})
            self.relationship_types = self.config.get("relationship_types", {})
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error initializing knowledge graph: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def shutdown(self) -> bool:
        """Shutdown plugin."""
        try:
            self.graph.clear()
            return True
        except Exception as e:
            raise ReputationError(
                message=f"Error shutting down knowledge graph: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def add_entity(
        self,
        entity_type: str,
        attributes: Dict[str, Any]
    ) -> str:
        """Add entity to knowledge graph."""
        try:
            # Validate entity type
            if entity_type not in self.entity_types:
                raise ReputationError(
                    message=f"Invalid entity type: {entity_type}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check entity limit
            if len(self.graph.nodes) >= self.config.get("max_entities", 10000):
                raise ReputationError(
                    message="Maximum number of entities reached",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Generate entity ID
            entity_id = f"{entity_type}_{len(self.graph.nodes)}"
            
            # Create entity data
            now = datetime.now(timezone.utc)
            entity_data: EntityData = {
                "id": entity_id,
                "type": entity_type,
                "attributes": attributes,
                "created_at": now,
                "updated_at": now
            }
            
            # Add entity to graph
            self.graph.add_node(entity_id, **entity_data)
            
            return entity_id
            
        except Exception as e:
            raise ReputationError(
                message=f"Error adding entity: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add relationship to knowledge graph."""
        try:
            # Validate relationship type
            if relationship_type not in self.relationship_types:
                raise ReputationError(
                    message=f"Invalid relationship type: {relationship_type}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check if source and target exist
            if not self.graph.has_node(source_id):
                raise ReputationError(
                    message=f"Source entity not found: {source_id}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            if not self.graph.has_node(target_id):
                raise ReputationError(
                    message=f"Target entity not found: {target_id}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Check relationship limit
            if len(self.graph.edges) >= self.config.get("max_relationships", 50000):
                raise ReputationError(
                    message="Maximum number of relationships reached",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Validate entity types
            source_type = self.graph.nodes[source_id]["type"]
            target_type = self.graph.nodes[target_id]["type"]
            
            relationship_def = self.relationship_types[relationship_type]
            if source_type not in relationship_def.get("source_types", []):
                raise ReputationError(
                    message=f"Invalid source entity type for relationship: {source_type}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            if target_type not in relationship_def.get("target_types", []):
                raise ReputationError(
                    message=f"Invalid target entity type for relationship: {target_type}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Create relationship data
            relationship_data: RelationshipData = {
                "source_id": source_id,
                "target_id": target_id,
                "type": relationship_type,
                "attributes": attributes or {},
                "created_at": datetime.now(timezone.utc)
            }
            
            # Add relationship to graph
            self.graph.add_edge(source_id, target_id, **relationship_data)
            
            return True
            
        except Exception as e:
            raise ReputationError(
                message=f"Error adding relationship: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def query_entities(
        self,
        query: Dict[str, Any]
    ) -> List[EntityData]:
        """Query entities in knowledge graph."""
        try:
            # Extract query parameters
            entity_type = query.get("type")
            attributes = query.get("attributes", {})
            
            # Find matching entities
            matches: List[EntityData] = []
            
            for node_id, node_data in self.graph.nodes(data=True):
                # Check entity type
                if entity_type and node_data["type"] != entity_type:
                    continue
                
                # Check attributes
                if attributes:
                    if not all(
                        node_data["attributes"].get(key) == value
                        for key, value in attributes.items()
                    ):
                        continue
                
                matches.append(node_data)
            
            return matches
            
        except Exception as e:
            raise ReputationError(
                message=f"Error querying entities: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None
    ) -> List[RelationshipData]:
        """Get relationships for an entity."""
        try:
            # Check if entity exists
            if not self.graph.has_node(entity_id):
                raise ReputationError(
                    message=f"Entity not found: {entity_id}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Get relationships
            relationships: List[RelationshipData] = []
            
            # Get outgoing relationships
            for _, target_id, edge_data in self.graph.out_edges(entity_id, data=True):
                if relationship_type and edge_data["type"] != relationship_type:
                    continue
                relationships.append(edge_data)
            
            # Get incoming relationships
            for source_id, _, edge_data in self.graph.in_edges(entity_id, data=True):
                if relationship_type and edge_data["type"] != relationship_type:
                    continue
                relationships.append(edge_data)
            
            return relationships
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting entity relationships: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_entity_influence(
        self,
        entity_id: str
    ) -> Dict[str, Any]:
        """Calculate influence metrics for an entity."""
        try:
            # Check if entity exists
            if not self.graph.has_node(entity_id):
                raise ReputationError(
                    message=f"Entity not found: {entity_id}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Calculate metrics
            metrics = {
                "direct_followers": len(list(self.graph.predecessors(entity_id))),
                "direct_following": len(list(self.graph.successors(entity_id))),
                "total_reach": 0,
                "influence_score": 0.0
            }
            
            # Calculate total reach (number of entities within 2 degrees)
            reach = set()
            for follower in self.graph.predecessors(entity_id):
                reach.add(follower)
                for follower_of_follower in self.graph.predecessors(follower):
                    reach.add(follower_of_follower)
            metrics["total_reach"] = len(reach)
            
            # Calculate influence score
            if metrics["direct_followers"] > 0:
                metrics["influence_score"] = (
                    metrics["direct_followers"] * 0.4 +
                    metrics["total_reach"] * 0.6
                ) / 100  # Normalize to 0-1 range
            
            return metrics
            
        except Exception as e:
            raise ReputationError(
                message=f"Error calculating entity influence: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )
    
    async def get_entity_recommendations(
        self,
        entity_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get entity recommendations based on relationships."""
        try:
            # Check if entity exists
            if not self.graph.has_node(entity_id):
                raise ReputationError(
                    message=f"Entity not found: {entity_id}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.BUSINESS
                )
            
            # Get entity's relationships
            entity_relationships = await self.get_entity_relationships(entity_id)
            
            # Find similar entities
            similarities: List[Dict[str, Any]] = []
            
            for other_id in self.graph.nodes():
                if other_id == entity_id:
                    continue
                
                # Get other entity's relationships
                other_relationships = await self.get_entity_relationships(other_id)
                
                # Calculate similarity score
                common_relationships = len(
                    set(r["type"] for r in entity_relationships) &
                    set(r["type"] for r in other_relationships)
                )
                
                if common_relationships > 0:
                    similarities.append({
                        "entity_id": other_id,
                        "entity_type": self.graph.nodes[other_id]["type"],
                        "similarity_score": common_relationships / max(
                            len(entity_relationships),
                            len(other_relationships)
                        )
                    })
            
            # Sort by similarity score and return top recommendations
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            raise ReputationError(
                message=f"Error getting entity recommendations: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            ) 