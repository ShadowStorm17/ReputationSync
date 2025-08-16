"""
GraphQL service for schema and resolver management.
Provides GraphQL integration and query handling.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import strawberry
from strawberry.schema import Schema

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SchemaField:
    """GraphQL schema field representation."""

    name: str
    type: str
    description: Optional[str] = None
    args: Dict[str, str] = None
    is_list: bool = False
    is_required: bool = False
    default_value: Any = None


@dataclass
class SchemaType:
    """GraphQL schema type representation."""

    name: str
    description: Optional[str] = None
    fields: List[SchemaField] = None
    interfaces: List[str] = None
    is_input: bool = False


class SchemaBuilder:
    """GraphQL schema builder."""

    def __init__(self):
        """Initialize schema builder."""
        self.types: Dict[str, SchemaType] = {}
        self.queries: Dict[str, SchemaField] = {}
        self.mutations: Dict[str, SchemaField] = {}
        self.subscriptions: Dict[str, SchemaField] = {}
        self.resolvers: Dict[str, Callable] = {}

    def add_type(self, type_def: SchemaType) -> bool:
        """Add type to schema."""
        try:
            if type_def.name in self.types:
                return False

            self.types[type_def.name] = type_def
            return True

        except Exception as e:
            logger.error("Add type error: %s", e)
            return False

    def add_query(self, field: SchemaField, resolver: Callable) -> bool:
        """Add query to schema."""
        try:
            if field.name in self.queries:
                return False

            self.queries[field.name] = field
            self.resolvers[f"query_{field.name}"] = resolver
            return True

        except Exception as e:
            logger.error("Add query error: %s", e)
            return False

    def add_mutation(self, field: SchemaField, resolver: Callable) -> bool:
        """Add mutation to schema."""
        try:
            if field.name in self.mutations:
                return False

            self.mutations[field.name] = field
            self.resolvers[f"mutation_{field.name}"] = resolver
            return True

        except Exception as e:
            logger.error("Add mutation error: %s", e)
            return False

    def add_subscription(self, field: SchemaField, resolver: Callable) -> bool:
        """Add subscription to schema."""
        try:
            if field.name in self.subscriptions:
                return False

            self.subscriptions[field.name] = field
            self.resolvers[f"subscription_{field.name}"] = resolver
            return True

        except Exception as e:
            logger.error("Add subscription error: %s", e)
            return False

    def build_schema(self) -> Schema:
        """Build GraphQL schema."""
        try:
            # Build types
            type_defs = {}
            for type_name, type_def in self.types.items():
                fields = {}

                for field in type_def.fields or []:
                    field_type = self._get_field_type(field)

                    if field.args:
                        args = {
                            name: self._get_field_type(
                                SchemaField(name=name, type=arg_type)
                            )
                            for name, arg_type in field.args.items()
                        }

                        fields[field.name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            args=args,
                        )
                    else:
                        fields[field.name] = strawberry.field(
                            type=field_type, description=field.description
                        )

                if type_def.is_input:
                    type_defs[type_name] = strawberry.input(
                        type(
                            type_name,
                            (),
                            {
                                "__annotations__": fields,
                                "__doc__": type_def.description,
                            },
                        )
                    )
                else:
                    type_defs[type_name] = strawberry.type(
                        type(
                            type_name,
                            (),
                            {
                                "__annotations__": fields,
                                "__doc__": type_def.description,
                            },
                        )
                    )

            # Build Query type
            if self.queries:
                query_fields = {}

                for field_name, field in self.queries.items():
                    resolver = self.resolvers[f"query_{field_name}"]
                    field_type = self._get_field_type(field)

                    if field.args:
                        args = {
                            name: self._get_field_type(
                                SchemaField(name=name, type=arg_type)
                            )
                            for name, arg_type in field.args.items()
                        }

                        query_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            args=args,
                            resolver=resolver,
                        )
                    else:
                        query_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            resolver=resolver,
                        )

                Query = type(
                    "Query",
                    (),
                    {
                        "__annotations__": query_fields,
                        "__doc__": "Query root type",
                    },
                )
                Query = strawberry.type(Query)
            else:
                Query = None

            # Build Mutation type
            if self.mutations:
                mutation_fields = {}

                for field_name, field in self.mutations.items():
                    resolver = self.resolvers[f"mutation_{field_name}"]
                    field_type = self._get_field_type(field)

                    if field.args:
                        args = {
                            name: self._get_field_type(
                                SchemaField(name=name, type=arg_type)
                            )
                            for name, arg_type in field.args.items()
                        }

                        mutation_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            args=args,
                            resolver=resolver,
                        )
                    else:
                        mutation_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            resolver=resolver,
                        )

                Mutation = type(
                    "Mutation",
                    (),
                    {
                        "__annotations__": mutation_fields,
                        "__doc__": "Mutation root type",
                    },
                )
                Mutation = strawberry.type(Mutation)
            else:
                Mutation = None

            # Build Subscription type
            if self.subscriptions:
                subscription_fields = {}

                for field_name, field in self.subscriptions.items():
                    resolver = self.resolvers[f"subscription_{field_name}"]
                    field_type = self._get_field_type(field)

                    if field.args:
                        args = {
                            name: self._get_field_type(
                                SchemaField(name=name, type=arg_type)
                            )
                            for name, arg_type in field.args.items()
                        }

                        subscription_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            args=args,
                            resolver=resolver,
                        )
                    else:
                        subscription_fields[field_name] = strawberry.field(
                            type=field_type,
                            description=field.description,
                            resolver=resolver,
                        )

                Subscription = type(
                    "Subscription",
                    (),
                    {
                        "__annotations__": subscription_fields,
                        "__doc__": "Subscription root type",
                    },
                )
                Subscription = strawberry.type(Subscription)
            else:
                Subscription = None

            # Create schema
            return Schema(
                query=Query,
                mutation=Mutation,
                subscription=Subscription,
                types=list(type_defs.values()),
            )

        except Exception as e:
            logger.error("Build schema error: %s", e)
            return None

    def _get_field_type(self, field: SchemaField) -> Any:
        """Get field type."""
        try:
            # Get base type
            if field.type in self.types:
                base_type = self.types[field.type]
            else:
                base_type = getattr(strawberry, field.type.lower())

            # Apply modifiers
            if field.is_list:
                base_type = List[base_type]

            if field.is_required:
                base_type = strawberry.Required[base_type]

            return base_type

        except Exception as e:
            logger.error("Get field type error: %s", e)
            return None


class GraphQLService:
    """GraphQL management service."""

    def __init__(self):
        """Initialize GraphQL service."""
        self.builder = SchemaBuilder()
        self.schema = None

    def add_type(
        self,
        name: str,
        fields: List[Dict[str, Any]],
        description: Optional[str] = None,
        interfaces: Optional[List[str]] = None,
        is_input: bool = False,
    ) -> Dict[str, Any]:
        """Add type to schema."""
        try:
            schema_fields = []

            for field_def in fields:
                field = SchemaField(
                    name=field_def["name"],
                    type=field_def["type"],
                    description=field_def.get("description"),
                    args=field_def.get("args"),
                    is_list=field_def.get("is_list", False),
                    is_required=field_def.get("is_required", False),
                    default_value=field_def.get("default_value"),
                )
                schema_fields.append(field)

            type_def = SchemaType(
                name=name,
                description=description,
                fields=schema_fields,
                interfaces=interfaces,
                is_input=is_input,
            )

            success = self.builder.add_type(type_def)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Type added successfully"
                    if success
                    else "Failed to add type"
                ),
            }

        except Exception as e:
            logger.error("Add type error: %s", e)
            return {"status": "error", "message": str(e)}

    def add_query(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        description: Optional[str] = None,
        args: Optional[Dict[str, str]] = None,
        is_list: bool = False,
        is_required: bool = False,
    ) -> Dict[str, Any]:
        """Add query to schema."""
        try:
            field = SchemaField(
                name=name,
                type=return_type,
                description=description,
                args=args,
                is_list=is_list,
                is_required=is_required,
            )

            success = self.builder.add_query(field, resolver)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Query added successfully"
                    if success
                    else "Failed to add query"
                ),
            }

        except Exception as e:
            logger.error("Add query error: %s", e)
            return {"status": "error", "message": str(e)}

    def add_mutation(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        description: Optional[str] = None,
        args: Optional[Dict[str, str]] = None,
        is_list: bool = False,
        is_required: bool = False,
    ) -> Dict[str, Any]:
        """Add mutation to schema."""
        try:
            field = SchemaField(
                name=name,
                type=return_type,
                description=description,
                args=args,
                is_list=is_list,
                is_required=is_required,
            )

            success = self.builder.add_mutation(field, resolver)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Mutation added successfully"
                    if success
                    else "Failed to add mutation"
                ),
            }

        except Exception as e:
            logger.error("Add mutation error: %s", e)
            return {"status": "error", "message": str(e)}

    def add_subscription(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        description: Optional[str] = None,
        args: Optional[Dict[str, str]] = None,
        is_list: bool = False,
        is_required: bool = False,
    ) -> Dict[str, Any]:
        """Add subscription to schema."""
        try:
            field = SchemaField(
                name=name,
                type=return_type,
                description=description,
                args=args,
                is_list=is_list,
                is_required=is_required,
            )

            success = self.builder.add_subscription(field, resolver)

            return {
                "status": "success" if success else "error",
                "message": (
                    "Subscription added successfully"
                    if success
                    else "Failed to add subscription"
                ),
            }

        except Exception as e:
            logger.error("Add subscription error: %s", e)
            return {"status": "error", "message": str(e)}

    def build_schema(self) -> Dict[str, Any]:
        """Build GraphQL schema."""
        try:
            schema = self.builder.build_schema()

            if not schema:
                return {"status": "error", "message": "Failed to build schema"}

            self.schema = schema

            return {
                "status": "success",
                "message": "Schema built successfully",
            }

        except Exception as e:
            logger.error("Build schema error: %s", e)
            return {"status": "error", "message": str(e)}

    async def execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute GraphQL query."""
        try:
            if not self.schema:
                return {"status": "error", "message": "Schema not built"}

            result = await self.schema.execute(
                query, variable_values=variables, context_value=context
            )

            return {
                "status": "success",
                "data": result.data,
                "errors": [str(error) for error in result.errors]
                if result.errors
                else None,
            }

        except Exception as e:
            logger.error("Execute query error: %s", e)
            return {"status": "error", "message": str(e)}
