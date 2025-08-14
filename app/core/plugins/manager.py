"""
Plugin manager.
Manages loading and initialization of plugins.
"""

import importlib.util
import os
from typing import Dict, List, Optional

from app.core.error_handling import (ErrorCategory, ErrorSeverity,
                                     ReputationError)
from app.core.plugins.base import BasePlugin, PluginMetadata, PluginType


class PluginManager:
    """Plugin manager class."""

    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {}

    async def load_plugins(self, plugin_dir: str) -> None:
        """Load plugins from directory."""
        try:
            # Create plugin directory if it doesn't exist
            if not os.path.exists(plugin_dir):
                os.makedirs(plugin_dir)

            # Load each plugin file
            for filename in os.listdir(plugin_dir):
                if not filename.endswith(".py"):
                    continue

                plugin_path = os.path.join(plugin_dir, filename)
                plugin_name = filename[:-3]  # Remove .py extension

                try:
                    # Load plugin module
                    spec = importlib.util.spec_from_file_location(
                        plugin_name, plugin_path)
                    if spec is None:
                        raise ReputationError(
                            message=f"Could not load plugin spec: {plugin_name}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS)

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find plugin class
                    plugin_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type) and
                            issubclass(attr, BasePlugin) and
                            attr != BasePlugin
                        ):
                            plugin_class = attr
                            break

                    if plugin_class is None:
                        raise ReputationError(
                            message=f"No plugin class found in {plugin_name}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS
                        )

                    # Initialize plugin
                    plugin = plugin_class({})
                    metadata = plugin.get_metadata()

                    # Store plugin
                    self.plugins[metadata.name] = plugin

                    # Add to type index
                    if metadata.type not in self.plugin_types:
                        self.plugin_types[metadata.type] = []
                    self.plugin_types[metadata.type].append(metadata.name)

                except Exception as e:
                    raise ReputationError(
                        message=f"Error loading plugin {plugin_name}: {
                            str(e)}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.BUSINESS)

        except Exception as e:
            raise ReputationError(
                message=f"Error loading plugins: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )

    async def initialize_plugins(self) -> None:
        """Initialize all loaded plugins."""
        try:
            for plugin_name, plugin in self.plugins.items():
                try:
                    if not await plugin.initialize():
                        raise ReputationError(
                            message=f"Failed to initialize plugin: {plugin_name}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS)
                except Exception as e:
                    raise ReputationError(
                        message=f"Error initializing plugin {plugin_name}: {
                            str(e)}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.BUSINESS)
        except Exception as e:
            raise ReputationError(
                message=f"Error initializing plugins: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )

    async def shutdown_plugins(self) -> None:
        """Shutdown all loaded plugins."""
        try:
            for plugin_name, plugin in self.plugins.items():
                try:
                    if not await plugin.shutdown():
                        raise ReputationError(
                            message=f"Failed to shutdown plugin: {plugin_name}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.BUSINESS)
                except Exception as e:
                    raise ReputationError(
                        message=f"Error shutting down plugin {plugin_name}: {
                            str(e)}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.BUSINESS)
        except Exception as e:
            raise ReputationError(
                message=f"Error shutting down plugins: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.BUSINESS
            )

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get plugin by name."""
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get plugins by type."""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names]

    def get_plugin_metadata(
            self,
            plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        plugin = self.get_plugin(plugin_name)
        return plugin.get_metadata() if plugin else None
