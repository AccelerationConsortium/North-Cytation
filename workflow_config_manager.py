#!/usr/bin/env python3
"""
Workflow Configuration Manager

Manages configuration files for North Robotics workflows.
- Automatically creates/loads workflow-specific config files from workflow_configs/ folder
- Supports YAML format for human-readable configuration
- Provides config validation and fallback to defaults

Usage:
    from workflow_config_manager import ConfigManager
    
    # Define current workflow defaults
    defaults = {
        'SIMULATE': True,
        'VALIDATE_LIQUIDS': False,
        'SURFACTANT_A': 'SDS',
        # ... other config variables
    }
    
    # Get or create configuration
    config = ConfigManager.get_or_create_config('surfactant_grid_adaptive', defaults)
    
    # Use configuration values
    SIMULATE = config['SIMULATE']
    SURFACTANT_A = config['SURFACTANT_A']
"""

import os
import yaml
from datetime import datetime
from pathlib import Path
import logging

class ConfigManager:
    """Manages workflow configuration files with automatic creation and validation."""
    
    CONFIG_DIR = "workflow_configs"
    
    @classmethod
    def get_or_create_config(cls, workflow_name, default_config, logger=None):
        """
        Get existing config or create new one with defaults.
        
        Args:
            workflow_name (str): Name of the workflow (will be used as filename)
            default_config (dict): Default configuration values
            logger: Optional logger instance
            
        Returns:
            dict: Configuration values loaded from file or defaults
        """
        # Ensure config directory exists
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        
        # Generate config file path
        config_file = os.path.join(cls.CONFIG_DIR, f"{workflow_name}.yaml")
        
        if logger:
            logger.info(f"Config Manager: Checking for workflow config: {config_file}")
        else:
            print(f"Config Manager: Checking for workflow config: {config_file}")
        
        # Check if config file exists
        if os.path.exists(config_file):
            # Load existing configuration
            try:
                config = cls._load_config_file(config_file, logger)
                
                # Check for bypass flag - skip merging if user wants file-only config
                if config.get('_USE_FILE_ONLY', False):
                    if logger:
                        logger.info(f"Config Manager: _USE_FILE_ONLY=true, using config file as-is without merging defaults")
                    else:
                        print(f"Config Manager: _USE_FILE_ONLY=true, using config file as-is without merging defaults")
                    return config
                
                # Validate and merge with defaults (in case new keys were added)
                validated_config = cls._validate_and_merge_config(config, default_config, logger)
                
                # Save back if we added new keys
                if len(validated_config) > len(config):
                    cls._save_config_file(config_file, validated_config, logger)
                    if logger:
                        logger.info(f"Config Manager: Updated config file with {len(validated_config) - len(config)} new parameters")
                
                if logger:
                    logger.info(f"Config Manager: Loaded existing config with {len(validated_config)} parameters")
                else:
                    print(f"Config Manager: Loaded existing config with {len(validated_config)} parameters")
                
                return validated_config
                
            except Exception as e:
                if logger:
                    logger.error(f"Config Manager: Failed to load {config_file}: {e}")
                    logger.info("Config Manager: Creating new config file with defaults")
                else:
                    print(f"Config Manager: Failed to load {config_file}: {e}")
                    print("Config Manager: Creating new config file with defaults")
        
        # Create new config file with defaults
        cls._save_config_file(config_file, default_config, logger)
        
        if logger:
            logger.info(f"Config Manager: Created new config file: {config_file}")
        else:
            print(f"Config Manager: Created new config file: {config_file}")
        
        return default_config.copy()
    
    @classmethod
    def _load_config_file(cls, config_file, logger=None):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            return config
        except Exception as e:
            if logger:
                logger.error(f"Config Manager: Error loading {config_file}: {e}")
            raise
    
    @classmethod
    def _save_config_file(cls, config_file, config, logger=None):
        """Save configuration to YAML file with descriptive header."""
        try:
            with open(config_file, 'w') as f:
                # Write header comment
                f.write(f"# Workflow Configuration File\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Workflow: {Path(config_file).stem}\n")
                f.write(f"#\n")
                f.write(f"# Edit this file to customize workflow behavior.\n")
                f.write(f"# Changes will be loaded automatically on next run.\n")
                f.write(f"#\n\n")
                
                # Write configuration with nice formatting
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
                
        except Exception as e:
            if logger:
                logger.error(f"Config Manager: Error saving {config_file}: {e}")
            raise
    
    @classmethod
    def _validate_and_merge_config(cls, loaded_config, default_config, logger=None):
        """
        Validate loaded config and merge with defaults for any missing keys.
        
        Args:
            loaded_config (dict): Configuration loaded from file
            default_config (dict): Default configuration values
            logger: Optional logger
            
        Returns:
            dict: Validated and merged configuration
        """
        merged_config = default_config.copy()
        
        # Override defaults with loaded values
        for key, value in loaded_config.items():
            if key in default_config:
                # Validate type consistency (optional - could be more strict)
                default_type = type(default_config[key])
                loaded_type = type(value)
                
                if default_type != loaded_type and logger:
                    logger.warning(f"Config Manager: Type mismatch for '{key}': expected {default_type.__name__}, got {loaded_type.__name__}")
                
                merged_config[key] = value
            else:
                # Unknown key - keep it but warn
                merged_config[key] = value
                if logger:
                    logger.warning(f"Config Manager: Unknown config key '{key}' - keeping value but consider updating defaults")
        
        return merged_config
    
    @classmethod
    def update_config(cls, workflow_name, updates, logger=None):
        """
        Update specific configuration values and save back to file.
        
        Args:
            workflow_name (str): Name of the workflow
            updates (dict): Dictionary of key-value pairs to update
            logger: Optional logger
        """
        config_file = os.path.join(cls.CONFIG_DIR, f"{workflow_name}.yaml")
        
        if os.path.exists(config_file):
            current_config = cls._load_config_file(config_file, logger)
        else:
            current_config = {}
        
        # Update with new values
        current_config.update(updates)
        
        # Save back to file
        cls._save_config_file(config_file, current_config, logger)
        
        if logger:
            logger.info(f"Config Manager: Updated {len(updates)} config parameters in {workflow_name}")
        else:
            print(f"Config Manager: Updated {len(updates)} config parameters in {workflow_name}")
        
        return current_config

    @classmethod
    def setup_workflow_config(cls, workflow_name, workflow_globals, logger=None):
        """
        Automatically detect constants from workflow globals() and create/load config.
        
        Args:
            workflow_name (str): Name of the workflow (will be used as filename)
            workflow_globals (dict): Result from globals() call in workflow
            logger: Optional logger instance
            
        Returns:
            dict: Configuration values loaded from file or detected defaults
        """
        # Automatically filter globals for workflow constants
        workflow_constants = {}
        for key, value in workflow_globals.items():
            if key.isupper() and not key.startswith('_') and not callable(value):
                # Only include constants (uppercase, not private, not functions)
                if isinstance(value, (str, int, float, bool, list, dict)):
                    workflow_constants[key] = value
        
        if logger:
            logger.info(f"Config Manager: Auto-detected {len(workflow_constants)} constants from workflow globals")
        else:
            print(f"Config Manager: Auto-detected {len(workflow_constants)} constants from workflow globals")
        
        # Use existing get_or_create_config method
        return cls.get_or_create_config(workflow_name, workflow_constants, logger)

    @classmethod
    def load_and_update_globals(cls, workflow_name, target_globals, logger=None):
        """
        Load config from file and update only the detected constants in globals.
        
        Args:
            workflow_name (str): Name of the workflow
            target_globals (dict): The globals() dict to update
            logger: Optional logger instance
            
        Returns:
            dict: The loaded configuration
        """
        config_file = os.path.join(cls.CONFIG_DIR, f"{workflow_name}.yaml")
        
        if not os.path.exists(config_file):
            if logger:
                logger.warning(f"Config Manager: No config file found at {config_file}")
            else:
                print(f"Config Manager: No config file found at {config_file}")
            return {}
        
        try:
            # Load current config from file
            config = cls._load_config_file(config_file, logger)
            
            # Only update globals for detected constants (uppercase, not private, not callable)
            updated_count = 0
            for key, value in config.items():
                if key in target_globals and key.isupper() and not key.startswith('_'):
                    target_globals[key] = value
                    updated_count += 1
            
            if logger:
                logger.info(f"Config Manager: Updated {updated_count} workflow constants from {workflow_name}.yaml")
            else:
                print(f"Config Manager: Updated {updated_count} workflow constants from {workflow_name}.yaml")
            
            return config
            
        except Exception as e:
            if logger:
                logger.error(f"Config Manager: Failed to load and update globals: {e}")
            else:
                print(f"Config Manager: Failed to load and update globals: {e}")
            return {}

    @classmethod  
    def setup_and_reload_config(cls, workflow_name, workflow_globals, logger=None):
        """
        One-line convenience method: setup config, then immediately reload from file.
        Useful when GUI might modify config between setup and reload.
        
        Args:
            workflow_name (str): Name of the workflow
            workflow_globals (dict): The globals() dict to detect constants and update
            logger: Optional logger instance
            
        Returns:
            dict: The final loaded configuration
        """
        # First setup (detects constants and creates file if needed)
        cls.setup_workflow_config(workflow_name, workflow_globals, logger)
        
        # Then reload from file (in case it was modified)
        return cls.load_and_update_globals(workflow_name, workflow_globals, logger)

    @classmethod
    def setup_config_if_missing(cls, workflow_name, workflow_globals, logger=None):
        """
        Create config file ONLY if it doesn't exist. If it exists, do nothing.
        Preserves user edits by never overwriting existing config files.
        
        Args:
            workflow_name (str): Name of the workflow
            workflow_globals (dict): The globals() dict to detect constants from
            logger: Optional logger instance
            
        Returns:
            bool: True if config was created, False if it already existed
        """
        config_file = os.path.join(cls.CONFIG_DIR, f"{workflow_name}.yaml")
        
        if os.path.exists(config_file):
            # Config exists - don't overwrite user changes
            if logger:
                logger.info(f"Config Manager: Config file exists, preserving user changes: {config_file}")
            else:
                print(f"Config Manager: Config file exists, preserving user changes: {config_file}")
            return False
        
        # Config doesn't exist - create it from detected constants
        workflow_constants = {}
        for key, value in workflow_globals.items():
            if key.isupper() and not key.startswith('_') and not callable(value):
                if isinstance(value, (str, int, float, bool, list, dict)):
                    workflow_constants[key] = value
        
        if logger:
            logger.info(f"Config Manager: Creating new config with {len(workflow_constants)} detected constants")
        else:
            print(f"Config Manager: Creating new config with {len(workflow_constants)} detected constants")
        
        # Create the config file
        cls._save_config_file(config_file, workflow_constants, logger)
        return True

    @classmethod
    def setup_and_load_config(cls, workflow_name, workflow_globals, logger=None):
        """
        One-line convenience: Setup config if missing, then load and update globals.
        Preserves user edits while ensuring config exists.
        
        Args:
            workflow_name (str): Name of the workflow
            workflow_globals (dict): The globals() dict to update
            logger: Optional logger instance
            
        Returns:
            dict: The loaded configuration
        """
        # Setup if missing (doesn't overwrite existing)
        cls.setup_config_if_missing(workflow_name, workflow_globals, logger)
        
        # Load from file and update globals
        return cls.load_and_update_globals(workflow_name, workflow_globals, logger)

# Convenience function for quick usage
def get_workflow_config(workflow_name, defaults, logger=None):
    """
    Convenience function to get workflow configuration.
    
    Args:
        workflow_name (str): Name of the workflow
        defaults (dict): Default configuration values
        logger: Optional logger
        
    Returns:
        dict: Configuration values
    """
    return ConfigManager.get_or_create_config(workflow_name, defaults, logger)


if __name__ == "__main__":
    # Example usage
    test_config = {
        'SIMULATE': True,
        'VALIDATE_LIQUIDS': False,
        'SURFACTANT_A': 'SDS',
        'SURFACTANT_B': 'TTAB',
        'MIN_CONC': 0.01,
        'NUMBER_CONCENTRATIONS': 9
    }
    
    config = ConfigManager.get_or_create_config('test_workflow', test_config)
    print("Loaded configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")