"""
ResearchRadar - Configuration Settings
An AI-powered research analysis tool.

Copyright (c) 2024 Pranav Rajpurkar. MIT License.
"""

from pathlib import Path
import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # User settings
    "AUTHOR_NAME": "Pranav Rajpurkar",  # The name of the author whose papers to focus on
    
    # Model settings
    "MODEL_NAME": "gpt-4o-mini",  # The OpenAI model to use
    "MODEL_TEMPERATURE": 0.1,  # Temperature for model responses
    
    # Directory settings
    "PDF_FOLDER": "pdfs",  # Directory containing PDF files to process
    "OUTPUT_DIR": "outputs",  # Directory for all outputs
    
    # Processing settings
    "NUM_TOPICS": 5,  # Number of research topics to cluster papers into
    "MAX_WORKERS": 16,  # Maximum number of parallel workers
    
    # Cache settings
    "CACHE_VERSION": "2.0",  # Version of the cache format
    
    # Output settings
    "CSV_OUTPUT": "outputs/research_summary.csv",  # Path for CSV summary output
}

def load_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load and validate configuration settings.
    
    Args:
        custom_config: Optional dictionary of custom configuration values
        
    Returns:
        Dict containing the final configuration
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Update with custom config if provided
    if custom_config:
        config.update(custom_config)
    
    # Ensure directories exist
    Path(config["PDF_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
            "in your .env file or environment."
        )
    
    # Validate other settings
    if config["NUM_TOPICS"] < 1:
        raise ValueError("NUM_TOPICS must be at least 1")
    
    if config["MAX_WORKERS"] < 1:
        raise ValueError("MAX_WORKERS must be at least 1")
    
    if not config["AUTHOR_NAME"]:
        raise ValueError("AUTHOR_NAME must be specified")
    
    return config

def get_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get standardized output paths based on configuration.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        Dict containing various output paths
    """
    output_dir = Path(config["OUTPUT_DIR"])
    return {
        "papers_dir": output_dir / "papers",
        "topics_dir": output_dir / "topics",
        "cache_dir": output_dir / "cache",
        "narrative_file": output_dir / "year_in_review_narrative.txt",
        "csv_file": Path(config["CSV_OUTPUT"]),
    } 