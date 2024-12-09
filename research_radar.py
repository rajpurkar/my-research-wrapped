# 1. Imports and Configuration
from __future__ import annotations
import glob
import os
import json
import hashlib
import sys
from pathlib import Path
import re
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import Counter, defaultdict
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import numpy as np
import logging
import traceback
from datetime import datetime
import shutil

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.traceback import install as install_rich_traceback
from rich.theme import Theme

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Import configuration
from config import DEFAULT_CONFIG, load_config, get_output_paths

class PaperAnalysis(TypedDict):
    title: str
    authors: List[str]
    summary: str
    weight: float

class ResearchArea(TypedDict):
    name: str
    paper_indices: List[int]

class ClusteringOutput(TypedDict):
    research_areas: List[ResearchArea]

# Configure logging
def setup_logging():
    """Configure rich-based logging with custom formatting and colors"""
    # Install rich traceback handler
    install_rich_traceback(show_locals=True)
    
    # Create custom theme for logging
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "debug": "dim cyan",
        "http": "magenta"
    })
    
    # Initialize rich console with theme
    console = Console(theme=custom_theme)
    
    # Configure rich logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=False,
            markup=True,
            log_time_format="[%X]"
        )]
    )
    
    # Set up logging levels for different components
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP noise
    logging.getLogger("openai").setLevel(logging.WARNING)  # Reduce OpenAI API noise
    
    logger = logging.getLogger("rich")
    return logger

logger = setup_logging()

# Initialize rich console for global use
console = Console()

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in .env file")

llm = ChatOpenAI(
    api_key=api_key,  # Pass key explicitly
    model="gpt-4o-mini",
    temperature=0.1
)

# Initialize LLM for author handling
name_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)

# At the top of the file, add these new constants for prompts
PROMPTS = {
    "NAME_NORMALIZATION": """
    Normalize this author name by following these rules EXACTLY:
    1. Keep ONLY first and last name, remove ALL middle names/initials
    2. Remove ALL titles (Dr., Prof., etc.) and suffixes (Jr., III, etc.)
    3. Convert ALL special characters to their basic form:
       - á,à,ä,â -> a
       - é,è,ë,ê -> e
       - í,ì,ï,î -> i
       - ó,ò,ö,ô -> o
       - ú,ù,ü,û -> u
       - ñ -> n
    4. MAINTAIN the EXACT capitalization from the original name
    5. Remove ALL extra spaces
    
    Original name: {name}
    Return ONLY the normalized name in the ORIGINAL case.
    """,
    
    "AUTHOR_EXTRACTION": """
    Extract the author names from this text. The names should be in their original casing.
    
    Requirements:
    1. Each name MUST have both first and last name components
    2. Keep original capitalization of names
    3. Remove all titles (Dr., Prof., PhD, etc.)
    4. Remove all affiliations and numbers
    5. Keep middle names/initials if present
    6. Preserve hyphenated names and special characters (é, ñ, etc.)
    
    Text: {text}
    Return ONLY the list of complete author names, one per line.
    """,
    
    "TITLE_EXTRACTION": """
    Extract the title of this research paper. The title might be:
    - At the beginning of the text
    - After keywords like "Title:" or "Paper:"
    - In a header or metadata section
    - In larger or bold font (though you can't see formatting)
    
    Text: {text}
    
    Return ONLY the title text, nothing else.
    If no clear title is found, return "Untitled".
    """,
    
    "PAPER_SUMMARY": """
    Provide a technical summary of this paper's key innovation and results:
    {text}
    
    Focus on:
    1. The specific research problem addressed
    2. The key technical innovation and methodological approach
       - If the paper introduces a named method/model/framework, highlight this name
       - Example: "We introduced MedBERT, a new language model for..."
    3. Main empirical findings and quantitative results
    
    Keep to 2-3 sentences that highlight the core technical contribution.
    Use "We" to describe our work, as this is authored by the researcher of interest.
    """,
    
    "CLUSTERING": """Group these papers into {num_topics} cohesive research areas that highlight key technical innovations and methodological advances.

Papers:
{papers}

Requirements:
1. Each research area MUST have AT LEAST 3 papers and no more than {max_papers} papers
2. Papers should be grouped based on shared technical approaches, methodologies, or research goals
3. Target number of papers per topic is {target_per_topic:.1f}
4. EVERY paper must be assigned to exactly one research area
5. Balance the distribution of papers across areas

Research Area Name Guidelines:
1. Create a specific, technical name that captures the shared innovation
2. Include both the methodology and its impact/application
3. Use active verbs and concrete technical terms
4. Highlight the transformative aspect of the work

Examples of good research area names:
- "Deep Learning Architectures Transform Visual Understanding"
- "Novel Validation Methods Enable Reliable AI Systems"
- "Multi-Modal Foundation Models Advance Decision Support"
- "Self-Supervised Learning Enhances Representation Quality"
- "Automated Analysis Systems Improve Data Processing"

Bad examples:
- "AI Applications" (too vague)
- "Computer Vision" (not specific to innovation)
- "Improving Performance" (not technical)
- "Deep_Learning_Research" (uses underscores)
- "Novel Methods" (not descriptive)

Return a list of research areas, where each area has:
1. A specific, technical name
2. A list of paper indices that belong to that area

IMPORTANT: Every paper index MUST appear exactly once across all areas. If a paper doesn't perfectly fit any area, assign it to the most related area.""",
    
    "TOPIC_SYNTHESIS": """Synthesize our contributions within this research area.

Papers:
{context}

Guidelines:
1. Present our work's role in advancing this technical area
2. Reference papers concisely by:
   - Method names when notable (e.g., "The model achieved...")
   - Brief descriptions (e.g., "Our approach to uncertainty quantification...")
   - Never use full paper titles
   - Never use chronological markers
3. Highlight key technical innovations and results
4. Show how different approaches complement each other
5. Use "we" and "our" naturally without being overly promotional
6. Write in a technical yet accessible style

Structure each synthesis to:
- Open with the area's technical significance
- Detail key methodological advances
- Present quantitative results and impact
- Show connections between different approaches
- Balance confidence with objectivity

Example:
"We developed novel approaches to uncertainty quantification that improve model reliability in high-stakes applications. Our calibration method achieved a 40% reduction in error rates while maintaining high performance. By combining this with robust validation techniques, we enabled more reliable deployment across diverse settings..."

Keep the tone confident but grounded, focusing on concrete technical achievements.""",
    
    "OVERALL_NARRATIVE": """Create an introductory narrative that connects our research across these areas: {themes}

Guidelines:
1. Frame our work's contribution to advancing the field
2. Present our innovations in addressing key technical challenges
3. Show how research areas complement each other
4. Balance enthusiasm with objectivity

Key Style Elements:
- Use first-person plural ("we", "our") naturally
- Focus on concrete technical achievements
- Connect advances to real challenges
- Maintain technical precision
- Balance confidence with humility

Example Style:
"Our research focuses on developing robust and adaptable AI systems that can handle complex real-world tasks. Through advances in uncertainty quantification and multi-modal learning, we've developed approaches that help bridge the gap between specialized models and flexible reasoning.

Our work spans several technical areas, from uncertainty estimation in high-stakes applications to integrating diverse data types. These complementary advances work together to improve the reliability and capability of AI systems..."

Keep the narrative focused on concrete technical achievements while maintaining accessibility."""
}

# Base Classes
class FileManager:
    """Base class for file operations with common utilities."""
    
    def __init__(self, base_dir: Path):
        """Initialize with base directory path.
        
        Args:
            base_dir: Path to the base directory for all outputs
        """
        # Convert to absolute path and resolve any symlinks
        self.base_dir = Path(base_dir).resolve()
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)
    
    def get_path(self, filename: str | Path) -> Path:
        """Get full path for a file."""
        path = Path(filename)
        # If path is absolute, return it directly
        if path.is_absolute():
            return path
        # Otherwise join with base_dir
        return self.base_dir / path
        
    def save_json(self, filename: str | Path, data: dict):
        """Save data as JSON."""
        path = self.get_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_json(self, filename: str | Path, default: dict = None) -> dict:
        """Load JSON data with version checking."""
        path = self.get_path(filename)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if data.get("version") != DEFAULT_CONFIG["CACHE_VERSION"]:
                    return default or {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
                return data
            except (json.JSONDecodeError, KeyError) as e:
                console.print(f"[red]Failed to load {filename}: {e}[/red]")
                return default or {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
        return default or {"version": DEFAULT_CONFIG["CACHE_VERSION"]}

class CacheManager:
    """Handles all caching operations."""
    
    def __init__(self, manager: 'ResearchSummaryManager'):
        self.manager = manager
    
    def get_cache_key(self, pdf_path: str) -> str:
        """Generate a cache key based on file path and last modified time."""
        pdf_stats = os.stat(pdf_path)
        content_key = f"{pdf_path}:{pdf_stats.st_mtime}"
        return hashlib.md5(content_key.encode()).hexdigest()
    
    def is_cached(self, pdf_path: str) -> bool:
        """Check if a paper is cached and valid."""
        paper_hash = self.get_cache_key(pdf_path)
        cache_file = self.manager.get_paper_cache_path(paper_hash)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                return cache_data.get("version") == DEFAULT_CONFIG["CACHE_VERSION"]
            except (json.JSONDecodeError, KeyError):
                return False
        return False
    
    def get_cache(self, pdf_path: str) -> dict:
        """Get cache data for a paper."""
        paper_hash = self.get_cache_key(pdf_path)
        cache_file = self.manager.get_paper_cache_path(paper_hash)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                if cache_data.get("version") == DEFAULT_CONFIG["CACHE_VERSION"]:
                    return cache_data
            except (json.JSONDecodeError, KeyError):
                pass
        return {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
    
    def set_cache(self, pdf_path: str, cache_data: dict):
        """Set cache data for a paper."""
        paper_hash = self.get_cache_key(pdf_path)
        cache_file = self.manager.get_paper_cache_path(paper_hash)
        
        cache_data["version"] = DEFAULT_CONFIG["CACHE_VERSION"]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=4)
        except Exception as e:
            console.print(f"[yellow]Failed to save cache for {Path(pdf_path).name}: {e}[/yellow]")

class ResearchSummaryManager(FileManager):
    """Manages all research summary data and operations."""
    
    def __init__(self, output_dir: str = DEFAULT_CONFIG["OUTPUT_DIR"]):
        """Initialize with output directory structure."""
        super().__init__(Path(output_dir))  # Initialize FileManager first
        
        # Use self.base_dir from FileManager instead of creating new Path
        self.papers_dir = self.base_dir / "papers"
        self.topics_dir = self.base_dir / "topics"
        self.narrative_path = self.base_dir / "year_in_review_narrative.txt"
        
        # Create directories
        for directory in [self.papers_dir, self.topics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
    def get_paper_path(self, paper_title: str) -> Path:
        """Get path for paper data and summary."""
        safe_name = re.sub(r'[^\w\s-]', '_', paper_title)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        safe_name = safe_name.strip('_').lower()
        safe_name = safe_name or 'unnamed_paper'
        return self.papers_dir / f"{safe_name}.json"
    
    def get_topic_path(self, topic_name: str) -> Path:
        """Get path for topic data with proper filename sanitization."""
        safe_name = re.sub(r'[^\w\s-]', '_', topic_name)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        safe_name = safe_name or 'unnamed_topic'
        return self.topics_dir / f"{safe_name}.json"

# Processing Classes
class LLMProcessor:
    """Handles all LLM interactions."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # Keep prompts as raw strings
        self.prompts = {
            "COMBINED_PAPER_ANALYSIS": """Analyze this academic paper and extract the following information in a structured format:
1. Title: Extract the paper title
2. Authors: List all authors with both first and last names
3. Summary: Provide a focused technical summary that:
   - Identifies the specific research problem addressed
   - Describes the key technical innovation and methodological approach
   - Highlights main empirical findings and quantitative results
   - Uses 2-3 clear, concise sentences
4. Impact Weight: Score from 0-1 based on paper significance

Text: {text}""",
            "NAME_NORMALIZATION": """Normalize these author names by following these rules EXACTLY:
1. Keep ONLY first and last name, remove ALL middle names/initials
2. Remove ALL titles (Dr., Prof., etc.) and suffixes (Jr., III, etc.)
3. Convert ALL special characters to their basic form:
   - á,à,ä,â -> a
   - é,è,ë,ê -> e
   - í,ì,ï,î -> i
   - ó,ò,ö,ô -> o
   - ú,ù,ü,û -> u
   - ñ -> n
4. MAINTAIN the EXACT capitalization from the original names
5. Remove ALL extra spaces

Original names:
{name}

Return ONLY the normalized names, one per line.""",
            "CLUSTERING": """Group these papers into {num_topics} cohesive research areas that highlight key technical innovations and methodological advances.

Papers:
{papers}

Requirements:
1. Each research area MUST have AT LEAST 3 papers and no more than {max_papers} papers
2. Papers should be grouped based on shared technical approaches, methodologies, or research goals
3. Target number of papers per topic is {target_per_topic:.1f}
4. EVERY paper must be assigned to exactly one research area
5. Balance the distribution of papers across areas

Research Area Name Guidelines:
1. Create a specific, technical name that captures the shared innovation
2. Include both the methodology and its impact/application
3. Use active verbs and concrete technical terms
4. Highlight the transformative aspect of the work

Examples of good research area names:
- "Deep Learning Architectures Transform Visual Understanding"
- "Novel Validation Methods Enable Reliable AI Systems"
- "Multi-Modal Foundation Models Advance Decision Support"
- "Self-Supervised Learning Enhances Representation Quality"
- "Automated Analysis Systems Improve Data Processing"

Bad examples:
- "AI Applications" (too vague)
- "Computer Vision" (not specific to innovation)
- "Improving Performance" (not technical)
- "Deep_Learning_Research" (uses underscores)
- "Novel Methods" (not descriptive)

Return a list of research areas, where each area has:
1. A specific, technical name
2. A list of paper indices that belong to that area

IMPORTANT: Every paper index MUST appear exactly once across all areas. If a paper doesn't perfectly fit any area, assign it to the most related area.""",
            "TOPIC_SYNTHESIS": """Synthesize our contributions within this research area.

Papers:
{context}

Guidelines:
1. Present our work's role in advancing this technical area
2. Reference papers concisely by:
   - Method names when notable (e.g., "The model achieved...")
   - Brief descriptions (e.g., "Our approach to uncertainty quantification...")
   - Never use full paper titles
   - Never use chronological markers
3. Highlight key technical innovations and results
4. Show how different approaches complement each other
5. Use "we" and "our" naturally without being overly promotional
6. Write in a technical yet accessible style

Structure each synthesis to:
- Open with the area's technical significance
- Detail key methodological advances
- Present quantitative results and impact
- Show connections between different approaches
- Balance confidence with objectivity

Example:
"We developed novel approaches to uncertainty quantification that improve model reliability in high-stakes applications. Our calibration method achieved a 40% reduction in error rates while maintaining high performance. By combining this with robust validation techniques, we enabled more reliable deployment across diverse settings..."

Keep the tone confident but grounded, focusing on concrete technical achievements.""",
            "OVERALL_NARRATIVE": """Create an introductory narrative that connects our research across these areas: {themes}

Guidelines:
1. Frame our work's contribution to advancing the field
2. Present our innovations in addressing key technical challenges
3. Show how research areas complement each other
4. Balance enthusiasm with objectivity

Key Style Elements:
- Use first-person plural ("we", "our") naturally
- Focus on concrete technical achievements
- Connect advances to real challenges
- Maintain technical precision
- Balance confidence with humility

Keep the narrative focused on concrete technical achievements while maintaining accessibility."""
        }
        # Create prompt templates
        self.prompts = {k: ChatPromptTemplate.from_template(v) for k, v in self.prompts.items()}
    
    def invoke(self, prompt_key: str, **kwargs) -> str:
        """Invoke LLM with error handling."""
        if prompt_key not in self.prompts:
            logger.error(f"Prompt key '{prompt_key}' not found in available prompts: {list(self.prompts.keys())}")
            raise KeyError(f"Invalid prompt key: {prompt_key}")
            
        try:
            prompt = self.prompts[prompt_key]
            if prompt_key == "COMBINED_PAPER_ANALYSIS":
                # Use structured output for paper analysis
                structured_llm = self.llm.with_structured_output(PaperAnalysis)
                result = structured_llm.invoke(prompt.format(**kwargs))
                return result  # Returns PaperAnalysis dict
            else:
                # Regular string output for other prompts
                result = self.llm.invoke(prompt.format(**kwargs))
                return result.content.strip()
        except Exception as e:
            logger.error(f"Error in LLM call for {prompt_key}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class DocumentProcessor:
    """Handles document loading and text extraction."""
    
    @staticmethod
    def load_pdf(pdf_path: str) -> Tuple[str, List[Document]]:
        """Load PDF and extract text."""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        text = "\n".join([doc.page_content for doc in docs]) if docs else ""
        return text, docs
    
    @staticmethod
    def extract_metadata(
        text: str,
        llm_processor: LLMProcessor
    ) -> Tuple[str, List[str]]:
        """Extract title and authors from text."""
        title = llm_processor.invoke_with_retry("TITLE_EXTRACTION", text=text[:2000])
        author_text = llm_processor.invoke_with_retry("AUTHOR_EXTRACTION", text=text[:2000])
        return title, [name.strip() for name in author_text.split('\n') if name.strip()]

class ParallelProcessor:
    """Handles parallel processing with consistent error handling and progress tracking."""
    
    def __init__(self, max_workers: int = DEFAULT_CONFIG["MAX_WORKERS"], llm_processor: Optional[LLMProcessor] = None, cache_manager: Optional[CacheManager] = None):
        self.max_workers = max_workers
        self.llm_processor = llm_processor
        self.cache_manager = cache_manager
    
    def process_items(
        self,
        items: List[Any],
        process_fn: Callable,
        status_msg: str,
        error_msg: str,
        status=None,
        **kwargs
    ) -> List[Any]:
        """Process items in parallel with consistent error handling and progress tracking."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if status:
                status.update(f"[bold blue]{status_msg}[/bold blue]")
            
            # Submit all tasks
            future_to_item = {
                executor.submit(process_fn, item, **kwargs): item
                for item in items
            }
            
            # Process results as they complete
            completed = 0
            total = len(items)
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if status:
                        status.update(f"[dim]Completed {completed}/{total} items[/dim]")
                except Exception as e:
                    console.print(f"[red]{error_msg} {str(e)}[/red]")
        
        return results

class SummaryGenerator:
    """Handles generation of all types of summaries."""
    
    def __init__(self, llm_processor: LLMProcessor):
        self.llm = llm_processor
    
    def create_paper_summary(
        self,
        paper: Paper,
        cache_key: Optional[str] = None
    ) -> PaperSummary:
        """Create a complete paper summary."""
        brief = self.llm.invoke_with_retry(
            "BRIEF_SUMMARY",
            summary=paper.summary
        )
        technical = self.llm.invoke_with_retry(
            "PAPER_SUMMARY",
            summary=paper.summary
        )
        return PaperSummary(
            paper=paper,
            brief_summary=brief,
            technical_summary=technical,
            weight=paper.weight
        )
    
    def create_topic_summary(
        self,
        name: str,
        papers: List[PaperSummary]
    ) -> TopicSummary:
        """Create a complete topic summary."""
        papers_text = "\n\n".join([
            f"Title: {p.paper.title}\nWeight: {p.weight:.1f}\nSummary: {p.technical_summary}"
            for p in sorted(papers, key=lambda x: x.weight, reverse=True)
        ])
        
        narrative = self.llm.invoke_with_retry(
            "TOPIC_NARRATIVE",
            context=papers_text
        )
        context = self.llm.invoke_with_retry(
            "TOPIC_CONTEXT",
            theme=name
        )
        
        return TopicSummary(
            name=name,
            paper_summaries=papers,
            flowing_narrative=narrative,
            context_summary=context
        )

# Data Classes
@dataclass
class Author:
    """Represents an author with normalized name handling."""
    full_name: str
    normalized_name: str
    role: str = 'contributing_author'
    
    @classmethod
    def create(cls, name: str, llm_processor: 'LLMProcessor') -> 'Author':
        """Create an Author instance with normalized name."""
        normalized = llm_processor.invoke_with_retry("NAME_NORMALIZATION", name=name)
        return cls(full_name=name, normalized_name=normalized)
    
    def matches(self, other_name: str, llm_processor: 'LLMProcessor') -> bool:
        """Check if this author matches another name."""
        other_normalized = llm_processor.invoke_with_retry("NAME_NORMALIZATION", name=other_name)
        return (
            self.normalized_name.lower() == other_normalized.lower() or
            self.full_name.lower() == other_name.lower()
        )

@dataclass
class Paper:
    """Represents a processed research paper."""
    file_path: str
    title: str
    authors: List[Author]
    summary: str  # Single focused technical summary
    weight: float
    role: str
    original_text: str = ""
    processed_time: Optional[str] = None
    
    @classmethod
    def from_cache(cls, cache_data: dict, pdf_file: str) -> 'Paper':
        """Create Paper instance from cache data."""
        authors = [
            Author(
                full_name=a["full_name"],
                normalized_name=a["normalized_name"]
            )
            for a in cache_data.get("authors", [])
        ]
        return cls(
            file_path=pdf_file,
            title=cache_data.get("title", "Untitled"),
            authors=authors,
            summary=cache_data["summary"],
            weight=cache_data.get("weight", 0.1),
            role=cache_data.get("role", "unknown"),
            original_text=cache_data.get("original_text", ""),
            processed_time=str(Path(pdf_file).stat().st_mtime)
        )
    
    @classmethod
    def create(cls, pdf_file: str, **kwargs) -> 'Paper':
        """Create new Paper instance with proper defaults."""
        return cls(
            file_path=pdf_file,
            processed_time=str(Path(pdf_file).stat().st_mtime),
            **kwargs
        )
    
    def to_cache_dict(self) -> dict:
        """Convert to cache format."""
        return {
            "title": self.title,
            "authors": [
                {
                    "full_name": a.full_name,
                    "normalized_name": a.normalized_name
                }
                for a in self.authors
            ],
            "summary": self.summary,
            "weight": self.weight,
            "role": self.role,
            "original_text": self.original_text,
            "version": DEFAULT_CONFIG["CACHE_VERSION"]
        }

@dataclass
class PaperSummary:
    """Represents a paper's summary."""
    paper: Paper
    summary: str
    weight: float
    
    def to_dict(self) -> dict:
        """Convert to serializable format."""
        return {
            "file_path": self.paper.file_path,
            "title": self.paper.title,
            "summary": self.summary,
            "weight": self.weight,
            "authors": [
                {
                    "full_name": a.full_name,
                    "normalized_name": a.normalized_name
                }
                for a in self.paper.authors
            ]
        }

@dataclass
class TopicSummary:
    """Represents a group of related papers with their narrative."""
    name: str
    paper_summaries: List[PaperSummary]
    narrative: str
    
    def to_dict(self) -> dict:
        """Convert to serializable format."""
        return {
            "name": self.name,
            "narrative": self.narrative,
            "paper_summaries": [ps.to_dict() for ps in self.paper_summaries]
        }

@dataclass
class TopicSynthesis:
    """Represents a synthesis of papers within a research topic."""
    name: str
    paper_summaries: List[PaperSummary]
    synthesis: str
    
    def to_dict(self) -> dict:
        """Convert to serializable format."""
        return {
            "name": self.name,
            "synthesis": self.synthesis,
            "paper_summaries": [ps.to_dict() for ps in self.paper_summaries]
        }

# Processing Functions
def parse_paper_analysis(text: str) -> Dict[str, Any]:
    """Parse the paper analysis from the model output.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        Parsed JSON object
        
    Raises:
        ValueError: If parsing fails
    """
    # First try parsing the raw text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Try extracting JSON between markdown code blocks
    try:
        # Look for content between ```json and ``` markers
        json_start = text.find("```json")
        if json_start != -1:
            json_start += 7  # Length of ```json
            json_end = text.find("```", json_start)
            if json_end != -1:
                json_str = text[json_start:json_end].strip()
                return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # Try extracting anything that looks like JSON
    try:
        # Look for content between { and }
        json_start = text.find("{")
        if json_start != -1:
            json_end = text.rfind("}") + 1
            if json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    raise ValueError("Failed to parse paper analysis result")

# Remove the Pydantic imports and class, and replace with JSON schema
PAPER_ANALYSIS_SCHEMA = {
    "title": "PaperAnalysis",
    "description": "Schema for the initial paper analysis",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the paper"
        },
        "authors": {
            "type": "array",
            "description": "List of author names as they appear in the paper",
            "items": {
                "type": "string"
            }
        },
        "technical_summary": {
            "type": "string",
            "description": "Detailed technical summary of the paper's content"
        },
        "brief_summary": {
            "type": "string", 
            "description": "Brief, high-level summary of the paper's main points"
        },
        "weight": {
            "type": "number",
            "description": "Importance weight from 0-1 based on paper impact and relevance",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": ["title", "authors", "technical_summary", "brief_summary", "weight"]
}

def process_pdf(
    pdf_file: str,
    llm_processor: LLMProcessor,
    manager: ResearchSummaryManager,
    status_display: Optional[StatusDisplay] = None
) -> Paper:
    """Process a single PDF file and return a Paper object."""
    logger.info(f"Starting processing of {pdf_file}")
    
    try:
        # Extract text from PDF
        logger.info(f"Extracting text from {pdf_file}")
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            logger.error(f"No text extracted from {pdf_file}")
            raise ValueError(f"No text could be extracted from {pdf_file}")
        
        logger.info(f"Text extracted successfully from {pdf_file} ({len(text)} chars)")

        # Single LLM call for combined analysis with full text
        analysis_data = llm_processor.invoke(
            "COMBINED_PAPER_ANALYSIS",
            text=text
        )
        # analysis_data is already a PaperAnalysis dict, no need to parse JSON
            
        # Create author objects (single batch of normalizations)
        author_names = analysis_data.get("authors", [])
        normalized_names = []
        if author_names:
            # Single LLM call to normalize all author names at once
            names_text = "\n".join(author_names)
            normalized_text = llm_processor.invoke(
                "NAME_NORMALIZATION",
                name=names_text
            )
            normalized_names = [name.strip() for name in normalized_text.split("\n") if name.strip()]
        
        authors = [
            Author(full_name=orig, normalized_name=norm)
            for orig, norm in zip(author_names, normalized_names)
        ]
        
        # Create paper object with single summary
        paper = Paper.create(
            pdf_file=pdf_file,
            title=analysis_data["title"],
            authors=authors,
            summary=analysis_data["summary"],
            weight=analysis_data["weight"],
            role="primary_research",
            original_text=text
        )
        
        # Save paper data
        paper_path = manager.get_paper_path(paper.title)
        paper_data = {
            "title": paper.title,
            "file_path": paper.file_path,
            "authors": [{"full_name": a.full_name, "normalized_name": a.normalized_name} for a in paper.authors],
            "summary": paper.summary,
            "weight": paper.weight,
            "role": paper.role
        }
        manager.save_json(paper_path, paper_data)
        
        logger.info(f"Successfully processed {pdf_file}")
        return paper
        
    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_papers_in_parallel(
    pdf_files: List[str],
    parallel_processor: ParallelProcessor,
    manager: ResearchSummaryManager,
    status_display: Optional[StatusDisplay] = None,
    progress: Optional[Progress] = None
) -> List[Paper]:
    """Process multiple PDFs in parallel."""
    logger.info("[bold blue]Starting parallel processing[/bold blue]")
    
    def process_single(pdf_file: str) -> Paper:
        try:
            logger.info(f"[dim]Processing {Path(pdf_file).name}[/dim]")
            paper = process_pdf(
                pdf_file,
                parallel_processor.llm_processor,
                manager,
                None  # Avoid nested status displays
            )
            if paper:
                logger.info(f"[green]✓[/green] Processed {Path(pdf_file).name}")
            return paper
        except Exception as e:
            logger.error(f"[red]✗[/red] Failed {Path(pdf_file).name}: {str(e)}")
            return None
    
    papers = []
    with ThreadPoolExecutor(max_workers=parallel_processor.max_workers) as executor:
        future_to_file = {
            executor.submit(process_single, pdf_file): pdf_file 
            for pdf_file in pdf_files
        }
        for future in as_completed(future_to_file):
            pdf_file = future_to_file[future]
            try:
                paper = future.result()
                if paper:
                    papers.append(paper)
                if progress:
                    progress.advance(0)  # Advance the main task
            except Exception as e:
                logger.error(f"[red]Error processing {Path(pdf_file).name}[/red]", exc_info=True)
                if progress:
                    progress.advance(0)  # Advance even on error
    
    logger.info(f"[bold green]✓[/bold green] Completed processing {len(papers)} papers")
    return papers

def generate_paper_summary(
    paper: Paper,
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> PaperSummary:
    """Generate summary for a paper."""
    if status_display:
        status_display.update(f"Generating summary for {paper.title}")
    
    summary = llm_processor.invoke_with_retry(
        "PAPER_SUMMARY",
        text=paper.original_text
    )
    
    return PaperSummary(
        paper=paper,
        summary=summary,
        weight=paper.weight
    )

def create_topic_synthesis(
    name: str,
    paper_summaries: List[PaperSummary],
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> TopicSynthesis:
    """Create a synthesis of papers within a research topic."""
    if status_display:
        status_display.update(f"Creating synthesis for topic: {name}")
    
    # Sort papers by weight
    sorted_summaries = sorted(
        paper_summaries,
        key=lambda x: x.weight,
        reverse=True
    )
    
    # Generate synthesis for this topic
    summaries_context = [
        {
            "title": ps.paper.title,
            "summary": ps.summary,
            "weight": ps.weight
        }
        for ps in sorted_summaries
    ]
    
    synthesis = llm_processor.invoke(
        "TOPIC_SYNTHESIS",
        context=summaries_context
    )
    
    return TopicSynthesis(
        name=name,
        paper_summaries=sorted_summaries,
        synthesis=synthesis
    )

def generate_overall_narrative(
    topic_syntheses: List[TopicSynthesis],
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> str:
    """Generate the overall narrative of research accomplishments."""
    if status_display:
        status_display.update("Generating overall narrative")
    
    # Get topic names
    theme_titles = [topic.name for topic in topic_syntheses]
    
    # Generate introductory narrative
    intro_narrative = llm_processor.invoke(
        "OVERALL_NARRATIVE",
        themes=theme_titles
    )
    
    # Build complete narrative with topic syntheses
    complete_narrative = [intro_narrative]
    
    # Add each topic synthesis
    for topic in topic_syntheses:
        # Add topic header
        complete_narrative.append(f"\n\n## {topic.name}")
        # Add topic synthesis
        complete_narrative.append(topic.synthesis)
    
    # Join all parts with proper spacing
    return "\n".join(complete_narrative)

def validate_author_name(name: str) -> Tuple[bool, str]:
    """Validate author name and return validation status with message."""
    parts = name.split()
    
    # Must have at least first and last name
    if len(parts) < 2:
        return False, f"Name must include first and last name: '{name}'"
    
    # Check for invalid patterns
    error_patterns = [
        (r'^\d', "Name starts with number"),
        (r'(?i)^(dr|prof|mr|ms|mrs|md|phd|university|institute)\b', "Starts with title"),
        (r'(?i)@|\.com|\.edu', "Contains email-like pattern"),
        (r'[<>{}[\]\\|;]', "Contains invalid characters")
    ]
    
    for pattern, message in error_patterns:
        if any(re.search(pattern, part) for part in parts):
            return False, f"{message}: '{name}'"
    
    # Check name part lengths
    if len(parts[0]) <= 1 or len(parts[-1]) <= 1:
        return False, f"First or last name too short: '{name}'"
    
    return True, ""

def generate_csv_summary(
    topic_summaries: List[TopicSynthesis],
    output_path: str,
    status_display: Optional[StatusDisplay] = None
) -> None:
    """Generate a CSV summary of all topics and papers.
    
    Args:
        topic_summaries: List of topic syntheses
        output_path: Path to save the CSV file
        status_display: Optional status display for progress updates
    """
    if status_display:
        status_display.update("Generating CSV summary")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV data
    rows = []
    
    # Add header row
    headers = ["Topic", "Paper Title", "Authors", "Summary", "Weight"]
    
    # Add data for each topic and paper
    for topic in topic_summaries:
        for paper_summary in topic.paper_summaries:
            # Format authors as semicolon-separated list using normalized names
            authors = "; ".join(a.normalized_name for a in paper_summary.paper.authors)
            
            rows.append([
                topic.name,
                paper_summary.paper.title,
                authors,
                paper_summary.summary,
                f"{paper_summary.weight:.2f}"
            ])
    
    # Write to CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        if status_display:
            status_display.success(f"CSV summary saved to {output_path}")
            
    except Exception as e:
        if status_display:
            status_display.error(f"Failed to save CSV summary: {str(e)}")
        raise

def run_pdf_summarization(
    config: Optional[dict] = None,
    status_display: Optional[StatusDisplay] = None
) -> Tuple[List[TopicSummary], str]:
    """Main function to run the PDF summarization pipeline."""
    try:
        # Load and validate configuration
        cfg = load_config(config)
        output_paths = get_output_paths(cfg)
        
        logger.info("[bold blue]Starting PDF summarization pipeline[/bold blue]")
        logger.info(f"Source directory: [cyan]{cfg['PDF_FOLDER']}[/cyan]")
        logger.info(f"Target author: [cyan]{cfg['AUTHOR_NAME']}[/cyan]")
        
        # Clear output directory
        output_dir = Path(cfg["OUTPUT_DIR"])
        if output_dir.exists():
            logger.info(f"Clearing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        manager = ResearchSummaryManager(cfg["OUTPUT_DIR"])
        llm_processor = LLMProcessor(llm=llm)
        parallel_processor = ParallelProcessor(
            max_workers=cfg["MAX_WORKERS"],
            llm_processor=llm_processor
        )
        
        # Get PDF files
        pdf_files = glob.glob(os.path.join(cfg["PDF_FOLDER"], "*.pdf"))
        if not pdf_files:
            logger.error(f"[red]No PDF files found in {cfg['PDF_FOLDER']}[/red]")
            raise ValueError(f"No PDF files found in {cfg['PDF_FOLDER']}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
            expand=True
        ) as progress:
            # Create tasks for each stage
            task_papers = progress.add_task("[cyan]Processing papers...", total=len(pdf_files))
            task_summaries = progress.add_task("[cyan]Generating summaries...", total=len(pdf_files), visible=False)
            task_grouping = progress.add_task("[cyan]Grouping papers...", total=1, visible=False)
            task_topics = progress.add_task("[cyan]Creating topic summaries...", total=5, visible=False)
            task_narrative = progress.add_task("[cyan]Generating narrative...", total=1, visible=False)
            
            # Process papers
            papers = process_papers_in_parallel(
                pdf_files=pdf_files,
                parallel_processor=parallel_processor,
                manager=manager,
                status_display=status_display,
                progress=progress
            )
            progress.update(task_papers, completed=len(pdf_files))
            
            # Convert to summaries
            progress.update(task_summaries, visible=True)
            paper_summaries = []
            for paper in papers:
                paper_summaries.append(PaperSummary(
                    paper=paper,
                    summary=paper.summary,
                    weight=paper.weight
                ))
                progress.advance(task_summaries)
            
            # Group papers
            progress.update(task_grouping, visible=True)
            topic_groups = group_papers_by_topic(
                papers=papers,
                llm_processor=llm_processor,
                num_topics=cfg["NUM_TOPICS"],
                status_display=status_display
            )
            progress.update(task_grouping, completed=1)
            
            # Create topic summaries
            progress.update(task_topics, visible=True, total=len(topic_groups))
            topic_summaries = []
            for topic_name, topic_papers in topic_groups.items():
                topic_summary = create_topic_synthesis(
                    name=topic_name,
                    paper_summaries=[ps for ps in paper_summaries if ps.paper in topic_papers],
                    llm_processor=llm_processor,
                    status_display=status_display
                )
                topic_summaries.append(topic_summary)
                
                # Save topic summary
                topic_path = manager.get_topic_path(topic_name)
                manager.save_json(topic_path, topic_summary.to_dict())
                progress.advance(task_topics)
            
            # Generate narrative
            progress.update(task_narrative, visible=True)
            narrative = generate_overall_narrative(
                topic_syntheses=topic_summaries,
                llm_processor=llm_processor,
                status_display=status_display
            )
            progress.update(task_narrative, completed=1)
        
        # Save narrative
        with open(manager.narrative_path, "w") as f:
            f.write(narrative)
        
        # Generate CSV summary
        generate_csv_summary(
            topic_summaries=topic_summaries,
            output_path=cfg["CSV_OUTPUT"],
            status_display=status_display
        )
        
        logger.info("[bold green]✓ Processing completed successfully![/bold green]")
        return topic_summaries, narrative
        
    except Exception as e:
        logger.error("[bold red]Fatal error in PDF summarization pipeline[/bold red]", exc_info=True)
        raise

class StatusDisplay:
    """Handles progress display and status updates using rich."""
    
    def __init__(self):
        self.console = console  # Use global console
    
    def update(self, message: str):
        """Display a status update message."""
        self.console.print(f"[cyan]{message}[/cyan]")
    
    def error(self, message: str):
        """Display an error message."""
        self.console.print(f"[red bold]Error:[/red bold] {message}")
    
    def success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green bold]Success:[/green bold] {message}")
    
    def warning(self, message: str):
        """Display a warning message."""
        self.console.print(f"[yellow]Warning:[/yellow] {message}")

def extract_text_from_pdf(pdf_file: str) -> str:
    """Extract text from a PDF file."""
    try:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        text = "\n".join(page.page_content for page in pages)
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def calculate_similarity(text1: str, text2: str, embeddings: Optional[OpenAIEmbeddings] = None) -> float:
    """Calculate semantic similarity between two texts using embeddings."""
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    
    # Get embeddings for both texts
    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    
    # Convert to numpy arrays for efficient computation
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    
    # Calculate cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)

def group_papers_by_topic(
    papers: List[Paper],
    llm_processor: LLMProcessor,
    num_topics: int = 5,
    status_display: Optional[StatusDisplay] = None
) -> Dict[str, List[Paper]]:
    """Group papers into topics using LLM-based clustering."""
    if status_display:
        status_display.update("Grouping papers into topics")
    
    # Calculate target distribution
    total_papers = len(papers)
    target_per_topic = total_papers / num_topics
    min_papers = max(3, int(target_per_topic * 0.7))
    max_papers = int(target_per_topic * 1.3)
    
    if status_display:
        status_display.update(f"Target: {min_papers}-{max_papers} papers per topic")
    
    # Initialize embeddings once for reuse
    embeddings = OpenAIEmbeddings()
    
    # Prepare paper summaries for clustering
    paper_list = []
    for idx, paper in enumerate(papers):
        paper_list.append({
            "index": idx,
            "weight": paper.weight,
            "summary": paper.summary,
            "title": paper.title
        })
    
    # Use structured output for clustering
    structured_llm = llm_processor.llm.with_structured_output(ClusteringOutput)
    clustering_result = structured_llm.invoke(
        llm_processor.prompts["CLUSTERING"].format(
            papers=paper_list,
            num_topics=num_topics,
            min_papers=min_papers,
            max_papers=max_papers,
            target_per_topic=target_per_topic
        )
    )
    
    # Build topic groups from structured output
    topic_groups = defaultdict(list)
    used_indices = set()
    
    # First pass: assign papers based on clustering
    for area in clustering_result["research_areas"]:
        area_name = area["name"]
        for idx in area["paper_indices"]:
            if idx in used_indices:
                continue
            if 0 <= idx < len(papers):
                topic_groups[area_name].append(papers[idx])
                used_indices.add(idx)
    
    # Second pass: assign any unassigned papers to most relevant topics
    all_indices = set(range(len(papers)))
    unassigned = all_indices - used_indices
    if unassigned and status_display:
        status_display.update(f"Finding best topics for {len(unassigned)} unassigned papers using semantic similarity")
    
    for idx in unassigned:
        paper = papers[idx]
        # Find most thematically relevant topic using semantic similarity
        best_topic = None
        max_similarity = -1
        
        for topic_name, topic_papers in topic_groups.items():
            # Calculate semantic similarity based on paper summaries
            topic_summary = " ".join(p.summary for p in topic_papers)
            similarity = calculate_similarity(paper.summary, topic_summary, embeddings)
            if similarity > max_similarity:
                max_similarity = similarity
                best_topic = topic_name
        
        if best_topic:
            topic_groups[best_topic].append(paper)
            used_indices.add(idx)
            if status_display:
                status_display.update(f"Assigned paper to '{best_topic}' (similarity: {max_similarity:.2f})")
    
    if status_display:
        for topic, group in topic_groups.items():
            status_display.update(f"Topic '{topic}': {len(group)} papers")
    
    return dict(topic_groups)

if __name__ == "__main__":
    try:
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red bold]Error:[/red bold] OPENAI_API_KEY environment variable not set")
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        status_display = StatusDisplay()
        topics, narrative = run_pdf_summarization(status_display=status_display)
        console.print("\n[green bold]✓ Processing completed successfully![/green bold]")
        
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1)