# 1. Imports and Configuration
from __future__ import annotations
import glob
import os
import json
import hashlib
from pathlib import Path
import re
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import Counter, defaultdict
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
import logging
import traceback
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Configure logging
def setup_logging():
    """Configure logging with timestamps and levels"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('pdf_summarizer.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize rich console
console = Console()

# Configuration Constants
DEFAULT_CONFIG = {
    # Author settings
    "AUTHOR_NAME": "Pranav Rajpurkar",
    
    # Model settings
    "MODEL_NAME": "gpt-4o-mini",
    "MODEL_TEMPERATURE": 0.1,
    
    # Directory settings
    "PDF_FOLDER": "pdfs",
    "OUTPUT_DIR": "outputs",  # Single outputs directory
    
    # Processing settings
    "NUM_TOPICS": 5,
    "MAX_WORKERS": 32,  # Increased for faster processing
    
    # Cache settings
    "CACHE_VERSION": "2.0",
}

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
    Provide a detailed technical summary of this paper's key contribution:
    {summary}
    
    Include:
    1. The specific research problem or challenge being addressed
    2. The key technical innovations and methodological approach
    3. The main empirical findings and quantitative results
    4. The broader implications or applications of the work
    
    Focus on concrete details - use numbers, metrics, and specific technical terms.
    Keep to 3-4 sentences that highlight the core technical contribution.
    """,
    
    "BRIEF_SUMMARY": """
    Provide a 1-2 sentence technical summary of this paper's key contribution:
    {text}
    
    Focus on the specific technical innovation and results.
    """,
    
    "TOPIC_NARRATIVE": """
    Create a flowing technical narrative that connects these papers within the same research area.
    
    Papers:
    {context}
    
    Guidelines:
    1. Start with the foundational work or key methodology
    2. Show how each paper builds upon or complements the others
    3. Use explicit transitions between papers
    4. Highlight technical relationships and methodological connections
    5. Keep the discussion focused and technical
    
    The narrative should be concise (3-4 paragraphs) but show clear connections between the papers.
    """,
    
    "CLUSTERING": """
    You will be provided with a list of research paper summaries along with their weights.
    Please cluster these papers into exactly {num_topics} topics, each representing a clear research claim or contribution.
    
    Distribution requirements:
    - Each topic should have between {min_papers} and {max_papers} papers
    - Target number of papers per topic is {target_per_topic:.1f}
    - Ensure at least one major paper (weight 1.0) per topic if possible
    
    Guidelines for topic names:
    1. Make a clear, declarative statement about the research contribution
    2. Focus on specific technical innovations and their impact
    3. Use natural language (no underscores)
    4. Emphasize the novel methodology and its demonstrated benefits
    
    Bad topic names:
    - Too vague (just stating a field)
    - Not making a specific claim
    - Not mentioning technical approach
    - Using technical notation or jargon
    
    Each topic should:
    - Make a specific claim about technical innovation
    - Focus on shared methodological advances
    
    Return ONLY a JSON object with topic names as keys and paper indices as values.
    """,
    
    "TECHNICAL_OVERVIEW": """
    Create a technical overview (2-3 paragraphs) that:
    1. Identifies the major research themes across these areas: {themes}
    2. Highlights specific technical innovations and methodological advances
    3. Shows how different technical approaches complement each other
    4. Emphasizes concrete outcomes and impact
    
    Style guidelines:
    - Use active voice and technical language
    - Focus on methodological connections
    - Highlight specific technical challenges solved
    - Maintain academic tone while being engaging
    - Include quantitative results where relevant
    """,
    
    "TOPIC_CONTEXT": """
    Write a single technical sentence that:
    1. Places this research claim in context: {theme}
    2. Connects it to broader technical challenges
    3. Highlights its methodological significance
    
    Be specific about technical aspects and avoid generic statements.
    """,
    
    "FUTURE_DIRECTIONS": """
    Write a technical conclusion (2-3 sentences) that:
    1. Identifies specific methodological challenges remaining in: {themes}
    2. Suggests concrete technical approaches for future work
    3. Highlights opportunities for combining methods across themes
    
    Focus on technical aspects and methodological innovations.
    """,
    
    "TOPIC_CLUSTERING": """
    You will be provided with a list of research paper summaries along with their weights.
    Please cluster these papers into exactly {num_topics} topics, each representing a clear research claim or contribution.
    
    Papers:
    {papers}
    
    Distribution requirements:
    - Each topic should have between {min_papers} and {max_papers} papers
    - Target number of papers per topic is {target_per_topic:.1f}
    - Ensure at least one major paper (weight 1.0) per topic if possible
    
    Guidelines for topic names:
    1. Make a clear, declarative statement about the research contribution
    2. Focus on specific technical innovations and their impact
    3. Use natural language (no underscores)
    4. Emphasize the novel methodology and its demonstrated benefits
    
    Bad topic names:
    - Too vague (just stating a field)
    - Not making a specific claim
    - Not mentioning technical approach
    - Using technical notation or jargon
    
    Each topic should:
    - Make a specific claim about technical innovation
    - Focus on shared methodological advances
    
    Return ONLY a JSON object with topic names as keys and paper indices as values.
    """,
    
    "TECHNICAL_SUMMARY": """
    Provide a detailed technical summary of this paper's key contribution:
    {text}
    
    Include:
    1. The specific research problem or challenge being addressed
    2. The key technical innovations and methodological approach
    3. The main empirical findings and quantitative results
    4. The broader implications or applications of the work
    
    Focus on concrete details - use numbers, metrics, and specific technical terms.
    Keep to 3-4 sentences that highlight the core technical contribution.
    """,
    
    "PAPER_ANALYSIS": """
    Analyze this research paper and extract the following information:
    
    Text: {text}
    
    Return a JSON object with these fields:
    1. "title": The paper's full title
    2. "authors": List of author names (first and last names)
    3. "summary": A technical summary of the key contributions (3-4 sentences)
    4. "weight": A score from 0.1 to 1.0 indicating the paper's technical depth and significance
    5. "role": One of ["primary_research", "survey", "case_study", "technical_report"]
    
    Focus on extracting accurate technical details and maintain the original formatting of names.
    """,
    
    "PAPER_INITIAL_ANALYSIS": """
    Analyze this research paper and extract the following information:
    
    Text: {text}
    
    Return a JSON object with these fields:
    1. "title": The paper's full title
    2. "authors": List of complete author names (first and last names)
    3. "technical_summary": A detailed technical summary of the key contributions (3-4 sentences)
    4. "brief_summary": A 1-2 sentence technical summary focusing on the core innovation
    5. "weight": A score from 0.1 to 1.0 indicating the paper's technical depth and significance
    
    Guidelines:
    - For authors: Include both first and last names, remove titles (Dr., Prof., etc.)
    - For summaries: Focus on concrete technical details, methods, and results
    - For weight: Consider novelty, technical depth, and empirical validation
    
    Return ONLY the JSON object, nothing else.
    """,
    
    "NORMALIZE_AUTHORS": """
    Normalize these author names by following these rules EXACTLY:
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
    
    Original names: {names}
    Return a JSON array of normalized names, maintaining original case.
    """,
}

# Base Classes
class FileManager:
    """Base class for file operations with common utilities."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def get_path(self, filename: str | Path) -> Path:
        """Get full path for a file."""
        if isinstance(filename, Path) and filename.is_absolute():
            return filename
        return self.base_dir / str(filename)
        
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
    """Manages the processing and storage of research summaries."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.papers_dir = self.output_dir / "papers"
        self.topics_dir = self.output_dir / "topics"
        self.narrative_path = self.output_dir / "year_in_review_narrative.txt"
        
        # Create directories if they don't exist
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        
        super().__init__(self.output_dir)
        
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
    """Handles all LLM interactions with consistent error handling and caching."""
    
    def __init__(self, llm: ChatOpenAI, cache_manager: Optional[CacheManager] = None):
        self.llm = llm
        self.cache = cache_manager
        self.prompts = {k: ChatPromptTemplate.from_template(v) for k, v in PROMPTS.items()}
    
    def invoke_with_retry(
        self,
        prompt_key: str,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Invoke LLM with retry logic and error handling."""
        if prompt_key not in self.prompts:
            logger.error(f"Prompt key '{prompt_key}' not found in available prompts: {list(self.prompts.keys())}")
            raise KeyError(f"Invalid prompt key: {prompt_key}")
            
        for attempt in range(max_retries):
            try:
                prompt = self.prompts[prompt_key]
                logger.debug(f"Attempting {prompt_key} (try {attempt + 1}/{max_retries})")
                result = self.llm.invoke(prompt.format(**kwargs))
                return result.content.strip()
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1} for {prompt_key}: {str(e)}")
                logger.debug(f"Prompt args: {kwargs}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed all retries for {prompt_key}")
                    logger.error(traceback.format_exc())
                    raise
                logger.warning(f"Retrying {prompt_key} ({attempt + 1}/{max_retries})")

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
    summary: str
    brief_summary: str
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
            brief_summary=cache_data.get("brief_summary", ""),
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
            "brief_summary": self.brief_summary,
            "weight": self.weight,
            "role": self.role,
            "original_text": self.original_text,
            "version": DEFAULT_CONFIG["CACHE_VERSION"]
        }

@dataclass
class PaperSummary:
    """Represents a paper's generated summaries."""
    paper: Paper
    brief_summary: str
    technical_summary: str
    weight: float
    
    def to_dict(self) -> dict:
        """Convert to serializable format."""
        return {
            "file_path": self.paper.file_path,
            "title": self.paper.title,
            "brief_summary": self.brief_summary,
            "technical_summary": self.technical_summary,
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
    """Represents a group of related papers with generated summaries."""
    name: str
    paper_summaries: List[PaperSummary]
    flowing_narrative: str
    context_summary: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to serializable format."""
        return {
            "name": self.name,
            "flowing_narrative": self.flowing_narrative,
            "context_summary": self.context_summary,
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

        # Use structured output with JSON schema
        structured_llm = llm_processor.llm.with_structured_output(PAPER_ANALYSIS_SCHEMA)
        analysis_data = structured_llm.invoke(
            f"Analyze this academic paper and extract the key information: {text}"
        )
        
        # analysis_data is now a dict, so use dict access
        if analysis_data["authors"]:
            normalized_names = llm_processor.invoke_with_retry(
                "NORMALIZE_AUTHORS",
                names=json.dumps(analysis_data["authors"])
            )
            try:
                normalized_names = json.loads(normalized_names)
            except json.JSONDecodeError:
                normalized_names = analysis_data["authors"]  # Fallback to original names
        else:
            normalized_names = []
        
        # Create author objects
        authors = [
            Author(full_name=orig, normalized_name=norm)
            for orig, norm in zip(analysis_data["authors"], normalized_names)
        ]
        
        # Create paper object
        paper = Paper.create(
            pdf_file=pdf_file,
            title=analysis_data["title"],
            authors=authors,
            summary=analysis_data["technical_summary"],
            brief_summary=analysis_data["brief_summary"],
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
            "brief_summary": paper.brief_summary,
            "technical_summary": paper.summary,
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
    status_display: Optional[StatusDisplay] = None
) -> List[Paper]:
    """Process multiple PDFs in parallel."""
    if status_display:
        status_display.update(f"Processing {len(pdf_files)} papers in parallel")
    
    def process_single(pdf_file: str) -> Paper:
        try:
            return process_pdf(
                pdf_file,
                parallel_processor.llm_processor,
                manager,
                None  # Avoid nested status displays
            )
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            return None
    
    papers = parallel_processor.process_items(
        items=pdf_files,
        process_fn=process_single,
        status_msg="Processing PDF files",
        error_msg="Failed to process PDF file:",
        status=status_display
    )
    return [p for p in papers if p is not None]

def generate_paper_summary(
    paper: Paper,
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> PaperSummary:
    """Generate detailed summaries for a paper."""
    if status_display:
        status_display.update(f"Generating summaries for {paper.title}")
    
    # Generate brief and technical summaries without truncation
    brief_summary = llm_processor.invoke_with_retry(
        "BRIEF_SUMMARY",
        text=paper.original_text
    )
    
    technical_summary = llm_processor.invoke_with_retry(
        "TECHNICAL_SUMMARY",
        text=paper.original_text
    )
    
    return PaperSummary(
        paper=paper,
        brief_summary=brief_summary,
        technical_summary=technical_summary,
        weight=paper.weight
    )

def create_topic_summary(
    name: str,
    paper_summaries: List[PaperSummary],
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> TopicSummary:
    """Create a topic summary from a list of paper summaries."""
    if status_display:
        status_display.update(f"Creating topic summary for {name}")
    
    # Sort papers by weight
    sorted_summaries = sorted(
        paper_summaries,
        key=lambda x: x.weight,
        reverse=True
    )
    
    # Generate flowing narrative
    summaries_context = [
        {
            "title": ps.paper.title,
            "brief_summary": ps.brief_summary,
            "technical_summary": ps.technical_summary,
            "weight": ps.weight
        }
        for ps in sorted_summaries
    ]
    
    flowing_narrative = llm_processor.invoke_with_retry(
        "TOPIC_NARRATIVE",
        context=summaries_context
    )
    
    # Generate context summary if needed
    context_summary = None
    if len(paper_summaries) > 1:
        context_summary = llm_processor.invoke_with_retry(
            "TOPIC_CONTEXT",
            theme=name
        )
    
    return TopicSummary(
        name=name,
        paper_summaries=sorted_summaries,
        flowing_narrative=flowing_narrative,
        context_summary=context_summary
    )

def extract_text_from_pdf(pdf_file: str) -> str:
    """Extract text from a PDF file."""
    try:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        text = "\n".join(page.page_content for page in pages)
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

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
    min_papers = max(1, int(target_per_topic * 0.7))
    max_papers = int(target_per_topic * 1.3)
    
    if status_display:
        status_display.update(f"Target: {min_papers}-{max_papers} papers per topic")
    
    # Prepare paper summaries for clustering
    paper_list = []
    for idx, paper in enumerate(papers):
        paper_list.append({
            "index": idx,
            "weight": paper.weight,
            "summary": paper.summary,
            "title": paper.title
        })
    
    # Get clustering from LLM and parse JSON response
    clustering_result = llm_processor.invoke_with_retry(
        "TOPIC_CLUSTERING",
        papers=paper_list,
        num_topics=num_topics,
        min_papers=min_papers,
        max_papers=max_papers,
        target_per_topic=target_per_topic
    )
    
    # Parse the JSON string into a dictionary, handling potential backticks
    try:
        # Remove any backticks and 'json' tag that might be present
        cleaned_json = clustering_result.replace('```json', '').replace('```', '').strip()
        clustering_dict = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        if status_display:
            status_display.error(f"Failed to parse clustering result: {e}")
            status_display.error(f"Raw result: {clustering_result}")
        # Fallback: put all papers in one topic
        clustering_dict = {"Default Topic": list(range(len(papers)))}
    
    # Build topic groups
    topic_groups = defaultdict(list)
    for topic_name, indices in clustering_dict.items():
        for idx in indices:
            if 0 <= int(idx) < len(papers):  # Add bounds check
                topic_groups[topic_name].append(papers[int(idx)])
    
    if status_display:
        for topic, group in topic_groups.items():
            status_display.update(f"Topic '{topic}': {len(group)} papers")
    
    return dict(topic_groups)

def generate_narrative(
    topic_summaries: List[TopicSummary],
    llm_processor: LLMProcessor,
    status_display: Optional[StatusDisplay] = None
) -> str:
    """Generate a cohesive narrative connecting research themes."""
    if status_display:
        status_display.update("Generating final research narrative")
    
    # Get topic names
    theme_titles = [topic.name for topic in topic_summaries]
    
    # Generate technical overview
    if status_display:
        status_display.update("Creating technical overview")
    overview = llm_processor.invoke_with_retry(
        "TECHNICAL_OVERVIEW",
        themes=theme_titles
    )
    narrative = f"{overview}\n\n"
    
    # Add detailed topic discussions
    if status_display:
        status_display.update("Adding detailed topic discussions")
    
    for topic in topic_summaries:
        narrative += f"### {topic.name}\n\n"
        if topic.context_summary:
            narrative += f"{topic.context_summary}\n\n"
        narrative += f"{topic.flowing_narrative}\n\n"
    
    # Add future directions
    if status_display:
        status_display.update("Adding future directions")
    conclusion = llm_processor.invoke_with_retry(
        "FUTURE_DIRECTIONS",
        themes=theme_titles
    )
    narrative += f"\n{conclusion}"
    
    return narrative

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

def run_pdf_summarization(
    config: Optional[dict] = None,
    status_display: Optional[StatusDisplay] = None
) -> Tuple[List[TopicSummary], str]:
    """Main function to run the PDF summarization pipeline."""
    # Load configuration
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    if status_display:
        status_display.update(f"Starting PDF summarization from: {cfg['PDF_FOLDER']}")
        status_display.update(f"Analyzing contributions for: {cfg['AUTHOR_NAME']}")
    
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
        raise ValueError(f"No PDF files found in {cfg['PDF_FOLDER']}")
    
    # Process papers in parallel
    papers = process_papers_in_parallel(
        pdf_files=pdf_files,
        parallel_processor=parallel_processor,
        manager=manager,
        status_display=status_display
    )
    
    # Convert papers to paper summaries
    paper_summaries = []
    for paper in papers:
        paper_summaries.append(PaperSummary(
            paper=paper,
            brief_summary=paper.brief_summary,
            technical_summary=paper.summary,
            weight=paper.weight
        ))
    
    # Group papers by topic
    topic_groups = group_papers_by_topic(
        papers=papers,
        llm_processor=llm_processor,
        num_topics=cfg["NUM_TOPICS"],
        status_display=status_display
    )
    
    # Create and save topic summaries
    topic_summaries = []
    for topic_name, topic_papers in topic_groups.items():
        topic_summary = create_topic_summary(
            name=topic_name,
            paper_summaries=[
                ps for ps in paper_summaries
                if ps.paper in topic_papers
            ],
            llm_processor=llm_processor,
            status_display=status_display
        )
        topic_summaries.append(topic_summary)
        
        # Save individual topic summary
        topic_path = manager.get_topic_path(topic_name)
        manager.save_json(topic_path, topic_summary.to_dict())
    
    # Generate narrative
    narrative = generate_narrative(
        topic_summaries=topic_summaries,
        llm_processor=llm_processor,
        status_display=status_display
    )
    
    # Save narrative
    with open(manager.narrative_path, "w") as f:
        f.write(narrative)
    
    if status_display:
        status_display.update("Processing complete!")
    
    return topic_summaries, narrative

class StatusDisplay:
    """Handles progress display and status updates."""
    
    def __init__(self):
        self.console = Console()
    
    def update(self, message: str):
        """Display a status update message."""
        self.console.print(f"[dim]{message}[/dim]")
    
    def error(self, message: str):
        """Display an error message."""
        self.console.print(f"[red]Error: {message}[/red]")
    
    def success(self, message: str):
        """Display a success message."""
        self.console.print(f"[green]{message}[/green]")

# After the imports, add logging configuration
def setup_logging():
    """Configure logging with timestamps and levels"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('pdf_summarizer.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

if __name__ == "__main__":
    try:
        logger.info("Starting PDF summarization process")
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        
        status_display = StatusDisplay()
        
        # Run with default config
        topics, narrative = run_pdf_summarization(status_display=status_display)
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise