import glob
import os
import json
import hashlib
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from dataclasses import dataclass
from collections import Counter
import unicodedata
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from collections import defaultdict
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

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
    "OUTPUT_DIR": "outputs",
    
    # Processing settings
    "NUM_TOPICS": 5,
    "MAX_WORKERS": 16,
    
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
    {summary}
    
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
    
    "PAPER_EXTRACTION": """
    Summarize this research paper in 2-3 sentences, focusing on:
    1. The main contribution or innovation
    2. Key results or findings
    
    Text: {text}
    
    Provide ONLY the summary, no additional text.
    """
}

# Add a new base class for file operations
class FileManager:
    """Base class for file operations with common utilities."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def get_path(self, filename: str | Path) -> Path:
        """Get full path for a file."""
        # If filename is already a Path object and is absolute, return it directly
        if isinstance(filename, Path) and filename.is_absolute():
            return filename
        # Otherwise, join it with base_dir
        return self.base_dir / str(filename)
        
    def save_json(self, filename: str | Path, data: dict):
        """Save data as JSON."""
        path = self.get_path(filename)
        # Ensure parent directory exists
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
                print(f"[ERROR] Failed to load {filename}: {e}")
                return default or {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
        return default or {"version": DEFAULT_CONFIG["CACHE_VERSION"]}

# Update ResearchSummaryManager to inherit from FileManager
class ResearchSummaryManager(FileManager):
    """Manages the processing and storage of research summaries."""
    
    def __init__(self, output_dir: str = DEFAULT_CONFIG["OUTPUT_DIR"]):
        super().__init__(Path(output_dir))
        
        # Create subdirectories
        self.subdirs = {
            "cache": self.base_dir / "cache",
            "summaries": self.base_dir / "summaries",
            "papers": self.base_dir / "papers",
            "topics": self.base_dir / "topics",
            "data": self.base_dir / "data"
        }
        
        for directory in self.subdirs.values():
            directory.mkdir(exist_ok=True)
            
        # Define common paths
        self.paths = {
            "partial_results": self.subdirs["data"] / "partial_results.json",
            "final_summaries": self.subdirs["data"] / "final_summaries.json",
            "narrative": self.base_dir / "year_in_review_narrative.txt"
        }
    
    def get_paper_cache_path(self, paper_hash: str) -> Path:
        """Get path for individual paper cache file."""
        return self.subdirs["cache"] / f"{paper_hash}.json"
    
    def get_paper_path(self, paper_hash: str) -> Path:
        """Get path for processed paper data."""
        return self.subdirs["papers"] / f"{paper_hash}.json"
    
    def get_topic_path(self, topic_name: str) -> Path:
        """Get path for topic data with proper filename sanitization."""
        # Replace any non-alphanumeric chars (except spaces) with underscores
        safe_name = re.sub(r'[^\w\s-]', '_', topic_name)
        # Replace spaces with underscores and collapse multiple underscores
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        # Remove leading/trailing underscores and convert to lowercase
        safe_name = safe_name.strip('_').lower()
        # Ensure the filename isn't empty and add .json extension
        safe_name = safe_name or 'unnamed_topic'
        safe_name = f"{safe_name}.json"
        
        return self.subdirs["topics"] / safe_name

# Add a new LLMInterface class to handle all LLM operations
class LLMInterface:
    """Handles all interactions with the language model."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompts = {k: ChatPromptTemplate.from_template(v) for k, v in PROMPTS.items()}
    
    def invoke_prompt(self, prompt_key: str, **kwargs) -> str:
        """Invoke a predefined prompt with parameters."""
        try:
            prompt = self.prompts[prompt_key]
            result = self.llm.invoke(prompt.format(**kwargs))
            return result.content.strip()
        except Exception as e:
            print(f"[ERROR] Failed to invoke {prompt_key}: {e}")
            return ""
    
    def normalize_name(self, name: str) -> str:
        """Normalize an author name."""
        return self.invoke_prompt("NAME_NORMALIZATION", name=name)
    
    def extract_authors(self, text: str) -> List[str]:
        """Extract author names from text."""
        result = self.invoke_prompt("AUTHOR_EXTRACTION", text=text[:2000])
        return [name.strip() for name in result.split('\n') if name.strip()]
    
    def extract_title(self, text: str) -> str:
        """Extract paper title from text."""
        return self.invoke_prompt("TITLE_EXTRACTION", text=text[:2000]) or "Untitled"

@dataclass
class Author:
    """Structured representation of an author."""
    full_name: str
    normalized_name: str
    role: str = 'contributing_author'
    
    def matches(self, other_name: str, llm_interface: LLMInterface) -> bool:
        """Compare this author with another name"""
        other_normalized = normalize_author_name(other_name, llm_interface)
        return (
            self.normalized_name.lower() == other_normalized.lower() or
            self.full_name.lower() == other_name.lower()
        )

# Author handling prompts and functions
def normalize_author_name(name: str, llm_interface: LLMInterface) -> str:
    """Normalize an author name using LLM."""
    if not llm_interface:
        raise ValueError("LLM interface is required for name normalization")
    return llm_interface.normalize_name(name)

def extract_authors(text_or_dict, llm_interface: LLMInterface) -> List[str]:
    """Extract author names from text or dictionary"""
    if isinstance(text_or_dict, str):
        try:
            return llm_interface.extract_authors(text_or_dict)
        except Exception:
            return []
            
    elif isinstance(text_or_dict, dict):
        if 'authors' in text_or_dict:
            if isinstance(text_or_dict['authors'], list):
                return text_or_dict['authors']
            return extract_authors(text_or_dict['authors'], llm_interface)
            
        for key in ['author', 'full_name', 'name']:
            if key in text_or_dict:
                return extract_authors(text_or_dict[key], llm_interface)
                
        if 'original_text' in text_or_dict:
            return extract_authors(text_or_dict['original_text'], llm_interface)
    
    return []

@dataclass
class Paper:
    """Represents a processed research paper."""
    file_path: str
    summary: str
    weight: float
    processed_time: str
    role: str
    original_text: str = ""
    title: str = "Untitled"
    authors: List[Author] = None  # Changed to List[Author]
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
            
    def get_author_names(self) -> List[str]:
        """Get list of normalized author names"""
        return [author.normalized_name for author in self.authors]
    
    def has_author(self, name: str) -> bool:
        """Check if paper has a specific author"""
        return any(author.matches(name) for author in self.authors)
    
    def get_author_role(self, name: str) -> str:
        """Get role of specific author"""
        for author in self.authors:
            if author.matches(name):
                return author.role
        return "not_author"

@dataclass
class Topic:
    """Represents a group of related papers."""
    name: str
    papers: List[Paper]
    summary: Optional[str] = None
    paper_summaries: Dict[str, str] = None

    def __post_init__(self):
        if self.paper_summaries is None:
            self.paper_summaries = {}

def get_cache_key(pdf_path):
    """Generate a cache key based on file path and last modified time."""
    pdf_stats = os.stat(pdf_path)
    content_key = f"{pdf_path}:{pdf_stats.st_mtime}"
    return hashlib.md5(content_key.encode()).hexdigest()

def load_cache_for_paper(paper_path: str, manager: ResearchSummaryManager) -> dict:
    """Load cache for a specific paper."""
    paper_hash = get_cache_key(paper_path)
    cache_file = manager.get_paper_cache_path(paper_hash)
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            if cache_data.get("version") != DEFAULT_CONFIG["CACHE_VERSION"]:
                return {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
            return cache_data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[CACHE] Error loading cache for {paper_path}: {e}")
            return {"version": DEFAULT_CONFIG["CACHE_VERSION"]}
    return {"version": DEFAULT_CONFIG["CACHE_VERSION"]}

def save_cache_for_paper(paper_path: str, cache_data: dict, manager: ResearchSummaryManager):
    """Save cache for a specific paper."""
    paper_hash = get_cache_key(paper_path)
    cache_file = manager.get_paper_cache_path(paper_hash)
    
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=4)

def save_paper_data(paper: Paper, manager: ResearchSummaryManager):
    """Save processed paper data to individual file."""
    paper_hash = get_cache_key(paper.file_path)
    paper_file = manager.get_paper_path(paper_hash)
    
    paper_data = {
        "file_path": paper.file_path,
        "title": paper.title,
        "authors": paper.authors,
        "summary": paper.summary,
        "weight": paper.weight,
        "processed_time": paper.processed_time,
        "role": paper.role
    }
    
    with open(paper_file, "w") as f:
        json.dump(paper_data, f, indent=4)

def load_partial_results():
    """Load partial results from disk with version checking."""
    manager = ResearchSummaryManager()
    partial_results_path = manager.paths["partial_results"]  # Use the path from paths dictionary
    
    if partial_results_path.exists():
        try:
            with open(partial_results_path, "r") as f:
                data = json.load(f)
            
            if data.get("version") != DEFAULT_CONFIG["CACHE_VERSION"]:
                print(f"[PARTIAL] Outdated version. Expected {DEFAULT_CONFIG['CACHE_VERSION']}, found {data.get('version')}. Rebuilding.")
                return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}
            
            return data["entries"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[PARTIAL] Error loading partial results: {e}. Rebuilding.")
            return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}
    return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}

def save_partial_results(results):
    """Save partial results to disk with version information."""
    manager = ResearchSummaryManager()
    partial_results_path = manager.paths["partial_results"]  # Use the path from paths dictionary
    
    data = {
        "version": DEFAULT_CONFIG["CACHE_VERSION"],
        "entries": results
    }
    with open(partial_results_path, "w") as f:
        json.dump(data, f, indent=4)

def extract_title_with_cache(text: str, llm_interface: LLMInterface, cache_data: dict) -> str:
    """Extract paper title from text using cache data."""
    if cache_data and "title" in cache_data:
        print(f"[CACHE] Using cached title")
        return cache_data["title"]
    
    return llm_interface.extract_title(text)

def generate_weighted_topic_summary(papers: List[Paper], llm_interface: LLMInterface, author_name: str, status=None) -> Tuple[str, Dict[str, str]]:
    """Generate a flowing topic summary and paper summaries."""
    console.print("\n[cyan]Generating topic summaries...[/cyan]")
    
    # Sort papers by weight in descending order
    sorted_papers = sorted(papers, key=lambda x: x.weight, reverse=True)
    
    # Generate brief summaries for each paper
    paper_summaries = {}
    brief_summary_prompt = ChatPromptTemplate.from_template(PROMPTS["BRIEF_SUMMARY"])
    
    # Use the passed status instead of creating a new one
    for i, paper in enumerate(sorted_papers, 1):
        if status:
            status.update(f"[dim]Generating brief summary {i}/{len(sorted_papers)}[/dim]")
        brief_summary = llm_interface.llm.invoke(brief_summary_prompt.format(summary=paper.summary))
        paper_summaries[paper.file_path] = brief_summary.content
    
    if status:
        status.update("[bold blue]→ Generating flowing narrative...[/bold blue]")
    else:
        console.print("[bold blue]→ Generating flowing narrative...[/bold blue]")
    
    # Generate papers text for narrative
    papers_text = "\n\n".join([
        f"Title: {p.title}\nWeight: {p.weight:.1f}\nSummary: {paper_summaries[p.file_path]}"
        for p in sorted_papers
    ])
    
    narrative_prompt = ChatPromptTemplate.from_template(PROMPTS["TOPIC_NARRATIVE"])
    
    chain = create_stuff_documents_chain(
        llm=llm_interface.llm,
        prompt=narrative_prompt,
        document_prompt=PromptTemplate.from_template("{page_content}"),
        document_variable_name="context"
    )
    
    doc = Document(page_content=papers_text)
    flowing_summary = chain.invoke({"context": [doc]})
    
    console.print("[green]✓ Topic summary generated[/green]")
    
    return flowing_summary, paper_summaries

def process_pdf(pdf_file: str, manager: ResearchSummaryManager, llm_interface: LLMInterface, author_name: str) -> Paper:
    """Process a single PDF with improved organization."""
    cache_data = load_cache_for_paper(pdf_file, manager)
    
    if "summary" in cache_data:
        # Use cached data
        filename = Path(pdf_file).name
        if len(filename) > 40:
            filename = filename[:37] + "..."
        console.print(f"[dim]← Cached: {filename}[/dim]")
        
        # Convert cached author names to Author objects
        author_names = extract_authors(cache_data, llm_interface)
        authors = []
        for name in author_names:
            normalized = llm_interface.normalize_name(name)
            # Create Author object without role first
            authors.append(Author(full_name=name, normalized_name=normalized))
        
        # Determine role and weight after we have all authors
        role, weight = determine_author_role_from_authors(authors, author_name, llm_interface)
        
        return Paper(
            file_path=pdf_file,
            summary=cache_data['summary'],
            weight=weight,  # Use the determined weight
            processed_time=str(Path(pdf_file).stat().st_mtime),
            role=role,  # Use the determined role
            original_text=cache_data.get('original_text', ''),
            title=cache_data['title'],
            authors=authors
        )
    
    # Process new file
    filename = Path(pdf_file).name
    if len(filename) > 40:
        filename = filename[:37] + "..."
    console.print(f"[bold blue]→ Processing: {filename}[/bold blue]")
    
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    
    # Combine all pages into one text
    original_text = "\n".join([doc.page_content for doc in docs]) if docs else ""
    
    # Extract and process authors
    author_names = extract_authors(original_text, llm_interface)
    authors = []
    for name in author_names:
        normalized = llm_interface.normalize_name(name)
        # Create Author object without role first
        authors.append(Author(full_name=name, normalized_name=normalized))
    
    # Determine role and weight after we have all authors
    role, weight = determine_author_role_from_authors(authors, author_name, llm_interface)
    
    # Extract metadata using cache
    title = extract_title_with_cache(original_text, llm_interface, {})
    
    # Generate summary
    summary_prompt = PromptTemplate.from_template(PROMPTS["PAPER_EXTRACTION"])
    chain = load_summarize_chain(llm_interface.llm, chain_type="stuff", prompt=summary_prompt)
    summary = chain.invoke(docs)["output_text"]
    
    # Save cache and paper data
    cache_data.update({
        'summary': summary,
        'weight': weight,
        'role': role,
        'title': title,
        'authors': [{'full_name': a.full_name, 'normalized_name': a.normalized_name} for a in authors],
        'original_text': original_text,
        'version': DEFAULT_CONFIG["CACHE_VERSION"]
    })
    save_cache_for_paper(pdf_file, cache_data, manager)
    
    return Paper(
        file_path=pdf_file,
        summary=summary,
        weight=weight,
        processed_time=str(Path(pdf_file).stat().st_mtime),
        role=role,
        original_text=original_text,
        title=title,
        authors=authors
    )

def summarize_pdfs_from_folder(pdfs_folder: str, author_name: str, num_topics: int = 5, status=None) -> List[Topic]:
    """Process PDFs and return a list of Topics."""
    manager = ResearchSummaryManager()
    llm_interface = LLMInterface(llm)
    papers: List[Paper] = []
    partial_results = load_partial_results()

    pdf_files = glob.glob(pdfs_folder + "/*.pdf")
    total_pdfs = len(pdf_files)
    processed_count = 0
    save_threshold = max(1, min(5, total_pdfs // 10))

    console.print()
    console.rule("[bold cyan]Processing PDFs", style="cyan")
    console.print(f"[cyan]Found {total_pdfs} PDFs to process[/cyan]")
    console.print()

    with ThreadPoolExecutor(max_workers=DEFAULT_CONFIG["MAX_WORKERS"]) as executor:
        futures = {
            executor.submit(process_pdf, pdf_file, manager, llm_interface, author_name): pdf_file 
            for pdf_file in pdf_files
        }
        
        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                paper = future.result()
                papers.append(paper)

                processed_count += 1
                if processed_count % save_threshold == 0:
                    save_partial_results(partial_results)
                    if status:
                        status.update(f"[bold green]Progress: {processed_count}/{total_pdfs} PDFs")

            except Exception as e:
                console.print(f"[red]Error processing {Path(pdf_file).name}:[/red]")
                console.print(f"[red dim]{str(e)}[/red dim]")

    console.print()
    # Save final partial results
    if partial_results:
        save_partial_results(partial_results)

    if not papers:
        raise ValueError("No summaries were generated. Check the error messages above.")

    if status:
        status.update("[bold green]Grouping papers into topics...")

    # Group papers by topic using llm_interface
    topic_groups = group_papers_by_topic(papers, llm_interface, num_topics=num_topics)
    
    # Generate summaries for each topic
    topics = []
    for topic_name, topic_papers in topic_groups.items():
        topic_summary, paper_summaries = generate_weighted_topic_summary(
            topic_papers, llm_interface, author_name, status=status
        )
        
        topic = Topic(
            name=topic_name,
            papers=topic_papers,
            summary=topic_summary,
            paper_summaries=paper_summaries
        )
        topics.append(topic)
        
        # Save topic data
        topic_path = manager.get_topic_path(topic_name)
        manager.save_json(topic_path, {
            "name": topic_name,
            "summary": topic_summary,
            "paper_summaries": paper_summaries,
            "papers": [paper.file_path for paper in topic_papers]
        })
    
    return topics

def generate_narrative(topics: List[Topic], llm) -> str:
    """Generate a cohesive narrative connecting research themes and contributions."""
    console.print("\n[cyan]Generating final research narrative...[/cyan]")
    
    theme_titles = [topic.name for topic in topics]
    
    # 1. Technical Overview
    console.print("[bold blue]→ Creating technical overview...[/bold blue]")
    overview_prompt = ChatPromptTemplate.from_template(PROMPTS["TECHNICAL_OVERVIEW"])
    overview = llm.invoke(overview_prompt.format(themes=", ".join(theme_titles)))
    narrative = f"{overview.content}\n\n"

    # 2. Detailed Topic Discussions
    console.print("[bold blue]→ Adding detailed topic discussions...[/bold blue]")
    for i, topic in enumerate(topics, 1):
        console.print(f"[dim]  Processing topic {i}/{len(topics)}: {topic.name}[/dim]")
        narrative += f"### {topic.name}\n\n"
        
        # Add context sentence
        context_prompt = ChatPromptTemplate.from_template(PROMPTS["TOPIC_CONTEXT"])
        context = llm.invoke(context_prompt.format(theme=topic.name))
        narrative += f"{context.content}\n\n"
        
        # Use the existing detailed topic summary
        narrative += f"{topic.summary}\n\n"

    # 3. Future Directions
    console.print("[bold blue]→ Adding future directions...[/bold blue]")
    conclusion_prompt = ChatPromptTemplate.from_template(PROMPTS["FUTURE_DIRECTIONS"])
    conclusion = llm.invoke(conclusion_prompt.format(themes=", ".join(theme_titles)))
    narrative += f"\n{conclusion.content}"

    console.print("[green]✓ Research narrative generated[/green]")
    return narrative

def validate_author_name(name: str) -> Tuple[bool, str]:
    """
    Validate an author name and return if it's valid and any warning message.
    
    Returns:
        Tuple of (is_valid: bool, warning_message: str)
    """
    parts = name.split()
    
    # Must have at least two parts (first and last name)
    if len(parts) < 2:
        return False, f"Single word name: '{name}'"
        
    # Check for common error patterns
    error_patterns = [
        r'^\d',  # Numbers at start of name
        r'(?i)^(dr|prof|mr|ms|mrs|md|phd|university|institute|hospital|center|dept|department)\b',  # Titles or institutions
        r'(?i)@|\.com|\.edu',  # Email-like patterns
    ]
    
    for pattern in error_patterns:
        if any(re.search(pattern, part) for part in parts):
            return False, f"Contains invalid patterns: '{name}'"
    
    # Allow names with middle initials (e.g., "John A. Smith" or "John A Smith")
    # First and last name should be longer than 1 character
    if len(parts[0]) <= 1 or len(parts[-1]) <= 1:
        return False, f"First or last name too short: '{name}'"
    
    return True, ""

def run_pdf_summarization(config: dict = None):
    """Main function with improved organization."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Initialize core components
    manager = ResearchSummaryManager(cfg["OUTPUT_DIR"])
    
    console.print(f"[cyan]Starting PDF summarization from: {cfg['PDF_FOLDER']}[/cyan]")
    console.print(f"[cyan]Analyzing contributions for: {cfg['AUTHOR_NAME']}[/cyan]")
    
    with console.status("[bold green]Processing...") as status:
        topics = summarize_pdfs_from_folder(
            pdfs_folder=cfg["PDF_FOLDER"],
            author_name=cfg["AUTHOR_NAME"],
            num_topics=cfg["NUM_TOPICS"],
            status=status
        )
        
        status.update("[bold green]Generating final narrative...")
        llm_interface = LLMInterface(llm)
        narrative = generate_narrative(topics, llm_interface.llm)
        
        # Save narrative using manager
        with open(manager.paths["narrative"], "w") as f:
            f.write(narrative)
    
    console.print("[bold green]✓[/bold green] Processing complete!")
    return topics, narrative

def group_papers_by_topic(papers: List[Paper], llm_interface: LLMInterface, num_topics: int = 5) -> Dict[str, List[Paper]]:
    """Group papers into topics using LLM-based classification."""
    console.print("\n[cyan]Grouping papers into topics...[/cyan]")
    
    # Calculate target papers per topic
    total_papers = len(papers)
    target_per_topic = total_papers / num_topics
    min_papers = max(1, int(target_per_topic * 0.7))
    max_papers = int(target_per_topic * 1.3)
    
    console.print(f"[dim]Target distribution: {min_papers}-{max_papers} papers per topic[/dim]")
    
    # Build the list of papers with their summaries and weights
    console.print("[dim]Preparing paper summaries for clustering...[/dim]")
    paper_list = ""
    for idx, paper in enumerate(papers):
        paper_list += f"\nPaper {idx}:\nWeight: {paper.weight:.1f}\nSummary: {paper.summary}\n"

    console.print("[bold blue]→ Requesting topic clustering from LLM...[/bold blue]")
    
    # Format the clustering prompt with the distribution requirements
    clustering_prompt = ChatPromptTemplate.from_template(PROMPTS["CLUSTERING"]).format(
        num_topics=num_topics,
        min_papers=min_papers,
        max_papers=max_papers,
        target_per_topic=target_per_topic
    )
    
    full_prompt = clustering_prompt + paper_list + "\nResponse (in JSON format):"

    # Get the clustering result from the LLM
    clustering_result = llm_interface.llm.invoke(full_prompt).content.strip()
    
    try:
        json_match = re.search(r'\{.*\}', clustering_result, re.DOTALL)
        if json_match:
            clustering_result = json_match.group(0)
        
        topic_groups_indices = json.loads(clustering_result)
        
        # Validate distribution
        for topic, indices in topic_groups_indices.items():
            if len(indices) < min_papers or len(indices) > max_papers:
                print(f"[yellow]Warning: Topic '{topic}' has {len(indices)} papers, outside target range of {min_papers}-{max_papers}[/yellow]")
        
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[ERROR] Failed to parse clustering result: {e}")
        print(f"[DEBUG] Raw clustering result:\n{clustering_result}")
        topic_groups_indices = {"uncategorized_papers": list(range(len(papers)))}

    # Build the final topic groups
    topic_groups = {}
    for topic, indices in topic_groups_indices.items():
        topic_groups[topic] = [papers[int(idx)] for idx in indices]

    console.print(f"[green]✓ Created {len(topic_groups)} topic groups[/green]")
    for topic, papers_list in topic_groups.items():
        console.print(f"[dim]  • {topic}: {len(papers_list)} papers[/dim]")
    
    return topic_groups

def determine_author_role_from_authors(authors: List[Author], target_name: str, llm_interface: LLMInterface) -> Tuple[str, float]:
    """Determine role and weight based on author position."""
    if not authors:
        return "unknown", 0.1
    
    # Get normalized names for comparison
    normalized_names = [author.normalized_name for author in authors]
    normalized_target = llm_interface.normalize_name(target_name)
    
    # First or last author gets highest weight
    if normalized_names[0] == normalized_target:
        return "first_author", 1.0
    elif normalized_names[-1] == normalized_target:
        return "last_author", 1.0
    elif normalized_target in normalized_names:
        return "middle_author", 0.3
    return "other", 0.1

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable in .env file")
    
    # Run with default config
    topics, narrative = run_pdf_summarization()
