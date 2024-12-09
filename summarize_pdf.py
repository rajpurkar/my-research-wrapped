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
    "MODEL_NAME": "gpt-4-mini",
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

@dataclass
class Paper:
    """Represents a processed research paper."""
    file_path: str
    summary: str
    weight: float
    processed_time: str
    role: str  # major_author, minor_author, or not_author
    original_text: str = ""
    title: str = "Untitled"
    authors: List[str] = None  # Add authors field

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

@dataclass
class Topic:
    """Represents a group of related papers."""
    name: str
    papers: List[Paper]
    summary: Optional[str] = None
    paper_summaries: Dict[str, str] = None
    collaborators: Dict = None  # Add collaborator information

    def __post_init__(self):
        if self.paper_summaries is None:
            self.paper_summaries = {}
        if self.collaborators is None:
            self.collaborators = {}

class ResearchSummaryManager:
    """Manages the processing and storage of research summaries."""
    
    def __init__(self, output_dir: str = DEFAULT_CONFIG["OUTPUT_DIR"]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a more organized directory structure
        self.cache_dir = self.output_dir / "cache"
        self.summaries_dir = self.output_dir / "summaries"
        self.papers_dir = self.output_dir / "papers"
        self.topics_dir = self.output_dir / "topics"
        self.data_dir = self.output_dir / "data"
        
        # Create all directories
        for directory in [self.cache_dir, self.summaries_dir, 
                         self.papers_dir, self.topics_dir, self.data_dir]:
            directory.mkdir(exist_ok=True)
        
        # Define paths for common files
        self.partial_results_file = self.data_dir / "partial_results.json"
        self.final_summaries_file = self.data_dir / "final_summaries.json"
        self.narrative_file = self.output_dir / "year_in_review_narrative.txt"
    
    def get_paper_cache_path(self, paper_hash: str) -> Path:
        """Get path for individual paper cache file."""
        return self.cache_dir / f"{paper_hash}.json"
    
    def get_paper_path(self, paper_hash: str) -> Path:
        """Get path for processed paper data."""
        return self.papers_dir / f"{paper_hash}.json"
    
    def get_topic_path(self, topic_name: str) -> Path:
        """Get path for topic data."""
        # Sanitize topic name for filesystem
        safe_name = "".join(c for c in topic_name if c.isalnum() or c in (' ', '-', '_')).strip()
        return self.topics_dir / f"{safe_name}.json"
    
    def save_final_narrative(self, narrative: str):
        """Save the final narrative."""
        with open(self.narrative_file, "w") as f:
            f.write(narrative)

    def save_topic(self, topic: Topic):
        """Save topic data to individual file."""
        topic_file = self.get_topic_path(topic.name)
        
        topic_data = {
            "name": topic.name,
            "summary": topic.summary,
            "paper_summaries": topic.paper_summaries,
            "collaborators": topic.collaborators,
            "papers": [
                {
                    "file_path": p.file_path,
                    "file_name": Path(p.file_path).name,
                    "summary": p.summary,
                    "brief_summary": topic.paper_summaries.get(p.file_path, ""),
                    "weight": p.weight,
                    "processed_time": p.processed_time,
                    "role": p.role,
                    "authors": p.authors
                }
                for p in topic.papers
            ]
        }
        
        with open(topic_file, "w") as f:
            json.dump(topic_data, f, indent=4)

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
    if manager.partial_results_file.exists():
        try:
            with open(manager.partial_results_file, "r") as f:
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
    data = {
        "version": DEFAULT_CONFIG["CACHE_VERSION"],
        "entries": results
    }
    with open(manager.partial_results_file, "w") as f:
        json.dump(data, f, indent=4)

def normalize_author_name(name: str) -> str:
    """
    Normalize author name by:
    1. Removing titles and suffixes
    2. Normalizing Unicode characters
    3. Handling middle names/initials
    4. Standardizing format
    """
    # First normalize Unicode characters
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    
    # Remove common titles and suffixes
    titles = [
        "Dr", "Prof", "Professor", "PhD", "Ph.D", "MD", "M.D", 
        "MS", "M.S", "BSc", "B.Sc", "MSc", "M.Sc",
        "MBBS", "MPH", "DrPH", "DPhil", "ScD"
    ]
    
    name = name.strip()
    # Remove periods and commas
    name = re.sub(r'[.,]', '', name)
    
    # Convert to lowercase for comparison
    name_lower = name.lower()
    for title in titles:
        # Remove title with optional period and surrounding spaces
        pattern = rf'\b{title}\.?\s*'
        name_lower = re.sub(pattern, '', name_lower, flags=re.IGNORECASE)
    
    # Split into parts
    parts = name_lower.split()
    if not parts:
        return ""
    
    # Handle middle names/initials
    if len(parts) > 2:
        # Keep first and last name, initialize middle names
        first = parts[0]
        last = parts[-1]
        middles = [p[0] for p in parts[1:-1] if p]  # Get initials of middle names
        if middles:
            return f"{first} {''.join(middles)} {last}".strip()
        return f"{first} {last}"
    
    # Return normalized name
    return ' '.join(parts)

def are_same_author(name1: str, name2: str) -> bool:
    """
    Check if two author names refer to the same person.
    Handles variations in spelling, formatting, and middle names.
    """
    n1 = normalize_author_name(name1)
    n2 = normalize_author_name(name2)
    
    if n1 == n2:
        return True
    
    # Split into parts
    n1_parts = n1.split()
    n2_parts = n2.split()
    
    # Check first and last name match
    if len(n1_parts) > 0 and len(n2_parts) > 0:
        if n1_parts[0] == n2_parts[0] and n1_parts[-1] == n2_parts[-1]:
            return True
    
    return False

def merge_author_names(names: List[str]) -> List[str]:
    """Merge different variations of the same author name."""
    merged = []
    merged_groups = []
    
    for name in names:
        # Check if this name belongs to any existing group
        found = False
        for group in merged_groups:
            if any(are_same_author(name, existing) for existing in group):
                group.add(name)
                found = True
                break
        
        if not found:
            # Create new group
            merged_groups.append({name})
    
    # Take the longest name from each group as the canonical form
    for group in merged_groups:
        canonical = max(group, key=len)
        merged.append(canonical)
    
    return merged

def extract_authors_with_cache(text: str, llm, cache_data: dict) -> List[str]:
    """Extract authors from text using cache data."""
    if cache_data and "authors" in cache_data:
        print(f"[CACHE] Using cached authors")
        return cache_data["authors"]  # Already normalized in cache
    
    authors = extract_authors(text, llm)
    normalized_authors = [normalize_author_name(author) for author in authors]
    return normalized_authors

def categorize_paper(authors: List[str], your_name: str = "Pranav Rajpurkar") -> str:
    """Categorize paper based on author position."""
    if not authors:
        return "unknown"
    if authors[0] == your_name:
        return "first_author"
    elif authors[-1] == your_name:
        return "last_author"
    elif your_name in authors:
        return "middle_author"
    return "other"

def calculate_paper_weight(authors: List[str], your_name: str = "Pranav Rajpurkar") -> float:
    """Calculate paper weight based on author position."""
    if not authors:
        return 0.1
    
    if authors[0] == your_name or authors[-1] == your_name:
        return 1.0  # Equal high weight for first/last author
    elif your_name in authors:
        return 0.3  # Lower weight for middle author
    return 0.1  # Lowest weight for papers where not an author

def group_papers_by_topic(papers: List[Paper], llm, num_topics: int = 5) -> Dict[str, List[Paper]]:
    """Group papers into topics using LLM-based classification."""
    # Calculate target papers per topic
    total_papers = len(papers)
    target_per_topic = total_papers / num_topics
    min_papers = max(1, int(target_per_topic * 0.7))  # Allow 30% fewer than target
    max_papers = int(target_per_topic * 1.3)  # Allow 30% more than target
    
    clustering_prompt = f"""
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
    
    Return ONLY a JSON object with topic names as keys and paper indices as values, like this:
    {{
        "Novel attention mechanisms improve model performance": [0, 1, 2],
        "Automated data validation enhances reliability": [3, 4]
    }}
    """

    # Build the list of papers with their summaries and weights
    paper_list = ""
    for idx, paper in enumerate(papers):
        paper_list += f"\nPaper {idx}:\nWeight: {paper.weight:.1f}\nSummary: {paper.summary}\n"

    full_prompt = clustering_prompt + paper_list + "\nResponse (in JSON format):"

    # Get the clustering result from the LLM
    clustering_result = llm.invoke(full_prompt).content.strip()
    
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

    return topic_groups

def analyze_collaborators(papers: List[Paper], author_name: str) -> Dict[str, List[str]]:
    """Analyze key collaborators across papers in a topic."""
    normalized_author_name = normalize_author_name(author_name)
    
    # Create a table for paper listings
    paper_table = Table(show_header=True, header_style="bold magenta")
    paper_table.add_column("Paper", style="dim", width=30)
    paper_table.add_column("Weight", justify="right")
    paper_table.add_column("Authors", width=50)
    
    # Track collaborators
    author_counts = Counter()
    primary_collaborators = set()  # First/last authors we worked with
    
    for paper in papers:
        # Format authors for display
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += "..."
            
        paper_table.add_row(
            Path(paper.file_path).name,
            f"{paper.weight:.1f}",
            authors_str
        )
        
        # Analyze collaborators
        if paper.authors:
            # Add first and last authors to primary collaborators if we're also primary
            if paper.role == "primary_author":
                if len(paper.authors) > 0:
                    primary_collaborators.add(paper.authors[0])  # First author
                if len(paper.authors) > 1:
                    primary_collaborators.add(paper.authors[-1])  # Last author
            
            # Count all co-authors
            for author in paper.authors:
                if normalize_author_name(author) != normalized_author_name:
                    author_counts[author] += 1
    
    # Remove self from primary collaborators
    primary_collaborators = {
        author for author in primary_collaborators 
        if normalize_author_name(author) != normalized_author_name
    }
    
    # Find frequent collaborators (2+ papers)
    frequent_collaborators = {
        author for author, count in author_counts.items() 
        if count > 1
    }
    
    # Print summary
    console.rule(f"[bold blue]Papers and Collaborations", style="blue")
    console.print(paper_table)
    
    if primary_collaborators or frequent_collaborators:
        collab_text = []
        if primary_collaborators:
            collab_text.append("[bold]Primary Co-Authors:[/bold]")
            collab_text.append("  " + ", ".join(sorted(primary_collaborators)))
        if frequent_collaborators:
            if collab_text:
                collab_text.append("")
            collab_text.append("[bold]Frequent Co-Authors (2+ papers):[/bold]")
            collab_text.append("  " + ", ".join(sorted(frequent_collaborators)))
        
        console.print(Panel(
            "\n".join(collab_text),
            title="Key Collaborators",
            border_style="blue"
        ))
    
    console.print()
    
    return {
        "primary_collaborators": list(primary_collaborators),
        "frequent_collaborators": list(frequent_collaborators),
        "author_counts": dict(author_counts)
    }

def extract_title_with_cache(text: str, llm, cache_data: dict) -> str:
    """Extract paper title from text using cache data."""
    if cache_data and "title" in cache_data:
        print(f"[CACHE] Using cached title")
        return cache_data["title"]
    
    prompt = ChatPromptTemplate.from_template("""
        Extract the title of this research paper. The title might be:
        - At the beginning of the text
        - After keywords like "Title:" or "Paper:"
        - In a header or metadata section
        - In larger or bold font (though you can't see formatting)
        
        Text: {text}
        
        Return ONLY the title text, nothing else.
        If no clear title is found, return "Untitled".
    """)
    
    try:
        title = llm.invoke(prompt.format(text=text[:2000])).content.strip()
        return title
    except Exception as e:
        print(f"[ERROR] Failed to extract title: {e}")
        return "Untitled"

def generate_weighted_topic_summary(papers: List[Paper], llm, author_name: str) -> Tuple[str, Dict[str, str], Dict]:
    """Generate a flowing topic summary, paper summaries, and collaborator analysis."""
    # Sort papers by weight in descending order
    sorted_papers = sorted(papers, key=lambda x: x.weight, reverse=True)
    
    # First, generate brief summaries for each paper
    paper_summaries = {}
    for paper in sorted_papers:
        summary_prompt = ChatPromptTemplate.from_template("""
            Provide a 1-2 sentence technical summary of this paper's key contribution:
            {summary}
            
            Focus on the specific technical innovation and results.
        """)
        brief_summary = llm.invoke(summary_prompt.format(summary=paper.summary))
        paper_summaries[paper.file_path] = brief_summary.content
    
    # Then generate a flowing narrative connecting the papers
    papers_text = "\n\n".join([
        f"Title: {p.title}\nWeight: {p.weight:.1f}\nSummary: {paper_summaries[p.file_path]}"
        for p in sorted_papers
    ])
    
    # Add collaborator analysis
    collaborator_info = analyze_collaborators(papers, author_name)
    
    # Add collaborator information to the narrative prompt
    papers_text += "\n\nKey Collaborators:\n"
    papers_text += f"Primary Co-Authors: {', '.join(collaborator_info['primary_collaborators'])}\n"
    papers_text += f"Frequent Collaborators: {', '.join(collaborator_info['frequent_collaborators'])}\n"
    
    narrative_prompt = ChatPromptTemplate.from_template("""
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
    """)
    
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=narrative_prompt,
        document_prompt=PromptTemplate.from_template("{page_content}"),
        document_variable_name="context"
    )
    
    doc = Document(page_content=papers_text)
    flowing_summary = chain.invoke({"context": [doc]})
    
    return flowing_summary, paper_summaries, collaborator_info

def determine_authorship_role_from_authors(authors: List[str], author_name: str) -> Tuple[str, float]:
    """Determine role and weight based on author position in author list.
    
    Returns:
        Tuple of (role, weight) where:
        - role is one of: "primary_author", "contributing_author", "not_author"
        - weight is 1.0 for primary authors (first/last), 0.5 for others
    """
    normalized_author_name = normalize_author_name(author_name)
    
    if not authors:
        return "not_author", 0.0
    
    # Check if author is in the list at all
    author_present = any(normalize_author_name(author) == normalized_author_name for author in authors)
    if not author_present:
        return "not_author", 0.0
        
    # Primary authorship: first or last author
    if (normalize_author_name(authors[0]) == normalized_author_name or 
        normalize_author_name(authors[-1]) == normalized_author_name):
        return "primary_author", 1.0
    
    # Contributing author: anywhere else in author list
    return "contributing_author", 0.5

def process_pdf(pdf_file: str, manager: ResearchSummaryManager, author_name: str) -> Paper:
    """Process a single PDF and return a Paper object with all metadata."""
    cache_data = load_cache_for_paper(pdf_file, manager)
    
    if "summary" in cache_data:
        # Use cached data...
        filename = Path(pdf_file).name
        if len(filename) > 40:  # Truncate long filenames
            filename = filename[:37] + "..."
        console.print(f"[dim]← Cached: {filename}[/dim]")
        paper = Paper(
            file_path=pdf_file,
            summary=cache_data['summary'],
            weight=cache_data['weight'],
            processed_time=str(Path(pdf_file).stat().st_mtime),
            role=cache_data['role'],
            original_text=cache_data.get('original_text', ''),
            title=cache_data['title'],
            authors=cache_data['authors']
        )
        save_paper_data(paper, manager)
        return paper
    
    # If we're processing new file, show that clearly
    filename = Path(pdf_file).name
    if len(filename) > 40:
        filename = filename[:37] + "..."
    console.print(f"[bold blue]→ Processing: {filename}[/bold blue]")
    
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    
    # Combine all pages into one text
    original_text = "\n".join([doc.page_content for doc in docs]) if docs else ""
    
    # Extract metadata using cache
    authors = extract_authors_with_cache(original_text, llm, {})
    title = extract_title_with_cache(original_text, llm, {})
    
    # Determine role and weight based on authors
    role, weight = determine_authorship_role_from_authors(authors, author_name)
    
    # Generate summary
    summary_prompt = PromptTemplate.from_template("""
        Summarize this research paper in 2-3 sentences, focusing on:
        1. The main contribution or innovation
        2. Key results or findings
        
        Text: {text}
        
        Provide ONLY the summary, no additional text.
    """)
    
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
    summary = chain.invoke(docs)["output_text"]
    
    # Save cache and paper data
    cache_data.update({
        'summary': summary,
        'weight': weight,
        'role': role,
        'title': title,
        'authors': authors,
        'original_text': original_text,
        'version': DEFAULT_CONFIG["CACHE_VERSION"]
    })
    save_cache_for_paper(pdf_file, cache_data, manager)
    
    paper = Paper(
        file_path=pdf_file,
        summary=summary,
        weight=weight,
        processed_time=str(Path(pdf_file).stat().st_mtime),
        role=role,
        original_text=original_text,
        title=title,
        authors=authors
    )
    save_paper_data(paper, manager)
    return paper

def summarize_pdfs_from_folder(pdfs_folder: str, author_name: str, num_topics: int = 5, status=None) -> List[Topic]:
    """Process PDFs and return a list of Topics."""
    manager = ResearchSummaryManager()
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
            executor.submit(process_pdf, pdf_file, manager, author_name): pdf_file 
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

    # Group papers by topic
    topic_groups = group_papers_by_topic(papers, llm, num_topics=num_topics)
    
    if not topic_groups:
        raise ValueError("Failed to group papers into topics")

    # Generate summaries for each topic
    final_summaries = {}
    final_paper_summaries = {}
    final_collaborators = {}
    for topic, papers in topic_groups.items():
        console.rule(f"[bold cyan]Topic: {topic}")
        
        # Create a table for papers
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Paper", style="dim")
        table.add_column("Weight", justify="right")
        table.add_column("Authors")
        
        for paper in papers:
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += "..."
            
            table.add_row(
                Path(paper.file_path).name,
                f"{paper.weight:.1f}",
                authors_str
            )
        
        console.print(table)
        console.print()

        if status:
            status.update(f"[bold green]Generating summary for: {topic}")

        try:
            topic_summary, paper_summaries, collaborator_info = generate_weighted_topic_summary(
                papers, llm, author_name
            )
            final_summaries[topic] = topic_summary
            final_paper_summaries[topic] = paper_summaries
            final_collaborators[topic] = collaborator_info
            
            # Print collaborator panel
            collaborator_panel = Panel(
                "\n".join([
                    "[bold]Primary Co-Authors:[/bold]",
                    "  " + ", ".join(collaborator_info['primary_collaborators']),
                    "",
                    "[bold]Frequent Collaborators:[/bold]",
                    "  " + ", ".join(collaborator_info['frequent_collaborators'])
                ]),
                title="Collaborations",
                border_style="blue"
            )
            console.print(collaborator_panel)
            
        except Exception as e:
            console.print(f"[red]Error generating summary for {topic}: {e}[/red]")
            final_summaries[topic] = "Error generating summary"
            final_paper_summaries[topic] = {}
            final_collaborators[topic] = {}

    # Write final summaries to file
    with open(manager.final_summaries_file, "w") as f:
        json.dump(final_summaries, f, indent=4)

    # Convert to Topic objects
    topics = [
        Topic(
            name=topic_name,
            papers=topic_papers,
            summary=final_summaries[topic_name],
            paper_summaries=final_paper_summaries[topic_name],
            collaborators=final_collaborators[topic_name]
        )
        for topic_name, topic_papers in topic_groups.items()
    ]
    
    for topic in topics:
        manager.save_topic(topic)
    
    return topics

def generate_narrative(topics: List[Topic]) -> str:
    """Generate a cohesive narrative connecting research themes and contributions."""
    theme_titles = [topic.name for topic in topics]
    
    # 1. Technical Overview
    overview_prompt = ChatPromptTemplate.from_template("""
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
    """)
    
    overview = llm.invoke(overview_prompt.format(themes=", ".join(theme_titles)))
    narrative = f"{overview.content}\n\n"

    # 2. Detailed Topic Discussions
    for topic in topics:
        narrative += f"### {topic.name}\n\n"
        
        # Add context sentence
        context_prompt = ChatPromptTemplate.from_template("""
            Write a single technical sentence that:
            1. Places this research claim in context: {theme}
            2. Connects it to broader technical challenges
            3. Highlights its methodological significance
            
            Be specific about technical aspects and avoid generic statements.
        """)
        
        context = llm.invoke(context_prompt.format(theme=topic.name))
        narrative += f"{context.content}\n\n"
        
        # Use the existing detailed topic summary
        narrative += f"{topic.summary}\n\n"
        
        # Add collaborator insights
        if topic.collaborators:
            narrative += "Key Technical Collaborations:\n"
            if topic.collaborators.get('primary_collaborators'):
                narrative += f"- Led joint technical development with: {', '.join(topic.collaborators['primary_collaborators'])}\n"
            if topic.collaborators.get('frequent_collaborators'):
                narrative += f"- Sustained technical partnerships with: {', '.join(topic.collaborators['frequent_collaborators'])}\n"
            narrative += "\n"

    # 3. Future Directions
    conclusion_prompt = ChatPromptTemplate.from_template("""
        Write a technical conclusion (2-3 sentences) that:
        1. Identifies specific methodological challenges remaining in: {themes}
        2. Suggests concrete technical approaches for future work
        3. Highlights opportunities for combining methods across themes
        
        Focus on technical aspects and methodological innovations.
    """)
    
    conclusion = llm.invoke(conclusion_prompt.format(themes=", ".join(theme_titles)))
    narrative += f"\n{conclusion.content}"

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

def extract_authors(text: str, llm) -> List[str]:
    """Extract author names from text using LLM."""
    prompt = ChatPromptTemplate.from_template("""
        Extract the COMPLETE author names from this text. Authors may be listed in various formats.
        
        Requirements:
        1. Each name MUST have both first and last name components
        2. Convert all names to "First [Middle] Last" format
        3. Remove all titles (Dr., Prof., PhD, etc.)
        4. Remove all affiliations and numbers
        5. Keep middle names/initials if present
        6. Expand any abbreviated first names if found in the text
        
        Text: {text}
        
        Return ONLY a JSON array of complete author names, like this:
        ["John Andrew Smith", "Maria R Rodriguez", "David Michael Chang"]
        
        Critical rules:
        - Never include single-word names or just last names
        - Always look for full first names when only initials are given
        - If a name seems incomplete, search the full text for the complete version
        - If you can't find a complete name, skip it
        - If no valid authors are found, return an empty array: []
    """)
    
    try:
        result = llm.invoke(prompt.format(text=text)).content.strip()
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        authors = json.loads(result)
        
        # Validate each author name
        validated_authors = []
        for author in authors:
            is_valid, warning = validate_author_name(author)
            if is_valid:
                validated_authors.append(author)
            else:
                console.print(f"[yellow]Skipping invalid author: {warning}[/yellow]")
        
        if not validated_authors:
            console.print("[red]Warning: No valid authors found in text[/red]")
        
        return validated_authors
    except Exception as e:
        console.print(f"[red]Error extracting authors: {e}[/red]")
        return []

def run_pdf_summarization(config: dict = None):
    """Main function to run the PDF summarization pipeline with configuration."""
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
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
        narrative = generate_narrative(topics)
        
        manager = ResearchSummaryManager(output_dir=cfg["OUTPUT_DIR"])
        manager.save_final_narrative(narrative)
    
    console.print("[bold green]✓[/bold green] Processing complete!")
    return topics, narrative

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable in .env file")
    
    # Run with default config
    topics, narrative = run_pdf_summarization()
