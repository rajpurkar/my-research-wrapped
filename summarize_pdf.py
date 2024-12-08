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

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from collections import defaultdict
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

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
    "MAX_WORKERS": 8,  # For ThreadPoolExecutor
    
    # Cache settings
    "CACHE_VERSION": "1.0",
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
        
        # Create subdirectories for different outputs
        self.summaries_dir = self.output_dir / "summaries"
        self.data_dir = self.output_dir / "data"
        
        self.summaries_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Define paths for common files
        self.cache_file = self.data_dir / "papers_cache.json"
        self.partial_results_file = self.data_dir / "partial_results.json"
        self.final_summaries_file = self.data_dir / "final_summaries.json"
        self.narrative_file = self.output_dir / "year_in_review_narrative.txt"
    
    def save_paper(self, paper: Paper):
        """Save processed paper information."""
        paper_data = {
            "file_path": paper.file_path,
            "summary": paper.summary,
            "weight": paper.weight,
            "processed_time": paper.processed_time,
            "role": paper.role
        }
        
        # Use paper's hash as filename
        paper_hash = get_cache_key(paper.file_path)
        paper_file = self.cache_dir / f"{paper_hash}.json"
        
        with open(paper_file, "w") as f:
            json.dump(paper_data, f, indent=4)
    
    def save_topic(self, topic: Topic):
        """Save topic information."""
        topic_data = {
            "name": topic.name,
            "papers": [
                {
                    "file_path": p.file_path,
                    "file_name": Path(p.file_path).name,
                    "summary": p.summary,
                    "brief_summary": topic.paper_summaries.get(p.file_path, ""),
                    "weight": p.weight,
                    "processed_time": p.processed_time,
                    "role": p.role
                }
                for p in topic.papers
            ],
            "summary": topic.summary,
            "collaborators": topic.collaborators
        }
        
        topic_file = self.summaries_dir / f"{topic.name}.json"
        with open(topic_file, "w") as f:
            json.dump(topic_data, f, indent=4)
    
    def save_final_narrative(self, narrative: str):
        """Save the final narrative."""
        with open(self.narrative_file, "w") as f:
            f.write(narrative)

def get_cache_key(pdf_path):
    """Generate a cache key based on file path and last modified time."""
    pdf_stats = os.stat(pdf_path)
    content_key = f"{pdf_path}:{pdf_stats.st_mtime}"
    return hashlib.md5(content_key.encode()).hexdigest()

def load_cache():
    """Load the cache from disk with version checking."""
    manager = ResearchSummaryManager()
    if manager.cache_file.exists():
        try:
            with open(manager.cache_file, "r") as f:
                cache_data = json.load(f)
            
            if cache_data.get("version") != DEFAULT_CONFIG["CACHE_VERSION"]:
                print(f"[CACHE] Outdated cache version. Expected {DEFAULT_CONFIG['CACHE_VERSION']}, found {cache_data.get('version')}. Rebuilding cache.")
                return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}
            
            return cache_data["entries"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[CACHE] Error loading cache: {e}. Rebuilding cache.")
            return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}
    return {"version": DEFAULT_CONFIG["CACHE_VERSION"], "entries": {}}

def save_cache(cache):
    """Save the cache to disk with version information."""
    manager = ResearchSummaryManager()
    cache_data = {
        "version": DEFAULT_CONFIG["CACHE_VERSION"],
        "entries": cache
    }
    with open(manager.cache_file, "w") as f:
        json.dump(cache_data, f, indent=4)

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
    clustering_prompt = f"""
    You will be provided with a list of research paper summaries along with their weights.
    Please cluster these papers into exactly {num_topics} topics, each representing a clear research claim or contribution.
    
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
    - Group 2-4 papers that support this claim
    - Include at least one major paper (weight 1.0)
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
    """Analyze collaborators using pre-extracted author lists."""
    normalized_author_name = normalize_author_name(author_name)
    
    # Get all authors from papers
    paper_authors = {}
    all_authors = set()
    
    for paper in papers:
        # Use pre-extracted authors from Paper object
        authors = [
            a for a in paper.authors 
            if not are_same_author(a, author_name)
        ]
        paper_authors[paper.file_path] = authors
        all_authors.update(authors)
        print(f"[DEBUG] Authors for {Path(paper.file_path).name}: {authors}")
    
    # Merge author name variations
    all_authors = merge_author_names(list(all_authors))
    
    # Update paper_authors with merged names
    for paper_path, authors in paper_authors.items():
        merged_authors = []
        for author in authors:
            for canonical in all_authors:
                if are_same_author(author, canonical):
                    merged_authors.append(canonical)
                    break
        paper_authors[paper_path] = merged_authors
    
    # Find major authors (first/last/co-first/co-senior)
    major_authors = set()
    for authors in paper_authors.values():
        if authors:  # Check if list is not empty
            major_authors.add(authors[0])  # First author
            if len(authors) > 1:
                major_authors.add(authors[-1])  # Last/senior author
    
    # Count author appearances
    author_counts = Counter()
    for authors in paper_authors.values():
        author_counts.update(authors)
    
    # Find frequent collaborators (appear in more than one paper)
    frequent_collaborators = {
        author for author, count in author_counts.items() 
        if count > 1 and author != author_name
    }
    
    return {
        "major_authors": list(major_authors),
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
    papers_text += f"Major Authors: {', '.join(collaborator_info['major_authors'])}\n"
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

def process_pdf(pdf_file: str, cache: dict, author_name: str) -> Paper:
    """Process a single PDF and return a Paper object with all metadata."""
    cache_key = get_cache_key(pdf_file)
    if cache_key in cache:
        try:
            cached_data = cache[cache_key]
            print(f"[CACHE] Using cached data for: {pdf_file}")
            return Paper(
                file_path=pdf_file,
                summary=cached_data['summary'],
                weight=cached_data['weight'],
                processed_time=str(Path(pdf_file).stat().st_mtime),
                role=cached_data['role'],
                original_text=cached_data['original_text'],
                title=cached_data['title'],
                authors=cached_data['authors']
            )
        except Exception as e:
            print(f"[CACHE] Error reading cache for {pdf_file}: {e}. Regenerating.")
    
    print(f"[NEW] Processing: {pdf_file}")
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    
    # Combine all pages into one text
    original_text = "\n".join([doc.page_content for doc in docs]) if docs else ""
    
    # Create empty cache data for this file
    file_cache = {}
    
    # Extract metadata using cache
    authors = extract_authors_with_cache(original_text, llm, file_cache)
    title = extract_title_with_cache(original_text, llm, file_cache)
    
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
    
    # Cache all paper data
    cache[cache_key] = {
        'summary': summary,
        'weight': weight,
        'role': role,
        'original_text': original_text,
        'title': title,
        'authors': authors  # Already normalized
    }
    save_cache(cache)
    
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

def summarize_pdfs_from_folder(pdfs_folder: str, author_name: str, num_topics: int = 5) -> List[Topic]:
    """Process PDFs and return a list of Topics."""
    manager = ResearchSummaryManager()
    papers: List[Paper] = []
    cache = load_cache()
    partial_results = load_partial_results()

    pdf_files = glob.glob(pdfs_folder + "/*.pdf")
    total_pdfs = len(pdf_files)
    processed_count = 0
    save_threshold = max(1, min(5, total_pdfs // 10))  # Save every 10% or at least every 5 papers

    print(f"[INFO] Found {total_pdfs} PDFs to process")

    with ThreadPoolExecutor(max_workers=DEFAULT_CONFIG["MAX_WORKERS"]) as executor:
        futures = {
            executor.submit(process_pdf, pdf_file, cache, author_name): pdf_file 
            for pdf_file in pdf_files
        }
        
        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                paper = future.result()
                papers.append(paper)

                # Update partial results
                partial_results[pdf_file] = {
                    "file_path": paper.file_path,
                    "summary": paper.summary,
                    "weight": paper.weight,
                    "processed_time": paper.processed_time,
                    "role": paper.role
                }

                processed_count += 1
                # Save partial results periodically instead of every time
                if processed_count % save_threshold == 0:
                    save_partial_results(partial_results)
                    print(f"[PROGRESS] Processed {processed_count}/{total_pdfs} PDFs")

            except Exception as e:
                print(f"[ERROR] Processing {pdf_file}: {e}")
                print(f"[DEBUG] Error details: {str(e)}")

    # Save final partial results
    if partial_results:
        save_partial_results(partial_results)

    if not papers:
        raise ValueError("No summaries were generated. Check the error messages above.")

    # Group papers by topic
    topic_groups = group_papers_by_topic(papers, llm, num_topics=num_topics)
    
    if not topic_groups:
        raise ValueError("Failed to group papers into topics")

    # Generate summaries for each topic
    final_summaries = {}
    final_paper_summaries = {}
    final_collaborators = {}
    for topic, papers in topic_groups.items():
        print(f"[TOPIC] Summarizing papers about {topic}")
        print(f"Papers in this topic: {[Path(p.file_path).name for p in papers]}")
        try:
            topic_summary, paper_summaries, collaborator_info = generate_weighted_topic_summary(
                papers, 
                llm, 
                author_name
            )
            final_summaries[topic] = topic_summary
            final_paper_summaries[topic] = paper_summaries
            final_collaborators[topic] = collaborator_info
            
            # Print collaborator information
            print("\nKey Collaborators:")
            print(f"Major Authors: {', '.join(collaborator_info['major_authors'])}")
            print(f"Frequent Collaborators: {', '.join(collaborator_info['frequent_collaborators'])}")
        except Exception as e:
            print(f"[ERROR] Failed to generate summary for topic {topic}: {e}")
            final_summaries[topic] = "Error generating summary"
            final_paper_summaries[topic] = {}
            final_collaborators[topic] = {}

    # Write final summaries to file
    with open(manager.final_summaries_file, "w") as f:
        json.dump(final_summaries, f, indent=4)

    # Convert to Topic objects with collaborator information
    topics: List[Topic] = []
    for topic_name, topic_papers in topic_groups.items():
        topic = Topic(
            name=topic_name,
            papers=topic_papers,
            summary=final_summaries[topic_name],
            paper_summaries=final_paper_summaries[topic_name],
            collaborators=final_collaborators[topic_name]
        )
        topics.append(topic)
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
            if topic.collaborators.get('major_authors'):
                narrative += f"- Led joint technical development with: {', '.join(topic.collaborators['major_authors'])}\n"
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

def extract_authors(text: str, llm) -> List[str]:
    """Extract author names from text using LLM."""
    prompt = ChatPromptTemplate.from_template("""
        Extract the full list of authors from this text. Authors may be listed in various formats:
        - First Last
        - Last, First
        - With titles (Dr., Prof., etc.)
        - With affiliations
        - With symbols for corresponding authors
        - With footnotes for equal contribution
        - With department/institution numbers (e.g., ¹, ², etc.)
        
        Look for:
        - Author lists at the start of the paper
        - Corresponding author notations
        - Equal contribution footnotes
        - Author sections or blocks
        
        Text: {text}
        
        Return ONLY a JSON array of author names in "First Last" format, like this:
        ["John Smith", "Jane Doe"]
        
        Important:
        - Include all authors in the order they appear
        - Remove affiliations, numbers, and symbols
        - Keep the names in a consistent format
        - If no authors are found, return an empty array: []
    """)
    
    try:
        result = llm.invoke(prompt.format(text=text)).content.strip()
        # Extract JSON array from response (in case there's additional text)
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        return json.loads(result)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[ERROR] Failed to parse authors: {e}")
        print(f"[DEBUG] Raw LLM response:\n{result}")
        return []

def run_pdf_summarization(config: dict = None):
    """Main function to run the PDF summarization pipeline with configuration."""
    # Merge provided config with defaults
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Initialize OpenAI client
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=cfg["MODEL_NAME"],
        temperature=cfg["MODEL_TEMPERATURE"]
    )
    
    # Initialize manager with configured output directory
    manager = ResearchSummaryManager(output_dir=cfg["OUTPUT_DIR"])
    
    print(f"[START] Summarizing PDFs from folder: {cfg['PDF_FOLDER']}")
    print(f"[CONFIG] Analyzing contributions for author: {cfg['AUTHOR_NAME']}")
    
    topics = summarize_pdfs_from_folder(
        pdfs_folder=cfg["PDF_FOLDER"],
        author_name=cfg["AUTHOR_NAME"],
        num_topics=cfg["NUM_TOPICS"]
    )
    
    # Generate and save narrative
    narrative = generate_narrative(topics)
    manager.save_final_narrative(narrative)
    
    print("[DONE] Processing complete. Narrative generated successfully.")
    return topics, narrative

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable in .env file")
    
    # Run with default config
    topics, narrative = run_pdf_summarization()
