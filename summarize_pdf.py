import glob
import os
import json
import hashlib
from pathlib import Path
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from collections import defaultdict
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in .env file")

llm = ChatOpenAI(
    api_key=api_key,  # Pass key explicitly
    model="gpt-4o-mini",
    temperature=0.2
)

def get_cache_key(pdf_path):
    """Generate a cache key based on file path and last modified time."""
    pdf_stats = os.stat(pdf_path)
    content_key = f"{pdf_path}:{pdf_stats.st_mtime}"
    return hashlib.md5(content_key.encode()).hexdigest()

def load_cache():
    """Load the cache from disk."""
    cache_file = Path("summaries_cache.json")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save the cache to disk."""
    with open("summaries_cache.json", "w") as f:
        json.dump(cache, f)

def load_partial_results():
    """Load partial results from disk."""
    partial_file = Path("partial_results.json")
    if partial_file.exists():
        with open(partial_file, "r") as f:
            return json.load(f)
    return {}

def save_partial_results(results):
    """Save partial results to disk."""
    with open("partial_results.json", "w") as f:
        json.dump(results, f)

def extract_authors(text: str) -> List[str]:
    """Extract author names from text using regex patterns."""
    author_pattern = r'(?:Authors?|Contributors?):?\s*((?:[A-Z][a-z]+\s+[A-Z][a-z]+(?:,?\s+(?:and\s+)?)?)+)'
    matches = re.findall(author_pattern, text, re.IGNORECASE)
    if matches:
        authors = [name.strip() for name in matches[0].split(',')]
        # Clean up "and" and extra spaces
        authors = [re.sub(r'\s+and\s+', '', name).strip() for name in authors]
        return authors
    return []

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

def group_papers_by_topic(summaries: List[Tuple[str, str]], llm, num_topics: int = 5) -> Dict[str, List[Dict]]:
    """Group papers into a specified number of topics using LLM-based classification."""
    # Prepare the data
    papers_with_info = []
    for original_text, summary in summaries:
        authors = extract_authors(original_text)
        weight = calculate_paper_weight(authors)
        papers_with_info.append({
            "summary": summary,
            "authors": authors,
            "weight": weight
        })
    
    # Construct the prompt for clustering with explicit JSON format instructions
    clustering_prompt = f"""
    You will be provided with a list of research paper summaries along with their weights.
    Please cluster these papers into exactly {num_topics} topics.
    Each topic should be as specific as possible while still encompassing the papers.
    Ensure that each topic includes at least one major paper (first or last author) based on the weights (weight 1.0 indicates major paper).

    Return ONLY a JSON object with this exact format:
    {{
        "topic_name_1": [0, 1, 2],
        "topic_name_2": [3, 4],
        ...
    }}
    where the numbers are the indices of the papers (starting from 0) that belong to each topic.
    Topic names should be brief but descriptive. Use only alphanumeric characters, spaces, and underscores in topic names.

    Here are the papers:
    """

    # Build the list of papers with their summaries and weights
    paper_list = ""
    for idx, paper in enumerate(papers_with_info):
        paper_list += f"\nPaper {idx}:\nWeight: {paper['weight']:.1f}\nSummary: {paper['summary']}\n"

    full_prompt = clustering_prompt + paper_list + "\nResponse (in JSON format):"

    # Get the clustering result from the LLM and attempt to extract JSON
    clustering_result = llm.invoke(full_prompt).content.strip()
    
    # Try to find JSON content if there's any extra text
    try:
        # Look for content between curly braces
        json_match = re.search(r'\{.*\}', clustering_result, re.DOTALL)
        if json_match:
            clustering_result = json_match.group(0)
        
        topic_groups_indices = json.loads(clustering_result)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[ERROR] Failed to parse clustering result: {e}")
        print(f"[DEBUG] Raw clustering result:\n{clustering_result}")
        # Fallback: create a single topic with all papers
        topic_groups_indices = {"uncategorized_papers": list(range(len(papers_with_info)))}

    # Build the final topic groups
    topic_groups = {}
    for topic, indices in topic_groups_indices.items():
        topic_groups[topic] = [papers_with_info[int(idx)] for idx in indices]

    return topic_groups

def generate_weighted_topic_summary(papers: List[Dict], llm) -> str:
    """Generate a weighted summary for papers in a topic."""
    # Sort papers by weight in descending order
    sorted_papers = sorted(papers, key=lambda x: x['weight'], reverse=True)
    
    # Create the main prompt template
    prompt = ChatPromptTemplate.from_template("""
        Analyze these research papers, giving equal importance to first and last author papers, 
        and less importance to middle author papers.
        
        Papers and their weights: {context}
        
        Provide:
        1. Theme: Core focus of these papers (2-3 sentences)
        2. Key Contributions: Main findings/contributions, emphasizing first/last author papers equally
    """)
    
    # Create document prompt template
    document_prompt = PromptTemplate.from_template("{page_content}")
    
    # Format papers for input
    papers_text = "\n".join([
        f"Weight {p['weight']:.1f} paper: {p['summary']}" 
        for p in sorted_papers
    ])
    
    # Create and run the chain with proper configuration
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt,
        document_variable_name="context"
    )
    
    # Create document object and invoke chain
    doc = Document(page_content=papers_text)
    result = chain.invoke({"context": [doc]})
    return result

def process_pdf(pdf_file: str, cache: dict) -> Tuple[str, str]:
    """Process a single PDF: load, summarize, cache results.

    Returns:
        (original_text, summary)
    """
    cache_key = get_cache_key(pdf_file)
    if cache_key in cache:
        print(f"[CACHE] Using cached summary for: {pdf_file}")
        summary = cache[cache_key]
        # We still need the original text
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        original_text = docs[0].page_content if docs else ""
    else:
        print(f"[NEW] Generating summary for: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.invoke(docs)["output_text"]
        cache[cache_key] = summary
        save_cache(cache)
        original_text = docs[0].page_content if docs else ""

    return (original_text, summary)

def summarize_pdfs_from_folder(pdfs_folder, num_topics=5):
    summaries = []
    cache = load_cache()
    partial_results = load_partial_results()

    pdf_files = glob.glob(pdfs_folder + "/*.pdf")
    total_pdfs = len(pdf_files)
    processed_count = 0

    # Use a ThreadPoolExecutor to process PDFs in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_pdf, pdf_file, cache): pdf_file for pdf_file in pdf_files}
        for future in as_completed(futures):
            pdf_file = futures[future]
            try:
                original_text, summary = future.result()
                summaries.append((original_text, summary))

                # Update partial results
                partial_results[pdf_file] = summary
                save_partial_results(partial_results)

                processed_count += 1
                print(f"[PROGRESS] Processed {processed_count}/{total_pdfs} PDFs")

            except Exception as e:
                print(f"[ERROR] Processing {pdf_file}: {e}")

    # Group papers by topic with the specified number of topics
    topic_groups = group_papers_by_topic(summaries, llm, num_topics=num_topics)
    
    # Generate weighted summaries for each topic
    final_summaries = {}
    for topic, papers in topic_groups.items():
        print(f"[TOPIC] Summarizing papers about {topic}")
        topic_summary = generate_weighted_topic_summary(papers, llm)
        final_summaries[topic] = topic_summary

    # Write final summaries to file
    with open("final_summaries.json", "w") as f:
        json.dump(final_summaries, f, indent=4)

    return final_summaries

# Modify main execution
if __name__ == "__main__":
    folder_path = "pdfs"  # Replace with your PDF folder path
    num_topics = 5  # Set the desired number of topics
    print("[START] Summarizing PDFs from folder:", folder_path)
    topic_summaries = summarize_pdfs_from_folder(folder_path, num_topics=num_topics)
    
    print("\n[RESULT] Summaries by Research Topic:")
    print("====================================")
    for topic, summary in topic_summaries.items():
        print(f"\n{topic.upper()}:")
        print("----------------")
        print(summary)
        print()
    print("[DONE] All PDFs processed and topic summaries generated.")
