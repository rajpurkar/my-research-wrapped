# Research Paper Summarizer

A powerful tool to automatically summarize and analyze collections of research papers, particularly useful for researchers wanting to organize and understand their publication history or research contributions.

## Features

- 📚 Bulk PDF Processing: Process multiple research papers simultaneously
- 🎯 Smart Author Detection: Automatically identifies and normalizes author names
- 📊 Topic Clustering: Groups related papers into coherent research themes
- 💡 Intelligent Summarization: Generates concise technical summaries of papers and research areas
- 🔄 Caching System: Efficient caching with version control for faster subsequent runs
- 🎨 Rich Terminal Output: Beautiful progress indicators and formatted output

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- PDF files of research papers

## Installation

1. Clone this repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install langchain langchain-openai python-dotenv rich
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Directory Structure

```
.
├── pdfs/              # Place your PDF files here
├── outputs/           # Generated outputs
│   ├── cache/        # Cached processing results
│   ├── summaries/    # Individual paper summaries
│   ├── papers/       # Processed paper data
│   ├── topics/       # Topic-based groupings
│   └── data/         # Additional analysis data
├── summarize_pdf.py   # Main script
├── .env              # Environment variables
└── README.md
```

## Configuration

The tool uses a configuration system defined in `DEFAULT_CONFIG`:

```python
DEFAULT_CONFIG = {
    # Author settings
    "AUTHOR_NAME": "Your Name",
    
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
```

## Usage

1. Place your research papers (PDFs) in the `pdfs/` directory.

2. Run the script:
```bash
python summarize_pdf.py
```

The script will:
- Process all PDFs in parallel using ThreadPoolExecutor
- Generate technical summaries for each paper
- Group papers into coherent research topics
- Create a flowing narrative connecting the research themes
- Cache results for faster subsequent runs

## Output Files

The tool generates several types of output in the `outputs/` directory:

- `cache/`: Cached processing results with version control
- `papers/`: Individual paper data and summaries
- `topics/`: Topic-based groupings and analyses
- `data/`: Processing metadata and partial results
- `year_in_review_narrative.txt`: Overall research narrative

## Advanced Features

### Author Name Normalization
- Intelligent handling of author names
- Removes titles, middle names, and special characters
- Maintains consistent capitalization

### Topic Clustering
- Groups papers into coherent research themes
- Ensures balanced topic distribution
- Generates technical narratives connecting papers

### Caching System
- Version-controlled caching
- Efficient reprocessing of modified files
- Maintains processing state across runs

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Ensure your OpenAI API key is correctly set in the `.env` file
   - Check that python-dotenv is properly installed

2. **PDF Processing Errors**
   - Verify PDFs are text-searchable
   - Check file permissions
   - Ensure PDFs are not corrupted

3. **Memory Issues**
   - Adjust `MAX_WORKERS` in configuration
   - Process fewer PDFs at a time

4. **Cache Issues**
   - Clear the `outputs/cache/` directory if experiencing inconsistencies
   - Check `CACHE_VERSION` in configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

This tool uses several open-source libraries:
- LangChain for AI interactions
- PyPDF for PDF processing
- Rich for terminal output formatting
- python-dotenv for environment management