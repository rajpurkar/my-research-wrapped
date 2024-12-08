# Research Paper Summarizer

A powerful tool to automatically summarize and analyze collections of research papers, particularly useful for researchers wanting to organize and understand their publication history or research contributions.

## Features

- 📚 Bulk PDF Processing: Process multiple research papers simultaneously
- 🎯 Smart Author Detection: Automatically identifies your papers and contribution level
- 📊 Topic Clustering: Groups related papers into coherent research themes
- 💡 Intelligent Summarization: Generates concise summaries of individual papers and topic areas
- 👥 Collaboration Analysis: Identifies and analyzes your research collaborators
- 💾 Caching System: Saves processing results for faster subsequent runs

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
├── pdfs/           # Place your PDF files here
├── outputs/        # Generated summaries and analysis
├── summarize_pdf.py # Main script
├── .env           # Environment variables
└── README.md
```

## Usage

1. Place your research papers (PDFs) in the `pdfs/` directory.

2. Configure your settings in the script or use the defaults:
   - `AUTHOR_NAME`: Your name as it appears in publications
   - `MODEL_NAME`: The OpenAI model to use (default: "gpt-4-mini")
   - `NUM_TOPICS`: Number of topic clusters (default: 5)
   - Other settings can be found in the `DEFAULT_CONFIG` dictionary

3. Run the script:
```bash
python summarize_pdf.py
```

The script will:
- Process all PDFs in the `pdfs/` directory
- Generate summaries for each paper
- Group papers into topics
- Create a narrative summary of your research
- Save all results in the `outputs/` directory

## Output Structure

The tool generates several types of output in the `outputs/` directory:
- Individual paper summaries
- Topic-based groupings
- Collaboration analysis
- Overall research narrative

## Customization

You can modify the default configuration by editing the `DEFAULT_CONFIG` dictionary in `summarize_pdf.py`:

```python
DEFAULT_CONFIG = {
    "AUTHOR_NAME": "Your Name",
    "MODEL_NAME": "gpt-4-mini",
    "MODEL_TEMPERATURE": 0.1,
    "NUM_TOPICS": 5,
    "MAX_WORKERS": 16,
}
```

## Tips for Best Results

1. Ensure your PDFs are text-searchable (OCR processed if necessary)
2. Use your full name as it appears in publications
3. Keep your PDFs organized in the `pdfs/` directory
4. Check the `.env` file is properly configured with your API key

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Ensure your OpenAI API key is correctly set in the `.env` file
   - Check that the `.env` file is in the project root directory

2. **PDF Processing Errors**
   - Verify that PDFs are not corrupted
   - Ensure PDFs are text-searchable
   - Check file permissions

3. **Memory Issues**
   - Reduce `MAX_WORKERS` in the configuration
   - Process fewer PDFs at a time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

This tool uses several open-source libraries:
- LangChain for AI interactions
- PyPDF for PDF processing
- Rich for beautiful terminal output 