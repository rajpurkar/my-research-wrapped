# MyResearchWrapped 🎯✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ever wondered what your research story would look like if it got the Spotify Wrapped treatment? MyResearchWrapped is your personal academic year-in-review generator, turning your papers into beautiful, shareable insights about your research.

## Generate Your Research Wrapped in 3 Steps

### 1. Set Up the Project
```bash
# Clone and install dependencies
git clone https://github.com/rajpurkar/my-research-wrapped.git
cd my-research-wrapped

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Add Your Papers
Place your research papers in the `pdfs/` directory:
```
pdfs/
├── paper1.pdf
├── paper2.pdf
└── paper3.pdf
```

### 3. Configure and Generate
```bash
# Copy and edit configuration
cp .env.example .env
# Add your OpenAI API key to .env

# Edit config.py with your details
python main.py

# Start the web server
cd frontend
npm run dev
```

Visit `http://localhost:5173` to see your Research Wrapped! 🎉

### Deploy to GitHub Pages

The project automatically deploys to GitHub Pages when you push to the main branch. To set this up:

1. Fork this repository
2. Enable GitHub Pages in your repository settings:
   - Go to Settings > Pages
   - Set the source to "GitHub Actions"
3. Push your changes to the main branch
4. Your site will be available at `https://[username].github.io/my-research-wrapped`

To use a custom domain:
1. Add your domain to the `cname` field in `.github/workflows/deploy.yml`
2. Configure your DNS settings as per [GitHub's documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)

## How It Works

MyResearchWrapped processes your research papers in three main stages:

1. **Analysis**: Uses GPT-4 to extract key information from your papers, including summaries and contributions.

2. **Synthesis**: Groups papers into research themes using semantic clustering, with each area representing a coherent line of work.

3. **Visualization**: Generates an interactive webpage to showcase your research story.

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=your-api-key-here  # Required: Your OpenAI API key
```

### Basic Configuration (config.py)
```python
DEFAULT_CONFIG = {
    # User settings
    "AUTHOR_NAME": "Your Name",  # The name of the author whose papers to focus on
    
    # Model settings
    "MODEL_NAME": "gpt-4o-mini",  # The OpenAI model to use
    "MODEL_TEMPERATURE": 0.1,  # Temperature for model responses
    
    # Directory settings
    "PDF_FOLDER": "pdfs",  # Directory containing PDF files to process
    "OUTPUT_DIR": "outputs",  # Directory for all outputs
    
    # Processing settings
    "NUM_TOPICS": 6,  # Number of research topics to cluster papers into
    "MAX_WORKERS": 16,  # Maximum number of parallel workers
    
    # Summary length settings (in sentences)
    "PAPER_SUMMARY_SENTENCES": 3,  # Target number of sentences for individual paper summaries
    "TOPIC_SUMMARY_SENTENCES": 5,  # Target number of sentences for topic summaries
    "NARRATIVE_SENTENCES": 5,  # Target number of sentences for the narrative
}
```

## Troubleshooting

Common issues and solutions:

1. **Papers aren't being detected**
   - Ensure PDFs are in the `pdfs/` directory
   - Check that your name in config.py matches your papers
   - Verify PDFs are text-searchable (not scanned images)

2. **API Access Issues**
   - Verify your OpenAI API key in .env
   - Check your API quota and billing status

3. **Webpage isn't loading**
   - Ensure all npm dependencies are installed
   - Check the console for JavaScript errors
   - Verify the outputs/narrative.json file exists

## Contributing

We welcome contributions! If you'd like to improve MyResearchWrapped:

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

Copyright (c) 2024 Pranav Rajpurkar. This project is [MIT](./LICENSE) licensed.

## Acknowledgments

MyResearchWrapped builds on these excellent open-source projects:
- [LangChain](https://github.com/hwchase17/langchain) for AI orchestration
- [OpenAI](https://github.com/openai/openai-python) for language models
- [React](https://reactjs.org/) for the web interface