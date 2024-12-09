# MyResearchWrapped 🎯✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ever wondered what your research story would look like if it got the Spotify Wrapped treatment? MyResearchWrapped is your personal academic year-in-review generator, turning your papers into beautiful, shareable insights about your research in 2024.

## Your Research Story, Beautifully Told

Traditional academic summaries can be dry and one-dimensional. My Research Wrapped changes that by transforming your publications into an engaging, visual story that reveals:

- Your top research areas and their evolution over time 🚀
- Hidden connections between your different papers 🔄
- The impact and reach of your work 📊
- Your unique "research personality" based on your methodological preferences 🎨

Perfect for:
- Sharing your year's achievements on social media
- Adding flair to your tenure package
- Understanding your research evolution
- Discovering surprising patterns in your work

## Get Your Wrapped

```bash
# Clone and set up
git clone https://github.com/pranavrajpurkar/research-wrapped.git
cd research-wrapped
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up your personalization
cp .env.example .env
# Add your OpenAI API key to .env

# Run the analysis
python main.py
```

## How It Works

MyResearchWrapped processes your research papers in three main stages:

1. **Analysis**: Each paper is processed using advanced language models to extract key information, normalize author names, and identify core contributions.

2. **Synthesis**: Papers are grouped into research themes using semantic similarity and topic modeling, ensuring each area represents a coherent line of work.

3. **Narrative**: The tool generates a flowing narrative that connects your research areas, highlighting methodological advances and impact.

## Configuration

MyResearchWrapped is highly configurable through `config.py`. Here are the key settings:

```python
DEFAULT_CONFIG = {
    "AUTHOR_NAME": "Your Name",     # Target author for analysis
    "NUM_TOPICS": 5,                # Number of research areas to identify
    "MAX_WORKERS": 16,              # Parallel processing threads
}
```

## Output Structure

The tool generates a comprehensive analysis in the `outputs` directory:

```
outputs/
├── papers/                 # Individual paper analyses
├── topics/                # Research area summaries
├── research_summary.csv   # Tabulated overview
└── year_in_review.txt     # Narrative synthesis
```

## Advanced Features

### Smart Author Detection
MyResearchWrapped uses AI to handle the complexities of author names, accounting for variations in formatting, special characters, and academic titles. This ensures consistent author tracking across your publications.

### Topic Analysis
The semantic clustering algorithm identifies research themes while maintaining balance - ensuring each area has enough papers to be meaningful, but not so many that distinct contributions are lost.

### Performance Optimization
The tool includes smart caching and parallel processing, making it efficient even with large paper collections. Changes to individual papers trigger selective reprocessing, saving time on subsequent runs.

## Troubleshooting

If you encounter issues:

1. **API Access**: Verify your OpenAI API key is correctly set in `.env`
2. **PDF Processing**: Ensure your PDFs are text-searchable
3. **Performance**: Adjust `MAX_WORKERS` in config.py if needed

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
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output