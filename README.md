# ResearchRadar 🎯

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ResearchRadar is an AI-powered tool that transforms how researchers understand and present their academic contributions. By analyzing your research papers, it automatically discovers meaningful patterns, generates insightful summaries, and weaves your work into a cohesive narrative.

## Why ResearchRadar?

Traditional paper organization is time-consuming and often misses the bigger picture. ResearchRadar changes this by using advanced AI to analyze your entire body of work at once. It identifies research themes, tracks your methodological evolution, and helps you understand your impact across different areas.

The tool excels at finding connections between papers that might not be immediately obvious. Using semantic analysis and topic modeling, it groups related work into meaningful research areas, making it perfect for:

- Preparing research statements or tenure packages
- Understanding your research trajectory
- Identifying emerging themes in your work
- Creating compelling research narratives

## Getting Started

Setting up ResearchRadar is straightforward:

```bash
# Clone and set up the environment
git clone https://github.com/pranavrajpurkar/research-radar.git
cd research-radar
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Add your OpenAI API key to .env

# Run the analysis
python research_radar.py
```

## How It Works

ResearchRadar processes your research papers in three main stages:

1. **Analysis**: Each paper is processed using advanced language models to extract key information, normalize author names, and identify core contributions.

2. **Synthesis**: Papers are grouped into research themes using semantic similarity and topic modeling, ensuring each area represents a coherent line of work.

3. **Narrative**: The tool generates a flowing narrative that connects your research areas, highlighting methodological advances and impact.

## Configuration

ResearchRadar is highly configurable through `config.py`. Here are the key settings:

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
ResearchRadar uses AI to handle the complexities of author names, accounting for variations in formatting, special characters, and academic titles. This ensures consistent author tracking across your publications.

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

We welcome contributions! If you'd like to improve ResearchRadar:

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

Copyright (c) 2024 Pranav Rajpurkar. This project is [MIT](./LICENSE) licensed.

## Acknowledgments

ResearchRadar builds on these excellent open-source projects:
- [LangChain](https://github.com/hwchase17/langchain) for AI orchestration
- [OpenAI](https://github.com/openai/openai-python) for language models
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output