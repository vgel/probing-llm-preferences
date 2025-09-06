### Experiment 2 Code

This folder contains:

### Main Experiment Notebooks
- **`Anthropic_ryff.ipynb`** - Administers Ryff Scale AI to Opus 4, Sonnet 4, and Sonnet 3.7
- **`FreedomGPT_ryff.ipynb`** - Administers Ryff Scale AI to Hermes-LLama 3.1-70b

### Response Parsing Scripts
- **`AnthropicJsonparser.ipynb`** - Extracts numeric ratings from AI responses and separates them into numbers-only and text-only files
- **`DatacleaningHermes.ipynb`** - Same as the Anthropic one, but for Hermes the parser also provides confidence scoring, manual field entry, and improved error handling for number extraction

### Analysis
- **`RyffDATAANALYZER.ipynb`** - Complete tool for statistical analysis specific for our experiment. It calculates Ryff subscale scores and performs statistical analysis on the extracted numeric data
