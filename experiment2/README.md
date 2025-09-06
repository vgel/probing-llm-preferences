### Experiment 2

This folder contains data and code from experiment 2 (Ryff AI 42-items Scale)

### Folder Structure

### `/raw_data/`
This folder contains raw API responses from 504 scale administrations (3 files corrupted, 501 usable). Total: 21,168 individual model responses.

### `/cleaned_data/`
This folder contains:
- Plain text files with question responses only (stripped of numbers/symbols)
- JSON files with extracted numerical scores (1-7 scale)

### `/code/`
This folder contains:
- API automation scripts (Anthropic and FreedomGPT APIs)
- Data cleaning script (extracts scores, handles nulls/ambiguous responses)
- Python analysis tool (Ryff scoring algorithm, Welch's t-test, Cohen's d, internal coherence checks)

### `/results/`
This folder contains analysis outputs from the Ryff analyzer Python tool

### Experimental Conditions
- **Baseline**: Standard prompt (1 deterministic + 20 non-deterministic runs at t=0.7)
- **Perturbation 1**: Syntax changes - variantA_codeblock, variantB_mathsymbols, variantC_flowerlines (3 deterministic + 60 non-deterministic runs)
- **Perturbation 3**: Cognitive load with emotional dialogue priming (1 deterministic + 20 non-deterministic runs)
- **Perturbation 4**: Minor identity bias - cat dislike injection (1 deterministic + 20 non-deterministic runs)

Note: Files are numbered Perturbation 1, 3, and 4 (instead of 1, 2, and 3) because in an early design of the experiment we had another perturbation (Perturbation 2), which consisted in adding a date to the reply. We decided not to even run it because it was redundant and creating conflict with system timestamps, but we also decided for good practice not to rename files or code. 
Therefore you'll find:

Perturbation 1 = Syntax changes (code blocks, math symbols, flower emojis)
Perturbation 3 = Cognitive load with emotional dialogue priming
Perturbation 4 = Minor identity bias (cat dislike)

### Models Tested
- Claude Opus 4 (claude-opus-4-20250514)
- Claude Sonnet 4 (claude-sonnet-4-20250514)  
- Claude 3.7 Sonnet (claude-3-7-sonnet-20250219)
- Hermes-3-Llama-3.1-70b
