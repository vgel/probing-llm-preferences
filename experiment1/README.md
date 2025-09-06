## Experiment 1

This folder contains data and code from experiment 1 (Agent Think Tank) testing preference consistency between verbal reports and behavioral choices in a virtual environment with economic trade-offs.

### Folder Structure

**phase0_dataset**  
Raw API responses from baseline preference identification (300 calls per model across 3 prompts). Contains semantic analysis results identifying top conversational attractors for each model.

**code**  
Complete "Agent Think Tank" platform in a click-and-run Jupyter notebook (one per model, with pre-loaded letters text):
- Python backend with API management and state tracking
- Web-based real-time interface (HTML/CSS/JavaScript) 
- Custom JSON action parsing and safety mechanisms
- All prompts and letter content with Theme A variants per model

**results**  
Contains 90 timestamped session logs (30 per model) plus qualitative behavioral reports for Opus 4 and Sonnet 4.
