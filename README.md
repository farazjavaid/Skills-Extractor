# SynapseScope Skill Extractor

A sophisticated skill extraction system that identifies professional skills from resumes and documents with context-aware matching.

## Overview

The Skill Extractor system consists of two main components:

1. **Skills Database Processor (`skills_simple.py`)**: Prepares a skills database from an Excel file containing standardized skills.
2. **Skill Extractor (`extractorv6.py`)**: Extracts skills from resumes/documents using the prepared database.

The system features:
- Three-trie matching for efficient skill identification
- Enhanced longest-match prioritization algorithm
- Context-aware disambiguation to handle ambiguous terms
- Slash-separated skill handling with false-positive prevention
- HTML and text output formats, including skill IDs

## Requirements

- Python 3.6+
- Required Python packages:
  ```
  pandas
  spacy
  docx
  pygtrie
  ```
- spaCy English model: `python -m spacy download en_core_web_sm`

## Project Files

The project consists of these main files:

1. `skills_simple.py` - Skills database processor script
2. `extractorv6.py` - Main skill extraction engine
3. `skill_patterns.json` - Configuration file for skill patterns and rules
4. `README.md` - This documentation file

## Usage

### Step 1: Process Skills Database

First, create a skills database from an Excel file containing skills:

```bash
python skills_simple.py --input "path/to/skills_file.xlsx"
```

This will:
- Read skills from the Excel file (first column should be Skill ID)
- Process and standardize skills
- Save a database file (`skills.db`) to your Downloads folder
  (Default path: `C:\Users\[username]\Downloads\skills.db`)

### Step 2: Extract Skills from Resumes

Use the prepared database to extract skills from Word documents:

```bash
python extractorv6.py --preprocessed "path/to/skills.db" --resume "path/to/resume.docx" --output "skills_output.html" --text-output --text-file "skill_ids.txt"
```

Default output files:
- HTML output: `C:\Users\[username]\Downloads\skills_output.html`
- Text output: `C:\Users\[username]\Downloads\skill_ids.txt`

Parameters:
- `--preprocessed`: Path to the skills database file created in Step 1
- `--resume`: Path to the resume (Word document)
- `--output`: Output HTML file path (default: skills_output.html)
- `--text-output`: Include text-only output with skill IDs
- `--text-file`: Path for text output file (default: skill_ids.txt)
- `--no-open`: Don't automatically open the results in browser
- `--debug`: Show detailed debug information
- `--config`: Path to custom skill patterns configuration

## Output Formats

### HTML Output
The HTML output includes:
- Split-pane view with highlighted skills in the original text
- Alphabetical list of skills with classifications
- Interactive highlighting between skills list and text
- Summary statistics

### Text Output
When using the `--text-output` option, a text file is created containing:
- One skill ID per line for each extracted skill
- If a skill ID is not available, it's marked with "no_id:" prefix

## Configuration

The system uses a `skill_patterns.json` file to configure skill extraction behavior:
- `no_lemmatize_terms`: Terms that should not be reduced to root form
- `preserve_slash_terms`: Whether to preserve terms containing slashes
- `compound_terms`: Multi-word skills that should be treated as units
- `ambiguous_terms`: Terms that need context disambiguation
- And more specialized patterns

## Troubleshooting

### Common Issues:

1. **No skills found**: 
   - Check that the skills database was created correctly
   - Ensure the Excel format has skills in the expected columns
   - Verify the path to the database file is correct

2. **False positives with slash skills**:
   - The system has special handling for slash skills (like "Gravitational Force / Field")
   - Individual parts should not be matched on their own

3. **Lemmatization issues**:
   - If skills are being incorrectly lemmatized, add them to the "no_lemmatize_terms" list in skill_patterns.json

## Example Workflow

```bash
# Step 1: Create skills database
python skills_simple.py --input "C:\[Users]\Documents\Leadership\Skill Module\Libraries\Skills refined library\Skill_Class\skills_classification_output.xlsx"

# Step 2: Extract skills from a resume
python extractorv6.py --preprocessed "C:\Users\skills.db" --resume "C:\[Users]\Downloads\resume1.docx" --text-output --text-file "C:\[Users]\Downloads\skill_ids.txt"

# View the results in your browser and check the text output file
# Default results locations:
# - HTML: C:\Users\[User]\Downloads\skills_output.html
# - Text: C:\Users\[User]\Downloads\skill_ids.txt
```

<!-- Freeze -->
pip freeze > requirements.txt
