#!/usr/bin/env python
"""
Optimized Skills Extractor with Improved Matching Algorithm

Features:
- Three-trie matching for efficient skill identification
- Enhanced longest-match prioritization algorithm
- Robust context-aware disambiguation system
- Multi-term skill preference
- Memory-efficient processing with chunking
- Comprehensive error handling
"""

import re
import os
import time
import pickle
import logging
import argparse
import unicodedata
import traceback
import json
from pathlib import Path
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Any, Generator, Optional, Union

import pandas as pd
import docx
import spacy
from pygtrie import StringTrie

# ———————— Logging Setup —————————————————————————————————————————
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    force=True
)
log = logging.getLogger(__name__)
DEBUG_MODE = False

# Print a visible marker when the script starts
print("🔍 Skills Extractor v6 — Enhanced Algorithm Edition")

# ———————— Configuration Loading ———————————————————————————————————
def load_skill_patterns(config_path=None):
    """Load skill patterns from configuration file"""
    if config_path is None:
        # Default to skill_patterns.json in the same directory as this script
        config_path = Path(__file__).parent / "skill_patterns.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        log.info(f"Loaded skill patterns from {config_path}")
        return patterns
    except Exception as e:
        log.error(f"Error loading skill patterns: {e}")
        log.debug(traceback.format_exc())
        # Return minimal default patterns
        return {
            "valid_single_letters": ["r", "c", "m"],
            "single_letter_skill_patterns": ["\\b([a-zA-Z])\\s+language\\b"],
            "compound_terms": ["machine learning", "data science"],
            "ambiguous_terms": {"excel": {"pos_weight": 1.0, "software": True}},
            "positive_anchors": ["microsoft", "proficient", "skill"],
            "negative_anchors": ["excel at", "word of"],
            "software_indicators": ["\\bmicrosoft\\s+excel\\b"],
            "irregular_plurals": {"analyses": "analysis"}
        }

# Load skill patterns
SKILL_PATTERNS = load_skill_patterns()

# ———————— NLP Utilities —————————————————————————————————————————
_nlp = None
def get_nlp():
    """Lazy-load spaCy model to avoid memory overhead until needed"""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise ImportError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    return _nlp

@lru_cache(maxsize=16384)
def normalize_text(text):
    """Normalize unicode text with performance caching"""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).strip().lower()

# Load irregular plurals from configuration
IRREGULAR_PLURALS = SKILL_PATTERNS.get("irregular_plurals", {})
log.info(f"Loaded {len(IRREGULAR_PLURALS)} irregular plural forms")

@lru_cache(maxsize=16384)
def simple_normalize(tok):
    """
    Efficiently normalize tokens with optimized logic, preserving:
    1. ALL CAPS words (acronyms/abbreviations)
    2. Special -ing technical terms that should not be normalized
    3. Slash-containing product names and technical standards
    """
    if not tok or len(tok)<3: return tok
    
    # Preserve ALL CAPS words (acronyms/abbreviations)
    if tok.isupper() and len(tok) > 1:
        return tok
        
    # Preserve slash-containing terms (product names, technical standards)
    preserve_slash_terms = SKILL_PATTERNS.get("preserve_slash_terms", True)
    if '/' in tok and preserve_slash_terms:
        return tok.lower()
        
    # Get our list of words that should not be normalized
    no_lemmatize_terms = set(SKILL_PATTERNS.get("no_lemmatize_terms", []))
    
    # Protect special terms like "branding", "programming", "tools", etc.
    if tok.lower() in no_lemmatize_terms:
        return tok.lower()
        
    # Treat hyphens and slashes similarly - keep the whole term together
    if '-' in tok or '/' in tok:
        if '-' in tok:
            parts = tok.split('-')
            return '-'.join(simple_normalize(p) for p in parts)
        else:  # '/' in tok
            # For slash terms, keep the whole term intact
            return tok.lower()
        
    low = tok.lower()
    if low in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[low]
    
    # Check if this might be a product name (capitalized)
    if tok[0].isupper() and tok not in {"The", "A", "An", "In", "Of", "On", "For", "With"}:
        return tok.lower()  # Preserve likely product names
        
    if tok.endswith("ies") and len(tok)>4: return tok[:-3]+"y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok)>3: return tok[:-1]
    
    # Only remove "ing" if not in our protected list
    if tok.endswith("ing") and len(tok)>5 and not tok.lower() in no_lemmatize_terms: 
        return tok[:-3]
        
    if tok.endswith("ed") and len(tok)>4: return tok[:-2]
    return tok

@lru_cache(maxsize=8192)
def lemmatize_phrase(phrase):
    """
    Lemmatize a phrase using spaCy with caching, preserving:
    1. ALL CAPS words (likely abbreviations)
    2. Special -ing technical terms that should not be lemmatized
    3. Terms containing slashes (product names, technical standards)
    4. Proper nouns and product names
    """
    if not isinstance(phrase, str) or not phrase.strip(): return ""
    
    # Get our list of words that should not be lemmatized
    no_lemmatize_terms = set(SKILL_PATTERNS.get("no_lemmatize_terms", []))
    preserve_slash_terms = SKILL_PATTERNS.get("preserve_slash_terms", True)
    
    try:
        # If this is a product name or slash-containing term, preserve it
        if '/' in phrase and preserve_slash_terms:
            log.debug(f"Preserving slash-containing term: '{phrase}'")
            return phrase.lower()  # Keep the whole term intact
            
        # Convert to lowercase for processing but remember original case
        words = phrase.split()
        is_all_caps = [w.isupper() and len(w) > 1 for w in words]  # Track which words are ALL CAPS
        is_protected_term = [w.lower() in no_lemmatize_terms for w in words]  # Track protected terms
        
        # Check for product names (typically FirstWord + generic term)
        has_product_pattern = False
        if len(words) >= 2:
            # Look for patterns like "ProductName Tools" or "CompanyName System"
            if (words[-1].lower() in no_lemmatize_terms and 
                (words[0][0].isupper() or words[0].isupper())):
                has_product_pattern = True
                log.debug(f"Detected product name pattern: '{phrase}'")
        
        # If this looks like a product name, preserve it
        if has_product_pattern:
            return phrase.lower()
            
        # Process with spaCy on lowercase version
        doc = get_nlp()(phrase.lower())
        
        # Create result while preserving special words
        result = []
        for i, token in enumerate(doc):
            # If the original word was ALL CAPS, use it as-is without lemmatizing
            if i < len(is_all_caps) and is_all_caps[i]:
                result.append(words[i])  # Use original ALL CAPS word
            # If word is in our protected list (like "branding"), preserve it
            elif i < len(is_protected_term) and is_protected_term[i]:
                log.debug(f"Preserving protected term: '{words[i]}'")
                result.append(words[i].lower())  # Use original word (but lowercase)
            # If it's a proper noun (likely a product/company name), preserve it
            elif token.pos_ == "PROPN":
                result.append(words[i] if i < len(words) else token.text)
            else:
                result.append(token.lemma_)  # Use lemmatized form
                
        return " ".join(result)
    except Exception as e:
        log.warning(f"Lemmatization error: {e}")
        return phrase.lower()  # Fallback to lowercase if lemmatization fails

# ———————— Build Tries ——————————————————————————————————————————
def build_tries_from_skills(data):
    """Build optimized tries from skills data with better error handling"""
    log.info("Building tries from skills data…")
    t0 = time.time()
    exact = StringTrie(separator=' ')
    std = StringTrie(separator=' ')
    lemma = StringTrie(separator=' ')
    
    try:
        unwanted = {"Not a Skill", "Vague/Unclear Terms", "Error"}
        classes = data.get("classification", {}) if isinstance(data, dict) else {}
        skill_ids = data.get("skill_ids", {}) if isinstance(data, dict) else {}
        
        # Debug info about database structure
        log.info(f"Skills database keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        log.info(f"Exact skills count: {len(data.get('exact_skills', {})) if isinstance(data, dict) else 0}")
        log.info(f"Skill IDs count: {len(skill_ids)}")
        
        # Check a sample of skills
        exact_skills = data.get("exact_skills", {})
        if exact_skills:
            sample_keys = list(exact_skills.keys())[:3]
            for key in sample_keys:
                log.info(f"Sample skill: '{key}' -> {exact_skills[key]}")
        
        # Process exact skills
        for norm, orig in data.get("exact_skills", {}).items():
            try:
                # Check if this skill has an ID stored in the original data
                skill_id = None
                if isinstance(orig, dict) and "id" in orig:
                    skill_id = orig["id"]
                # If no ID in original data, check if it's stored elsewhere in the database
                elif norm in skill_ids:
                    skill_id = skill_ids[norm]
                
                # For debugging
                if not isinstance(orig, dict):
                    log.debug(f"Skill '{norm}' has non-dict value: {type(orig)}")
                
                # Create info dict regardless of input format
                if isinstance(orig, dict):
                    info = orig.copy()  # Use a copy to avoid modifying the original
                    if "classification" not in info:
                        info["classification"] = classes.get(norm, "Unknown")
                else:
                    info = {"name": orig, "classification": classes.get(norm, "Unknown")}
                
                # Add skill ID to the info dictionary if available
                if skill_id and "id" not in info:
                    info["id"] = skill_id
                
                # Debug for classification
                if "classification" not in info:
                    log.warning(f"No classification for '{norm}'")
                
                # Allow skills without classification to be included
                if "classification" not in info or info["classification"] not in unwanted:
                    exact[norm] = info
                    
                    # Also add lowercase variant for case-insensitive matching
                    norm_lower = norm.lower()
                    if norm_lower != norm and norm_lower not in exact:
                        exact[norm_lower] = info
                        
                    # Add normalized form
                    words = norm.split()
                    nm = " ".join(simple_normalize(w) if not w.isupper() or len(w) <= 1 else w for w in words)
                    if nm != norm and nm not in lemma:
                        lemma[nm] = info
                        
                    # Add hyphen variant
                    if '-' in norm:
                        sp = norm.replace('-', ' ')
                        if sp not in exact and sp not in lemma:
                            lemma[sp] = info
                            
                    # Handle slash-separated skills
                    # DON'T add individual parts as separate skills to avoid false matches
                    # For example, "Gravitational Force / Field" should not match just "field"
                    if '/' in norm:
                        # Only add the complete term with slash, but don't add individual parts
                        log.debug(f"Slash skill: '{norm}' - NOT adding individual parts as separate skills")
                        # Optionally add a version with slash replaced by space, but don't decompose further
                        norm_with_space = norm.replace('/', ' ')
                        if norm_with_space != norm and norm_with_space not in exact and norm_with_space not in lemma:
                            lemma[norm_with_space] = info
            except Exception as e:
                log.warning(f"Error processing skill '{norm}': {e}")
                continue
        
        # Process standardized skills
        for norm, orig in data.get("standardized_skills", {}).items():
            try:
                info = orig if isinstance(orig, dict) else {"name": orig, "classification": classes.get(norm, "Unknown")}
                if info["classification"] not in unwanted:
                    std[norm] = info
            except Exception as e:
                log.warning(f"Error processing standardized skill '{norm}': {e}")
                continue
        
        # Process lemmatized skills
        for norm, orig in data.get("lemmatized_skills", {}).items():
            try:
                info = orig if isinstance(orig, dict) else {"name": orig, "classification": classes.get(norm, "Unknown")}
                if info["classification"] not in unwanted:
                    lemma[norm] = info
            except Exception as e:
                log.warning(f"Error processing lemmatized skill '{norm}': {e}")
                continue
                
    except Exception as e:
        log.error(f"Error building tries: {e}")
        log.debug(traceback.format_exc())
        # Provide at least basic trie functionality even with errors
        
    log.info(f"Tries built in {time.time()-t0:.2f}s — exact={len(exact)}, std={len(std)}, lemma={len(lemma)}")
    return exact, std, lemma

def extract_quick_check_words(exact, std, lemma):
    """Extract all words from skills for comprehensive filtering"""
    try:
        qcw = set()
        for trie in (exact, std, lemma):
            for key in trie.keys():
                # Add all words from the skill, not just the first
                for word in key.split():
                    qcw.add(word)
                    if '-' in word: 
                        for part in word.split('-'):
                            qcw.add(part)
        log.info(f"Generated {len(qcw)} quick-check words")
        return qcw
    except Exception as e:
        log.error(f"Error extracting quick check words: {e}")
        log.debug(traceback.format_exc())
        # Return a minimal set of common professional words as fallback
        return {"experience", "skill", "development", "management", "data", "analysis"}

# ———————— Load Preprocessed DB ————————————————————————————————————
def load_preprocessed_skills(db_path):
    """Load preprocessed skills with robust error handling"""
    log.info(f"Loading preprocessed skills database from {db_path}")
    t0 = time.time()
    
    try:
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        
        # Debug info about the loaded data
        log.info(f"Database type: {type(data)}")
        if isinstance(data, dict):
            log.info(f"Database keys: {list(data.keys())}")
            log.info(f"Metadata: {data.get('metadata', {})}")
            
        classes = data.get("classification", {}) if isinstance(data, dict) else {}
        if classes:
            log.info(f"Loaded {len(classes)} classifications")
        else:
            log.warning("No classifications found in database")
            
        # Check if we have skill IDs in the database
        skill_ids = data.get("skill_ids", {}) if isinstance(data, dict) else {}
        if skill_ids:
            log.info(f"Loaded {len(skill_ids)} skill IDs")
        else:
            log.warning("No skill IDs found in database")
            
        # Check structure of exact_skills
        exact_skills = data.get("exact_skills", {})
        if exact_skills:
            log.info(f"Loaded {len(exact_skills)} exact skills")
            
            # Check a sample skill
            if list(exact_skills.keys()):
                sample_key = list(exact_skills.keys())[0]
                sample_value = exact_skills[sample_key]
                log.info(f"Sample skill: '{sample_key}' -> {type(sample_value)}")
                if isinstance(sample_value, dict):
                    log.info(f"Sample skill keys: {list(sample_value.keys())}")
        else:
            log.warning("No exact skills found in database")
            
        exact, std, lemma = build_tries_from_skills(data)
        qcw = extract_quick_check_words(exact, std, lemma)
        fmt = data.get("metadata", {}).get("format", "") if isinstance(data, dict) else ""
        is_std = (fmt == "standardized")
        log.info(f"Using standardized format: {is_std}")
        log.info(f"Preprocessed loaded in {time.time()-t0:.2f}s")
        return exact, std, lemma, is_std, qcw, classes
    
    except FileNotFoundError:
        log.error(f"Skills database not found: {db_path}")
        raise
    except pickle.UnpicklingError:
        log.error(f"Invalid or corrupted skills database: {db_path}")
        raise
    except Exception as e:
        log.error(f"Error loading skills database: {e}")
        log.debug(traceback.format_exc())
        raise

# ———————— Fallback Excel Loader ———————————————————————————————————
def load_skills_from_excel(xl):
    """Load skills from Excel with comprehensive error handling"""
    log.info(f"Loading skills from Excel: {xl}")
    
    try:
        df = pd.read_excel(xl, engine="openpyxl")
        exact_s, std_s, lem_s = {}, {}, {}
        
        if 'Skill' in df.columns:
            for s in df['Skill'].dropna():
                exact_s[normalize_text(s)] = s
                
            if 'Standardized_Skill' in df.columns:
                for _, r in df.iterrows():
                    if pd.notna(r['Standardized_Skill']):
                        std_s[normalize_text(r['Standardized_Skill'])] = r['Skill']
                        
            for c in df.columns:
                if c.lower() in ("lemmatized skill", "lemma"):
                    for _, r in df.iterrows():
                        if pd.notna(r[c]):
                            lem_s[normalize_text(r[c])] = r['Skill']
                    break
        else:
            for s in df.iloc[:, 0].dropna():
                exact_s[normalize_text(s)] = s
                
        exact, std, lemma = build_tries_from_skills({
            "exact_skills": exact_s,
            "standardized_skills": std_s,
            "lemmatized_skills": lem_s,
            "classification": {}
        })
        
        return exact, std, lemma, bool(std_s), extract_quick_check_words(exact, std, lemma), {}
        
    except pd.errors.EmptyDataError:
        log.error(f"Excel file is empty: {xl}")
        raise
    except pd.errors.ParserError:
        log.error(f"Excel parsing error - corrupted file: {xl}")
        raise
    except Exception as e:
        log.error(f"Error loading Excel file: {e}")
        log.debug(traceback.format_exc())
        raise

# ———————— Text & Tokenization ———————————————————————————————————
def extract_text_from_docx(path):
    """Extract text from Word document with robust error handling and chunking for large files"""
    log.info(f"Extracting text from {path}")
    
    try:
        t0 = time.time()
        doc = docx.Document(path)
        
        # Use a generator to process paragraphs in chunks for memory efficiency
        def paragraph_generator():
            for para in doc.paragraphs:
                if para.text.strip():  # Skip empty paragraphs
                    yield para.text
                    
        # Process paragraphs in chunks to handle very large documents
        chunk_size = 100  # Process 100 paragraphs at a time
        chunks = []
        current_chunk = []
        para_count = 0
        
        for para in paragraph_generator():
            current_chunk.append(para)
            para_count += 1
            
            if len(current_chunk) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                log.debug(f"Processed {para_count} paragraphs...")
                
        # Add the last chunk if any paragraphs remain
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        txt = " ".join(chunks)
        log.info(f"Extracted {len(txt)} characters in {time.time()-t0:.2f}s from {para_count} paragraphs")
        return txt
        
    except docx.opc.exceptions.PackageNotFoundError:
        log.error(f"Invalid or corrupted Word document: {path}")
        raise
    except docx.opc.exceptions.OpcError:
        log.error(f"Word document structure error: {path}")
        raise
    except PermissionError:
        log.error(f"Permission denied to read file: {path}")
        raise
    except Exception as e:
        log.error(f"Error extracting text from Word document: {e}")
        log.debug(traceback.format_exc())
        raise

def tokenize_with_indices(txt):
    """Tokenize text with optimized pattern matching for better performance"""
    try:
        t0 = time.time()
        # Optimized regex pattern - compiled once for better performance
        # Modified to better handle contractions and special programming language syntax
        pattern = re.compile(r"[A-Za-z0-9]+(?:[.\-!#''+/][A-Za-z0-9]+)*|[,\.!]")
        
        # Common contractions to keep together
        contractions = [
            r"I'm", r"I'll", r"I'd", r"I've", 
            r"you're", r"you'll", r"you'd", r"you've",
            r"he's", r"he'll", r"he'd", r"he's",
            r"she's", r"she'll", r"she'd", r"she's",
            r"we're", r"we'll", r"we'd", r"we've",
            r"they're", r"they'll", r"they'd", r"they've",
            r"that's", r"that'll", r"that'd",
            r"who's", r"who'll", r"who'd",
            r"what's", r"what'll", r"what'd",
            r"where's", r"where'll", r"where'd",
            r"when's", r"when'll", r"when'd",
            r"why's", r"why'll", r"why'd",
            r"how's", r"how'll", r"how'd",
            r"isn't", r"aren't", r"wasn't", r"weren't",
            r"hasn't", r"haven't", r"hadn't",
            r"doesn't", r"don't", r"didn't",
            r"won't", r"wouldn't", r"can't", r"couldn't",
            r"shouldn't", r"mightn't", r"mustn't"
        ]
        
        # Create a pattern that preserves contractions
        contraction_pattern = re.compile(
            '|'.join(rf'\b{c}\b' for c in contractions), 
            re.IGNORECASE
        )
        
        # Pattern to find slash-separated skills
        # This will help us split things like "Java/Python/JavaScript" into separate tokens
        slash_pattern = re.compile(r'([A-Za-z0-9][A-Za-z0-9\-_\+\#]*)/([A-Za-z0-9][A-Za-z0-9\-_\+\#]*)')
        
        # Process in batches for large documents
        batch_size = 100000  # Process 100K characters at a time
        tokens = []
        
        for i in range(0, len(txt), batch_size):
            batch = txt[i:i+batch_size]
            
            # First identify contractions and temporarily replace them
            contraction_positions = []
            for m in contraction_pattern.finditer(batch):
                contraction_positions.append((
                    m.group(0),
                    i+m.start(),
                    i+m.end()
                ))
            
            # Then do normal tokenization
            batch_tokens = [(m.group(0), i+m.start(), i+m.end()) 
                           for m in pattern.finditer(batch)]
            
            # Add tokens while filtering out single-letter tokens from contractions
            for token, start, end in batch_tokens:
                # Skip single letters that are part of contractions
                is_part_of_contraction = False
                if len(token) == 1:
                    for cont, cont_start, cont_end in contraction_positions:
                        if cont_start <= start < cont_end:
                            is_part_of_contraction = True
                            break
                
                if not is_part_of_contraction:
                    # If this token contains slashes, also add the parts as individual tokens
                    # This helps detect skills listed as "Java/Python/JavaScript"
                    if '/' in token and len(token) > 3:
                        # Check if it looks like a slash-separated skill list
                        # Split token at slashes and add each part as a separate token
                        parts = token.split('/')
                        if len(parts) >= 2 and all(len(p.strip()) > 1 for p in parts):
                            # First add the full token with slashes (for exact matching)
                            tokens.append((token, start, end))
                            
                            # Then add each part as a separate token
                            log.debug(f"Splitting slash-separated token: '{token}' into {parts}")
                            current_pos = start
                            for part in parts:
                                part = part.strip()
                                if part and len(part) > 1:  # Skip empty parts and single letters
                                    # Calculate approximate position for the part
                                    part_start = current_pos
                                    part_end = part_start + len(part)
                                    tokens.append((part, part_start, part_end))
                                    log.debug(f"Added token part: '{part}' at {part_start}-{part_end}")
                                    current_pos = part_end + 1  # +1 for the slash
                        else:
                            # Not a slash-separated skill list, add as regular token
                            tokens.append((token, start, end))
                    else:
                        # Regular token without slashes
                        tokens.append((token, start, end))
            
            if i > 0 and i % (batch_size * 10) == 0:
                log.debug(f"Tokenized {i:,} of {len(txt):,} characters...")
        
        log.info(f"Tokenized into {len(tokens)} tokens in {time.time()-t0:.2f}s")
        
        # Special handling for compound terms - explicitly look for known compound terms
        # This acts as a backup to ensure we don't miss important multi-word phrases
        compound_terms = SKILL_PATTERNS.get("compound_terms", [])
        log.debug(f"Using {len(compound_terms)} compound terms for direct matching")
        
        # Look for skills in special contexts like parentheses, forward slashes, etc.
        for term in compound_terms:
            # Make term case-insensitive for better matching
            # Look for variants with forward slashes, parentheses, etc.
            pattern_variants = [
                rf'\b{re.escape(term)}\b',  # Standard match
                rf'\({re.escape(term)}\)',  # In parentheses
                rf'/{re.escape(term)}',     # After slash
                rf'{re.escape(term)}/',     # Before slash
                rf'\b{re.escape(term)}s\b'  # Plural form
            ]
            
            for pattern in pattern_variants:
                try:
                    for match in re.finditer(pattern, txt, re.IGNORECASE):
                        start, end = match.span()
                        tokens.append((term, start, end))
                        log.debug(f"Added compound term with special handling: '{term}' at {start}-{end}")
                except Exception as e:
                    log.warning(f"Error processing pattern for '{term}': {e}")
        
        # Special handling for programming languages with special characters
        # Look for programming languages in common list contexts
        prog_lang_patterns = SKILL_PATTERNS.get("programming_language_patterns", [])
        if prog_lang_patterns:
            # Look for programming language lists and contexts
            for pattern in prog_lang_patterns:
                try:
                    for match in re.finditer(pattern, txt, re.IGNORECASE):
                        match_text = match.group(0)
                        start, end = match.span()
                        # Only add if we found a programming language marker
                        if len(match_text) > 1:
                            # Add special tokens for the programming language markers
                            tokens.append((match_text, start, end))
                            log.debug(f"Added programming context marker: '{match_text}' at {start}-{end}")
                except Exception as e:
                    log.warning(f"Error processing programming pattern {pattern}: {e}")
                    
        # Special handling for C, C#, C++ which might be missed
        prog_specials = ["C", "C#", "C\\+\\+"]
        for lang in prog_specials:
            try:
                stripped_lang = lang.replace("\\", "")  # Clean up for logging
                pattern = re.compile(f"\\b{lang}\\b", re.IGNORECASE)
                for match in pattern.finditer(txt):
                    start, end = match.span()
                    # Add special token for the programming language
                    tokens.append((stripped_lang, start, end))
                    log.debug(f"Added special programming token: '{stripped_lang}' at {start}-{end}")
            except Exception as e:
                log.warning(f"Error processing special language {lang}: {e}")
        
        for term in compound_terms:
            term_lower = term.lower()
            # Find all occurrences of the exact compound term
            for match in re.finditer(r'\b' + re.escape(term_lower) + r'\b', txt.lower()):
                start, end = match.span()
                # Add as a single token to ensure direct matching
                tokens.append((term_lower, start, end))
                log.debug(f"Added compound term token: {term_lower} at {start}-{end}")
        
        # Sort tokens by position
        tokens.sort(key=lambda x: x[1])
        
        # Make sure to keep important single-letter programming languages
        valid_single_letters = set(SKILL_PATTERNS.get("valid_single_letters", []))
        # Add common articles and pronouns
        valid_single_letters.update(['a', 'i', 'c'])
        
        # Add any single-letter skills from our database
        database_letters = set()
        
        # Filter tokens but preserve important single letters
        tokens = [(tok, start, end) for tok, start, end in tokens if 
                 not (len(tok) == 1 and tok.isalpha() and 
                      tok.lower() not in valid_single_letters and
                      tok.lower() not in database_letters)]
        
        # Special additions for known single-letter skills
        # Look for specific patterns like "R language" or "experience in M" and add explicit tokens
        single_letter_patterns = [
            (r'\b([A-Z])\s+language\b', r'\1'),  # "R language" -> add token "R"
            (r'\bexperience\s+(?:with|in)\s+([A-Z])\b', r'\1'),  # "experience in R" -> add token "R"
            (r'\bknowledge\s+of\s+([A-Z])\b', r'\1'),  # "knowledge of M" -> add token "M"
            (r'\busing\s+([A-Z])\b', r'\1'),  # "using R" -> add token "R"
            (r'\bin\s+([A-Z])\b', r'\1')  # "in R" -> add token "R"
        ]
        
        # Look for these patterns and add explicit tokens for the single letters
        text_lower = txt.lower()
        for pattern, group in single_letter_patterns:
            for match in re.finditer(pattern, txt, re.IGNORECASE):
                letter = match.group(1)
                if letter.lower() in valid_single_letters:
                    start, end = match.span(1)  # Get position of just the letter group
                    tokens.append((letter, start, end))
                    log.debug(f"Added explicit token for single letter skill: '{letter}' at {start}-{end}")
        
        # Resort tokens by position after adding these special tokens
        tokens.sort(key=lambda x: x[1])
        
        return tokens
    except Exception as e:
        log.error(f"Error during tokenization: {e}")
        log.debug(traceback.format_exc())
        # Return a minimal token list as fallback
        return [(w.lower(), i*10, i*10+len(w)) for i, w in enumerate(txt.split()[:1000])]

def get_contiguous_segments(tokens):
    """Group tokens into segments with optimized processing for large token lists"""
    try:
        t0 = time.time()
        delims = {',', '.', '!', '?', ';', ':', 'and', 'or', 'using', 'with', 'through', 'via'}
        segs, cur = [], []
        
        # Process tokens in batches for memory efficiency with very large documents
        batch_size = 5000  # Process 5000 tokens at a time
        for batch_start in range(0, len(tokens), batch_size):
            batch_end = min(batch_start + batch_size, len(tokens))
            batch = tokens[batch_start:batch_end]
            
            for i, (tok, s, e) in enumerate(batch):
                # Add token to current segment even if it's a delimiter
                # This ensures phrases like "clinical psychology" can be matched even across soft delimiters
                cur.append((tok, s, e))
                
                # Only create a new segment on hard delimiters
                if tok in {'.', '!', '?', ';', ':'}:
                    if cur:
                        segs.append(cur)
                        cur = []
                    
                # Emit segment at batch boundaries to avoid memory issues
                if i == len(batch) - 1 and batch_end < len(tokens):
                    next_token = tokens[batch_end][0] if batch_end < len(tokens) else None
                    if next_token in {'.', '!', '?', ';', ':'} and cur:
                        segs.append(cur)
                        cur = []
        
        # Add the last segment if non-empty
        if cur:
            segs.append(cur)
            
        # Create overlapping windows for better phrase matching
        windowed_segs = []
        window_size = 10  # Size of sliding window
        
        for seg in segs:
            if len(seg) <= window_size:
                windowed_segs.append(seg)
            else:
                # Create overlapping windows
                for i in range(0, len(seg) - window_size + 1, 5):  # Stride of 5 tokens
                    window = seg[i:i+window_size]
                    windowed_segs.append(window)
                
                # Make sure we don't miss the end of the segment
                if (len(seg) - window_size) % 5 != 0:
                    windowed_segs.append(seg[-window_size:])
        
        log.info(f"Created {len(windowed_segs)} segments in {time.time()-t0:.2f}s")
        return windowed_segs
    except Exception as e:
        log.error(f"Error during segmentation: {e}")
        log.debug(traceback.format_exc())
        # Return single-token segments as fallback
        return [[tok] for tok in tokens[:1000]]

# ———————— Context-Aware Disambiguation System —————————————————————
class ContextDisambiguator:
    """
    Context-aware system to disambiguate terms with multiple meanings
    (e.g., "Excel" as a verb vs. as Microsoft Excel)
    """
    def __init__(self):
        # Load configuration from skill patterns
        # Terms that might have ambiguous meanings
        self.ambiguous_terms = SKILL_PATTERNS.get("ambiguous_terms", {})
        
        # Positive context indicators (suggest technical skill)
        self.positive_anchors = SKILL_PATTERNS.get("positive_anchors", [])
        
        # Negative context indicators (suggest non-technical usage)
        self.negative_anchors = SKILL_PATTERNS.get("negative_anchors", [])
        
        # Special verb patterns with stronger weight - these are almost certainly NOT technical skills
        verb_patterns_config = SKILL_PATTERNS.get("verb_patterns", {})
        self.verb_patterns = {}
        for key, patterns in verb_patterns_config.items():
            self.verb_patterns[key] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        # Compile regex patterns for better performance
        self.pos_patterns = [re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE) for a in self.positive_anchors]
        self.neg_patterns = [re.compile(r'\b' + re.escape(a) + r'\b', re.IGNORECASE) for a in self.negative_anchors]
        
        # Specific software product indicators (very strong positive signals)
        self.software_indicators = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in SKILL_PATTERNS.get("software_indicators", [])
        ]
        
        # Special patterns for specific terms
        self.word_negative_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in SKILL_PATTERNS.get("word_negative_patterns", [])
        ]
        
        self.excel_negative_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in SKILL_PATTERNS.get("excel_negative_patterns", [])
        ]
        
        log.info(f"Initialized context disambiguator with {len(self.ambiguous_terms)} ambiguous terms")
        
    def is_technical_skill(self, text: str, start: int, end: int, skill: str) -> bool:
        """
        Determine if a potentially ambiguous term is a technical skill based on context
        
        Args:
            text: The full document text
            start: Start position of the match
            end: End position of the match
            skill: The skill name
        
        Returns:
            bool: True if the term is likely a technical skill, False otherwise
        """
        # Check for ALL CAPS terms - these are likely acronyms/abbreviations and should be treated as technical
        if any(token.isupper() and len(token) > 1 for token in skill.split()):
            # ALL CAPS words are likely technical terms/acronyms, treat as technical skills
            log.debug(f"'{skill}' contains ALL CAPS terms, treating as technical")
            return True
            
        # Skip checking if not an ambiguous term
        skill_lower = skill.lower()
        ambig_keys = [key for key in self.ambiguous_terms.keys() if key in skill_lower.split()]
        
        if not ambig_keys:
            return True
        
        # Get context for disambiguation
        context_window = 100  # Look at 100 chars before and after
        context_before = text[max(0, start - context_window):start].lower()
        context_after = text[end:min(len(text), end + context_window)].lower()
        full_context = f"{context_before} {skill_lower} {context_after}"
        
        # Debug: Show what we're analyzing
        log.debug(f"Disambiguating '{skill}'. Context: '{context_before} [SKILL] {context_after}'")
        
        # Extra debug for ambiguous Excel matches - higher verbosity
        if skill_lower == "excel" or "excel" in skill_lower.split():
            log.debug(f"EXCEL MATCH DETECTED - Investigating context for '{skill}'")
            log.debug(f"Context before: '{context_before}'")
            log.debug(f"Context after: '{context_after}'")
            log.debug(f"Full context: '{full_context}'")
            # Check for specific phrases
            verb_phrases = ["excel in", "excel at", "i excel", "we excel", "to excel"]
            for phrase in verb_phrases:
                log.debug(f"Checking for '{phrase}': {'FOUND' if phrase in full_context.lower() else 'not found'}")
            
        # Microsoft Office combo detection (e.g., "Excel and Word") - highly likely technical
        if len(set(["excel", "word", "powerpoint"]).intersection(skill_lower.split())) >= 2:
            log.debug(f"'{skill}' contains multiple Office products, treating as technical")
            return True
            
        if "microsoft" in skill_lower:
            log.debug(f"'{skill}' contains 'microsoft', treating as technical")
            return True
        
        # First check specific software product indicators (strongest signal)
        for pattern in self.software_indicators:
            if pattern.search(full_context):
                log.debug(f"Found software indicator in context: '{pattern.pattern}'")
                return True
        
        # Now check special verb patterns - these are high-confidence non-technical usages
        for key in ambig_keys:
            if key in self.verb_patterns:
                for pattern in self.verb_patterns[key]:
                    if pattern.search(full_context):
                        log.debug(f"Found verb pattern for '{key}': '{pattern.pattern}' - NOT a technical skill")
                        return False
        
        # Calculate context score
        score = 0.0
        
        # Check for positive anchors (higher score)
        for pattern in self.pos_patterns:
            if pattern.search(full_context):
                # Get the ambiguous term weight 
                term_weight = 1.0  # Default weight
                for key in ambig_keys:
                    term_weight = max(term_weight, self.ambiguous_terms[key]["pos_weight"])
                score += 1.0 * term_weight
                log.debug(f"Found positive pattern: '{pattern.pattern}', score +{1.0 * term_weight}")
        
        # Check for negative anchors (lower score)
        for pattern in self.neg_patterns:
            if pattern.search(full_context):
                score -= 2.0  # Negative patterns are stronger signals
                log.debug(f"Found negative pattern: '{pattern.pattern}', score -2.0")
        
        # Check for combined terms (e.g., "proficient in Excel and Word")
        office_combos = [
            "excel and word", "word and excel", 
            "powerpoint and excel", "excel and powerpoint",
            "proficient in excel", "experience with excel",
            "knowledge of word", "skilled in word",
            "microsoft office"
        ]
        
        for combo in office_combos:
            if combo in full_context:
                score += 3.0  # Strong positive signal
                log.debug(f"Found office combo: '{combo}', score +3.0")
        
        # Additional context patterns for specific ambiguous terms
        word_negative_patterns = [
            r'\bmy word\b', r'\byour word\b', r'\bhis word\b', r'\bher word\b', r'\btheir word\b',
            r'\bword of\b', r'\bwords of\b', r'\bgive.*word\b', r'\btake.*word\b', r'\bkeep.*word\b',
            r'\bby word\b', r'\bthe word\b'
        ]
        
        excel_negative_patterns = [
            r'\bi excel\b', r'\bwe excel\b', r'\bthey excel\b', r'\bto excel\b',
            r'\bhe excels\b', r'\bshe excels\b', r'\bwill excel\b', r'\bcan excel\b'
        ]
        
        # Add super-strict direct checks for most common verb forms - these have highest priority
        if "excel" in skill_lower:
            # Check for the exact verb phrases that cause the most false positives
            direct_excel_checks = [
                "excel in", "excel at", "excel with", "excels in", "excels at", 
                "i excel", "we excel", "they excel", "to excel"
            ]
            
            # Need a more robust check that handles extra whitespace
            # First normalize the context by replacing multiple spaces with a single space
            normalized_context = re.sub(r'\s+', ' ', full_context.lower())
            
            log.debug(f"Normalized context: '{normalized_context}'")
            
            for phrase in direct_excel_checks:
                if phrase in normalized_context:
                    log.debug(f"Direct string match for verb phrase '{phrase}' in normalized context - NOT a technical skill")
                    return False
                    
            # Also check for "i" followed by "excel" followed by "in" or "at" with any spacing
            i_excel_pattern = re.compile(r'\bi\s+excel\s+(in|at|with)', re.IGNORECASE)
            if i_excel_pattern.search(full_context):
                log.debug(f"Regex match for 'I excel in/at/with' pattern - NOT a technical skill")
                return False
        
        # Check for additional negative patterns based on the specific term
        for key in ambig_keys:
            if key == "word" and any(re.search(pattern, full_context, re.IGNORECASE) for pattern in word_negative_patterns):
                log.debug(f"Found specific negative pattern for 'word' - NOT a technical skill")
                return False
                
            if key == "excel" and any(re.search(pattern, full_context, re.IGNORECASE) for pattern in excel_negative_patterns):
                log.debug(f"Found specific negative pattern for 'excel' - NOT a technical skill")
                return False
        
        # Special handling for common cases
        for key in ambig_keys:
            # Software products are assumed technical ONLY if there's at least one positive signal
            if self.ambiguous_terms[key]["software"] and score > 0:  # Changed from >= 0 to > 0
                score += 0.5
                log.debug(f"Software product boost for '{key}', score +0.5")
                
        # Special handling for Excel specifically - check for telltale verb usages
        if "excel" in ambig_keys and "excel" in skill_lower.split():
            # Direct check for common verbal usage patterns
            excel_verb_phrases = ["excel at", "excel in", "excel with", "i excel", "we excel", "excels in"]
            for phrase in excel_verb_phrases:
                if phrase in full_context:
                    log.debug(f"Found Excel verb usage: '{phrase}', forcing negative result")
                    return False
                    
        # Return decision based on final score
        log.debug(f"Final context score for '{skill}': {score:+.1f} - {'IS' if score >= 0 else 'NOT'} a technical skill")
        return score >= 0

# ———————— Enhanced Trie Matching System —————————————————————————
class TrieMatcher:
    """
    Enhanced matching system that prioritizes longer multi-word matches
    and handles plurals/variations elegantly
    """
    def __init__(self, exact, std, lemma, is_std, qcw, classes, full_text=None):
        self.exact = exact
        self.std = std  
        self.lemma = lemma
        self.is_std = is_std
        self.qcw = qcw
        self.classes = classes
        self.full_text = full_text
        
        # Define valid single-letter programming languages from configuration
        self.valid_single_letters = set(SKILL_PATTERNS.get("valid_single_letters", []))
        
        # Patterns that suggest a single letter is a legitimate skill
        self.single_letter_skill_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SKILL_PATTERNS.get("single_letter_skill_patterns", [])
        ]
        
        # Common programming contexts for validating single-letter matches
        self.programming_contexts = [
            "language", "programming", "code", "coding", "development",
            "experience with", "experience in", "knowledge of", "proficiency in",
            "skilled in", "software", "framework", "library"
        ]
    
    def has_programming_language_context(self, token_pos, lookahead=None):
        """
        Check if a token at a given position has context suggesting it's a programming language
        
        Args:
            token_pos: Position of the token in the segment
            lookahead: Number of tokens looked ahead (for multi-token matching)
            
        Returns:
            bool: True if context suggests this is a programming language token
        """
        # If we don't have segment info, be conservative
        if not hasattr(self, 'current_segment'):
            return False
            
        segment = self.current_segment
        if not segment or token_pos >= len(segment):
            return False
            
        # Get the surrounding tokens (up to 3 on each side)
        start_idx = max(0, token_pos - 3)
        end_idx = min(len(segment), token_pos + 3 + (lookahead or 0))
        
        surrounding = " ".join(t[0].lower() for t in segment[start_idx:end_idx])
        
        # Check for programming language context
        for context in self.programming_contexts:
            if context in surrounding:
                return True
                
        # Check for patterns that suggest programming language
        for pattern in self.single_letter_skill_patterns:
            if pattern.search(surrounding):
                return True
                
        return False
        
    def has_product_name_context(self, segment, token_pos, lookahead=None):
        """
        Check if a generic term (like "tools") has context suggesting it's part of a product name
        
        Args:
            segment: The current segment
            token_pos: Position of the token in the segment
            lookahead: Number of tokens looked ahead (for multi-token matching)
            
        Returns:
            bool: True if context suggests this is a product name
        """
        if not segment or token_pos >= len(segment):
            return False
            
        # Get the token that might be a generic term
        token = segment[token_pos][0].lower() if token_pos < len(segment) else ""
        
        # If this doesn't look like a generic term, no need for validation
        no_lemmatize_terms = set(SKILL_PATTERNS.get("no_lemmatize_terms", []))
        if token not in no_lemmatize_terms:
            return True
            
        # Get surrounding context (5 tokens before, 2 tokens after)
        start_idx = max(0, token_pos - 5)
        end_idx = min(len(segment), token_pos + 2 + (lookahead or 0))
        
        # Get the tokens before this one
        preceding_tokens = [t[0] for t in segment[start_idx:token_pos]]
        
        # Look for capitalized words that might be product/company names
        for prev_token in preceding_tokens:
            # If there's a capitalized word before this generic term, it might be a product
            if prev_token and prev_token[0].isupper():
                log.debug(f"Found capitalized word '{prev_token}' before generic '{token}'")
                return True
                
        # Check for slash-containing terms nearby
        for i in range(start_idx, end_idx):
            if i < len(segment) and '/' in segment[i][0]:
                log.debug(f"Found slash in nearby token '{segment[i][0]}'")
                return True
                
        # Check surrounding text for specific product indicators
        surrounding = " ".join(t[0].lower() for t in segment[start_idx:end_idx])
        product_indicators = [
            "version", "platform", "system", "software", "application",
            "developed by", "created by", "vendor", "product", "technology",
            "interface", "sdk", "api", "framework"
        ]
        
        for indicator in product_indicators:
            if indicator in surrounding:
                log.debug(f"Found product indicator '{indicator}' near generic '{token}'")
                return True
                
        # No product context found - this is likely a generic usage
        return False
        
    def validate_single_letter_skill(self, token, segment_pos=None, token_start=None, token_end=None):
        """
        Comprehensive validation for single-letter skills to avoid false positives
        
        Args:
            token: The single-letter token
            segment_pos: Position in current segment
            token_start: Start position in text
            token_end: End position in text
            
        Returns:
            bool: True if this is likely a genuine skill reference
        """
        # Skip validation for non-single-letter tokens
        if len(token.strip()) != 1:
            return True
            
        # Skip validation for non-alphabetic tokens
        if not token.isalpha():
            return True
        
        # Special handling for C programming language
        if token.strip() == "C":
            # Check for programming language context
            if self.full_text and token_start is not None:
                # Look for programming language context within 100 characters
                context_start = max(0, token_start - 100)
                context_end = min(len(self.full_text), token_end + 100)
                context = self.full_text[context_start:context_end].lower()
                
                programming_indicators = [
                    "programming", "language", "code", "coding", "development",
                    "proficiency", "software", "leetcode", "hackerrank"
                ]
                
                if any(indicator in context for indicator in programming_indicators):
                    log.debug(f"'C' validated as programming language based on context")
                    return True
                    
                # Check for programming language list patterns
                prog_list_patterns = [
                    r"\b(javascript|python|java|typescript).{0,30}c\b",
                    r"\bc.{0,30}(python|java|javascript|typescript)\b",
                    r"\bc\s*(?:,|and|or|\(|\)|/)\s*c\+\+\b",
                    r"\bc#\s*(?:,|and|or|\(|\)|/)\s*c\b"
                ]
                
                for pattern in prog_list_patterns:
                    if re.search(pattern, context, re.IGNORECASE):
                        log.debug(f"'C' validated as programming language in list")
                        return True
            
        # First check: Is it even in our list of known programming languages?
        if token.lower() not in self.valid_single_letters:
            log.debug(f"Single letter '{token}' not in known programming languages")
            return False
            
        # Check for contraction markers
        is_from_contraction = False
        if hasattr(self, 'full_text') and token_start is not None:
            # Look for apostrophe right before this token
            if token_start > 0 and self.full_text[token_start-1:token_start] == "'":
                is_from_contraction = True
                log.debug(f"Single letter '{token}' appears to be from contraction")
            
            # Look for common contractions: I'm, I'll, you're, etc.
            if token_start > 2:
                prev_chars = self.full_text[max(0, token_start-4):token_start].lower()
                common_contraction_prefixes = ["i'", "i'", "you'", "he'", "she'", "we'", "they'"]
                if any(prefix in prev_chars for prefix in common_contraction_prefixes):
                    is_from_contraction = True
                    log.debug(f"Single letter '{token}' follows contraction prefix")
        
        if is_from_contraction:
            return False
            
        # Check for programming language context patterns
        if self.current_segment and segment_pos is not None:
            # Get surrounding tokens in segment
            start_idx = max(0, segment_pos - 3)
            end_idx = min(len(self.current_segment), segment_pos + 4)
            surrounding = " ".join(t[0].lower() for t in self.current_segment[start_idx:end_idx])
            
            # Check for programming language keywords
            language_markers = [
                "language", "programming", "code", "develop", "script",
                "experience with", "knowledge of", "proficient in"
            ]
            
            if any(marker in surrounding for marker in language_markers):
                log.debug(f"Single letter '{token}' has programming context: '{surrounding}'")
                return True
                
        # If we have full text, do a wider context check
        if hasattr(self, 'full_text') and token_start is not None and token_end is not None:
            # Get a wider context window
            context_start = max(0, token_start - 50)
            context_end = min(len(self.full_text), token_end + 50)
            wider_context = self.full_text[context_start:context_end].lower()
            
            # Check for programming language context
            language_phrases = [
                f"{token.lower()} language",
                f"{token.lower()} programming",
                f"experience with {token.lower()}",
                f"experience in {token.lower()}",
                f"knowledge of {token.lower()}",
                f"code in {token.lower()}"
            ]
            
            if any(phrase in wider_context for phrase in language_phrases):
                log.debug(f"Single letter '{token}' has programming context in wider window")
                return True
        
        # Default to rejecting single-letter tokens without clear context
        log.debug(f"Single letter '{token}' lacks sufficient programming context")
        return False
    
    def _get_surrounding_context(self, token, start_pos, end_pos, segment_str=None, position_in_segment=None):
        """Get surrounding context for disambiguation"""
        # If we have segment information, use it to build better context
        if segment_str and position_in_segment is not None:
            return segment_str
            
        # This is a helper method that tries to get context for a token
        # from nearby tokens in the segment
        context_window = 5  # Look at 5 tokens before and after
        
        # We need access to the full text, but we can't modify the method signature
        # Instead, we'll use patterns to look for common technical skill contexts
        
        # Check if it looks like a language name pattern (e.g., "M language")
        for pattern in self.single_letter_skill_patterns:
            match = pattern.search(f"{token} language")
            if match:
                return f"{token} language"
                
            match = pattern.search(f"knowledge of {token}")
            if match:
                return f"knowledge of {token}"
                
            match = pattern.search(f"experience with {token}")
            if match:
                return f"experience with {token}"
                
            match = pattern.search(f"{token} programming")
            if match:
                return f"{token} programming"
                
        # Check a variety of common programming language contexts
        potential_contexts = [
            f"{token} language",
            f"knowledge of {token}",
            f"experience with {token}",
            f"experience in {token}",
            f"{token} programming",
            f"programming in {token}",
            f"using {token}",
            f"code in {token}",
            f"{token} code",
            f"proficient in {token}",
            f"skilled in {token}",
            f"{token} skills"
        ]
        
        # Just return all potential contexts combined to increase chances of pattern matching
        return " ".join(potential_contexts)
        
    def _is_valid_single_letter_skill(self, token, context):
        """Check if a single letter token is a valid skill based on context"""
        # If it's a single-letter programming language that's in our valid list,
        # consider it valid regardless of context
        if token.lower() in self.valid_single_letters:
            log.debug(f"Accepting '{token}' as a valid single-letter skill (in whitelist)")
            return True
            
        # If it's a single letter and exists in our skills database, consider it valid
        if len(token) == 1 and (token.lower() in self.exact or token.lower() in self.std or token.lower() in self.lemma):
            log.debug(f"Accepting '{token}' as it exists in the skills database")
            return True
            
        # Check if the single letter is part of a known programming language pattern
        for pattern in self.single_letter_skill_patterns:
            if pattern.search(context):
                log.debug(f"Found valid pattern for '{token}': {pattern.pattern}")
                return True
                
        # Look for common patterns in the surrounding context
        normalized_context = re.sub(r'\s+', ' ', context.lower())
        common_patterns = [
            f"{token.lower()} language", 
            f"{token.lower()} programming", 
            f"experience with {token.lower()}", 
            f"knowledge of {token.lower()}",
            f"in {token.lower()}",
            f"using {token.lower()}"
        ]
        
        for pattern in common_patterns:
            if pattern in normalized_context:
                log.debug(f"Found context pattern '{pattern}' for '{token}' - accepting as skill")
                return True
        
        # Default to rejecting single-letter tokens
        return False
        
    def _has_viable_prefix(self, trie, phrase):
        """Check if trie has the phrase or a viable prefix, with special handling for short tokens and slash parts"""
        try:
            # Build a comprehensive list of words that should ONLY match exactly
            # (including common parts of slash-separated skills)
            phrase_lower = phrase.strip().lower()
            
            # Block false positives from very short tokens (a, is, to, etc.)
            if len(phrase_lower) <= 2:  # Single letters or two-letter words
                # Only return true if there's an EXACT match, not just a prefix
                return phrase in trie and not trie.has_subtrie(phrase)
                
            # Block common stop words that might be in slashed skill names
            # Expanded to include more common parts of slash-separated terms
            stop_words = {"is", "was", "be", "in", "on", "at", "to", "or", "and", "but", "of", "for", 
                          "tools", "a", "b", "i", "we", "us", "as", "it", "by", "an", "up", "if"}
            if phrase_lower in stop_words:
                # Only return true if there's an EXACT match, not just a prefix
                return phrase in trie and not trie.has_subtrie(phrase)
                
            # For slightly longer but still common words, be more careful with prefix matching
            common_words = {"the", "this", "that", "with", "from", "they", "she", "his", "her", "our", "your", 
                           "has", "fast", "smp", "test", "part", "system", "service", "app", "use", "data"}
            if phrase_lower in common_words:
                # Only return true if there's an EXACT match, not just a prefix
                return phrase in trie and not trie.has_subtrie(phrase)
            
            # Check if this might be part of a slash-separated skill in our database
            # First collect all slash-separated skills in the trie
            if hasattr(self, 'slash_parts_cache'):
                slash_parts = self.slash_parts_cache
            else:
                slash_parts = set()
                for key in trie.keys():
                    if '/' in key:
                        parts = [p.strip().lower() for p in key.split('/')]
                        for part in parts:
                            if len(part) > 0:
                                slash_parts.add(part)
                # Cache for future calls
                self.slash_parts_cache = slash_parts
                
            # If this is a known part of a slash-separated skill, be extra strict
            if phrase_lower in slash_parts:
                log.debug(f"Strict matching for slash part: '{phrase}'")
                return phrase in trie and not trie.has_subtrie(phrase)
            
            # Normal case - check if the phrase is in the trie or is a viable prefix
            return phrase in trie or trie.has_subtrie(phrase)
        except Exception:
            return False
            
    def normalize_plural(self, phrase):
        """Normalize potential plural forms for better matching"""
        if not phrase:
            return phrase
            
        words = phrase.split()
        if len(words) <= 1:
            return phrase
            
        # Check if last word might be a plural
        last_word = words[-1]
        if last_word.endswith('s') and not last_word.endswith('ss'):
            # Try to normalize just the last word
            normalized_last = simple_normalize(last_word)
            if normalized_last != last_word:
                words[-1] = normalized_last
                return " ".join(words)
                
        return phrase
        
    def find_matches(self, segment, used, matched):
        """
        Find all possible skill matches in a segment
        
        Args:
            segment: List of tokens with positions
            used: Dictionary of already used positions
            matched: Set of already matched skills
            
        Returns:
            List of matches with position and metadata
        """
        # Store current segment for context checking
        self.current_segment = segment
        if not segment:
            return []
        
        # Initialize slash parts cache if needed (used in _has_viable_prefix)
        if not hasattr(self, 'slash_parts_cache'):
            # Pre-compute parts of slash-separated skills to avoid matching them independently
            self.slash_parts_cache = set()
            for trie in [self.exact, self.std, self.lemma]:
                for key in trie.keys():
                    if '/' in key:
                        parts = [p.strip().lower() for p in key.split('/')]
                        for part in parts:
                            if part and len(part) > 0:
                                self.slash_parts_cache.add(part)
            log.debug(f"Built slash parts cache with {len(self.slash_parts_cache)} entries")
            
        # Quick check optimization - look for ALL tokens to see if any match quick check words
        # This ensures we don't miss matches that begin with uncommon words
        all_tokens = set(tok.lower() for tok, _, _ in segment)  # Convert to lowercase for comparison
        if all_tokens.isdisjoint(self.qcw):
            # No word in this segment matches any word in any skill
            return []
            
        out = []
        i = 0
        
        while i < len(segment):
            # Skip if position already used
            if any(used.get(p, False) for p in range(segment[i][1], segment[i][2])):
                i += 1
                continue
                
            # Find ALL possible matches at this position with different lengths
            all_matches = []
            curr = []
            
            # Look ahead up to 15 tokens to find longest possible match
            max_lookahead = min(15, len(segment) - i)
            
            for j in range(max_lookahead):
                # Break if position already used
                if i+j >= len(segment) or any(used.get(p, False) for p in range(segment[i+j][1], segment[i+j][2])):
                    break
                    
                # Add current token and create the phrase
                curr.append(segment[i+j][0])
                phrase = " ".join(curr).lower()
                
                # Normalize plural forms for better matching
                normalized_phrase = self.normalize_plural(phrase)
                
                # Early termination - if no prefix match in any trie (performance optimization)
                prefix_viable = (
                    self._has_viable_prefix(self.exact, phrase) or 
                    self._has_viable_prefix(self.std, phrase) or 
                    self._has_viable_prefix(self.lemma, phrase) or
                    (normalized_phrase != phrase and (
                        self._has_viable_prefix(self.exact, normalized_phrase) or 
                        self._has_viable_prefix(self.std, normalized_phrase) or 
                        self._has_viable_prefix(self.lemma, normalized_phrase)
                    ))
                )
                
                if not prefix_viable:
                    break
                
                # Try to match the phrase in all tries
                match_info = None
                
                # Special handling for single-letter tokens - be more permissive
                if len(phrase.strip()) == 1:
                    # Pull more context from the original segment for surrounding text
                    start_pos = segment[i][1]
                    end_pos = segment[i][2]
                    
                    # Generate more context for analysis
                    position_in_segment = i
                    segment_str = ' '.join(t[0] for t in segment)
                    surround_text = self._get_surrounding_context(phrase, start_pos, end_pos, segment_str, position_in_segment)
                    
                    # Check if it's a valid single letter skill
                    if self._is_valid_single_letter_skill(phrase, surround_text):
                        log.debug(f"Allowing single-letter token '{phrase}' based on context: '{surround_text}'")
                    # Always check the database for exact single-letter skills
                    elif phrase.lower() in self.exact or phrase.lower() in self.std or phrase.lower() in self.lemma:
                        log.debug(f"Allowing single-letter token '{phrase}' as it's in the skills database")
                    else:
                        log.debug(f"Skipping ambiguous single-letter token: '{phrase}'")
                        continue
                
                # Check tries with both original and normalized phrases
                for curr_phrase in [phrase, normalized_phrase]:
                    # Skip single-letter phrases that weren't validated above
                    if len(curr_phrase.strip()) == 1 and not (
                        curr_phrase.lower() in self.exact or 
                        curr_phrase.lower() in self.std or 
                        curr_phrase.lower() in self.lemma or
                        curr_phrase.lower() in self.valid_single_letters
                    ):
                        continue
                        
                    if curr_phrase in self.exact and curr_phrase not in matched:
                        info = self.exact[curr_phrase]
                        nm = info["name"] if isinstance(info, dict) else info
                        bc = self.classes.get(curr_phrase, "Unknown")
                        
                        # Special validation for single-letter skills
                        if len(nm.strip()) == 1:
                            # Using our comprehensive validator
                            if not self.validate_single_letter_skill(
                                nm.strip(), segment_pos=i, 
                                token_start=segment[i][1], token_end=segment[i][2]
                            ):
                                log.debug(f"Skipping invalid single-letter skill name: '{nm}'")
                                continue
                            else:
                                log.debug(f"Keeping validated single-letter skill: '{nm}'")
                                # Continue processing this match
                                
                        # Block common words that should never match as standalone skills
                        stop_words = {"is", "was", "be", "in", "on", "at", "to", "or", "and", "but", 
                                      "of", "for", "a", "an", "it", "its", "by", "as", "are", "were",
                                      "am", "can", "will", "do", "does", "did", "has", "have", "had"}
                        if curr_phrase.strip().lower() in stop_words:
                            log.debug(f"Blocking stop word '{curr_phrase}' as a skill")
                            continue
                                
                        # Determine if this is a part of a slash-containing skill
                        if '/' in nm:
                            parts = [p.strip().lower() for p in nm.split('/')]
                            if curr_phrase.strip().lower() in parts:
                                log.debug(f"Blocking '{curr_phrase}' as it's just a part of '{nm}'")
                                continue
                        
                        # Also check if this is a part of ANY slash-containing skill using our cached parts
                        if hasattr(self, 'slash_parts_cache') and curr_phrase.strip().lower() in self.slash_parts_cache:
                            # Check if we're matching this as a standalone token or as part of a longer phrase
                            is_standalone = (j == 0)  # This is a standalone token match
                            if is_standalone:
                                log.debug(f"Blocking '{curr_phrase}' as it's a part of a slash-separated skill")
                                continue
                        
                        # Check for false positives with generic terms like "tools"
                        if curr_phrase.lower() in SKILL_PATTERNS.get("no_lemmatize_terms", []):
                            # For common generic terms, original skill must be more specific
                            # like "Yokogawa FAST/TOOLS" not just "tools"
                            if len(nm.split()) < 2 or nm.lower() == curr_phrase.lower():
                                # This is a generic term without specific context
                                log.debug(f"Blocking generic term '{curr_phrase}' without product name")
                                continue
                                
                            # Look at surrounding context for product name indicators
                            if not self.has_product_name_context(segment, i, j):
                                log.debug(f"Blocking '{curr_phrase}' - insufficient product context")
                                continue
                            
                        match_info = {
                            "skill": nm,
                            "match_type": "Exact Match",
                            "classification": bc,
                            "length": j+1,
                            "phrase": curr_phrase,
                            "token_count": len(curr_phrase.split()),  # For prioritizing multi-word skills
                            "priority": 4,  # Highest priority: exact match
                            "start": segment[i][1],
                            "end": segment[i+j][2]
                        }
                        break
                        
                    elif curr_phrase in self.std and curr_phrase not in matched:
                        info = self.std[curr_phrase]
                        nm = info["name"] if isinstance(info, dict) else info
                        bc = self.classes.get(curr_phrase, "Unknown")
                        
                        # Special validation for single-letter skills
                        if len(nm.strip()) == 1:
                            # Using our comprehensive validator
                            if not self.validate_single_letter_skill(
                                nm.strip(), segment_pos=i, 
                                token_start=segment[i][1], token_end=segment[i][2]
                            ):
                                log.debug(f"Skipping invalid single-letter skill name: '{nm}'")
                                continue
                            else:
                                log.debug(f"Keeping validated single-letter skill: '{nm}'")
                                # Continue processing this match
                            
                        match_info = {
                            "skill": nm,
                            "match_type": "Standardized Match",
                            "classification": bc,
                            "length": j+1,
                            "phrase": curr_phrase,
                            "token_count": len(curr_phrase.split()),
                            "priority": 3,  # Medium-high priority: standardized match
                            "start": segment[i][1],
                            "end": segment[i+j][2]
                        }
                        break
                        
                    elif curr_phrase in self.lemma and curr_phrase not in matched:
                        info = self.lemma[curr_phrase]
                        nm = info["name"] if isinstance(info, dict) else info
                        
                        # Special validation for single-letter skills
                        if len(nm.strip()) == 1:
                            # Using our comprehensive validator
                            if not self.validate_single_letter_skill(
                                nm.strip(), segment_pos=i, 
                                token_start=segment[i][1], token_end=segment[i][2]
                            ):
                                log.debug(f"Skipping invalid single-letter skill name: '{nm}'")
                                continue
                            else:
                                log.debug(f"Keeping validated single-letter skill: '{nm}'")
                                # Continue processing this match
                            
                        # Find classification from original skill, not lemmatized form
                        original_skill_norm = normalize_text(nm)
                        bc = self.classes.get(original_skill_norm, self.classes.get(curr_phrase, "Unknown"))
                        
                        match_info = {
                            "skill": nm,
                            "match_type": "Lemma Match",
                            "classification": bc,
                            "length": j+1,
                            "phrase": curr_phrase,
                            "token_count": len(curr_phrase.split()),
                            "priority": 2,  # Medium priority: lemma match
                            "start": segment[i][1],
                            "end": segment[i+j][2]
                        }
                        break
                
                # If we found a match, add it to candidates
                if match_info:
                    all_matches.append(match_info)
            
            # After examining all possible lengths, select the best match
            if all_matches:
                # First prioritize by longest token count (multi-word skills)
                # Then by length in tokens, then by priority (exact > std > lemma)
                best_match = sorted(
                    all_matches, 
                    key=lambda x: (x["token_count"], x["length"], x["priority"]),
                    reverse=True
                )[0]
                
                # Add to results
                out.append({
                    "skill": best_match["skill"],
                    "match_type": best_match["match_type"],
                    "classification": best_match["classification"],
                    "start": best_match["start"],
                    "end": best_match["end"]
                })
                
                # Mark positions as used
                for p in range(best_match["start"], best_match["end"]):
                    used[p] = True
                    
                matched.add(best_match["skill"])
                i += best_match["length"]
            else:
                i += 1
                
        return out

# ———————— Main Extraction Function ————————————————————————————————————
def extract_skills(txt, exact, std, lemma, is_std, qcw, classes):
    """
    Extract skills from text using an enhanced algorithm that:
    1. Prioritizes multi-word skills over single-word skills
    2. Uses context for disambiguation
    3. Handles plural forms and variations elegantly
    4. Special handling for compound terms
    """
    log.info("Extracting skills…")
    t0 = time.time()
    
    try:
        # Preprocess for single-letter skills
        # Look for common patterns for single-letter programming languages
        txt_processed = txt
        
        # Check for single-letter skills in our database
        single_letter_skills = []
        for letter in SKILL_PATTERNS.get("valid_single_letters", []):
            if letter in exact or letter in std or letter in lemma:
                single_letter_skills.append(letter.upper())
                
        log.debug(f"Detected single-letter skills in database: {single_letter_skills}")
        
        # Special direct pattern matching for R, M and other single-letter languages
        single_letter_patterns = [
            r'\b([RM])\s+language\b',
            r'\bexperience\s+(?:with|in)\s+([RM])\b',
            r'\bknowledge\s+of\s+([RM])\b',
            r'\b(?:code|coding)\s+in\s+([RM])\b',
            r'\busing\s+([RM])\b',
            r'\bin\s+([RM])\b'
        ]
        
        # Special handling for skills in special contexts like "continuous integration/build automation"
        special_context_patterns = [
            # The specific case mentioned - continuous integration/build automation
            (r'continuous\s+integration/build\s+automation', ['continuous integration', 'build automation']),
            # Other common slash-separated skills
            (r'ci/cd', ['continuous integration', 'continuous deployment']),
            (r'html/css', ['html', 'css']),
            # Parenthetical skills
            (r'\(([^)]*?build\s+automation[^)]*?)\)', ['build automation']),
            (r'\(([^)]*?continuous\s+integration[^)]*?)\)', ['continuous integration']),
            (r'\(([^)]*?quality\s+assurance[^)]*?)\)', ['quality assurance']),
            # Specific for build automation in various contexts
            (r'build\s+automation', ['build automation'])
        ]
        
        # Apply the special context patterns
        for pattern, skills in special_context_patterns:
            try:
                for match in re.finditer(pattern, txt, re.IGNORECASE):
                    match_start, match_end = match.span()
                    for skill in skills:
                        # Add a specific token for each skill in this context
                        skill_start = txt.lower().find(skill.lower(), match_start, match_end)
                        if skill_start >= 0:
                            skill_end = skill_start + len(skill)
                            # Add the token
                            toks.append((skill, skill_start, skill_end))
                            log.debug(f"Added special context skill: '{skill}' at {skill_start}-{skill_end}")
                        else:
                            # If skill not found directly within match, add at match position
                            toks.append((skill, match_start, match_end))
                            log.debug(f"Added implied skill from context: '{skill}' at {match_start}-{match_end}")
            except Exception as e:
                log.warning(f"Error processing special context pattern {pattern}: {e}")
        
        # Tokenize and segment
        toks = tokenize_with_indices(txt)
        
        # Add special tokens for single-letter languages
        for pattern in single_letter_patterns:
            for match in re.finditer(pattern, txt, re.IGNORECASE):
                letter = match.group(1)
                start, end = match.span(1)
                # Add a token specifically for this match
                toks.append((letter, start, end))
                log.debug(f"Added special token for single-letter language: {letter} at {start}-{end}")
        
        # Re-sort tokens by position
        toks.sort(key=lambda x: x[1])
        
        segs = get_contiguous_segments(toks)
        
        if not segs:
            log.warning("No segments found in text")
            return []
        
        # Initialize matchers
        trie_matcher = TrieMatcher(exact, std, lemma, is_std, qcw, classes, full_text=txt)
        context_disambiguator = ContextDisambiguator()
        
        # Configure thread pool
        cpu_count = os.cpu_count() or 4
        max_workers = min(16, cpu_count * 2)  # Use more threads for IO-bound work
        log.info(f"Using {max_workers} worker threads for parallel processing")
        
        # Process in parallel with progress reporting
        def process_segment(segment):
            used = {}
            matched = set()
            return trie_matcher.find_matches(segment, used, matched)
            
        # Run all segments in parallel
        allm = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Submit all tasks
            future_to_segment = {ex.submit(process_segment, seg): i for i, seg in enumerate(segs)}
            
            # Process results as they complete
            total = len(future_to_segment)
            complete = 0
            for future in as_completed(future_to_segment):
                idx = future_to_segment[future]
                try:
                    result = future.result()
                    allm.extend(result)
                except Exception as e:
                    log.warning(f"Error processing segment {idx}: {e}")
                    log.debug(traceback.format_exc())
                
                # Update progress every 10%
                complete += 1
                if complete % max(1, total // 10) == 0 or complete == total:
                    progress = (complete / total) * 100
                    log.info(f"Progress: {progress:.1f}% ({complete}/{total} segments)")
        
        # Special direct pattern handling for critical compound skills
        # This ensures we catch skills like "clinical psychology" even if segment boundaries interfere
        log.info("Performing direct pattern matching for critical compound skills...")
        compound_terms = SKILL_PATTERNS.get("compound_terms", [])
        
        for term in compound_terms:
            # Check if the term exists in our skill database
            term_norm = normalize_text(term)
            if term_norm in exact or term_norm in std or term_norm in lemma:
                # Find all occurrences with word boundaries
                pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(txt.lower()):
                    start, end = match.span()
                    
                    # Get the classification
                    if term_norm in exact:
                        info = exact[term_norm]
                        match_type = "Exact Match"
                        priority = 4
                    elif term_norm in std:
                        info = std[term_norm]
                        match_type = "Standardized Match"
                        priority = 3
                    elif term_norm in lemma:
                        info = lemma[term_norm]
                        match_type = "Lemma Match"
                        priority = 2
                    else:
                        continue
                    
                    skill_name = info["name"] if isinstance(info, dict) else info
                    classification = classes.get(term_norm, "Unknown")
                    
                    # Add direct match with high priority
                    allm.append({
                        "skill": skill_name,
                        "match_type": match_type + " (Direct)",
                        "classification": classification,
                        "start": start,
                        "end": end,
                        "direct_match": True  # Flag to prioritize in deduplication
                    })
                    log.debug(f"Added direct pattern match for '{term}' at {start}-{end}")
        
        # Apply context disambiguation to filter out non-technical usages
        log.info("Applying context disambiguation to filter ambiguous terms...")
        filtered = []
        for m in allm:
            # Skip direct matches that were added manually - they're already validated
            if m.get("direct_match", False) or context_disambiguator.is_technical_skill(txt, m["start"], m["end"], m["skill"]):
                filtered.append(m)
            else:
                log.debug(f"Filtered ambiguous term: {m['skill']}")
        
        # Find duplicates and overlaps, keeping the longer, more specific versions
        log.info("Deduplicating and handling overlapping skills...")
        
        # Separate direct matches from normal matches
        direct_matches = [m for m in filtered if m.get("direct_match", False)]
        normal_matches = [m for m in filtered if not m.get("direct_match", False)]
        
        # First sort ALL matches by token count (prioritizing multi-word skills)
        # This ensures "clinical psychology" is processed before "psychology"
        # regardless of position in text
        all_sorted_matches = sorted(
            filtered,
            key=lambda x: (-len(x["skill"].split()), 
                          x.get("direct_match", False),  # Direct matches as tiebreaker
                          x["start"])
        )
        
        # Process matches in token-count order
        kept_matches = []
        used_positions = set()
        
        for match in all_sorted_matches:
            match_range = set(range(match["start"], match["end"]))
            
            # If it's a direct match or doesn't overlap with existing matches, keep it
            if match.get("direct_match", False) or match_range.isdisjoint(used_positions):
                kept_matches.append(match)
                used_positions.update(match_range)
                if match.get("direct_match", False):
                    log.debug(f"Keeping direct match: {match['skill']}")
                else:
                    log.debug(f"Keeping non-overlapping skill: {match['skill']}")
            else:
                # For partial overlaps, keep if significant new content
                overlap_size = len(match_range.intersection(used_positions))
                match_size = match["end"] - match["start"]
                
                if overlap_size < match_size / 2:
                    log.debug(f"Keeping partially overlapping skill: {match['skill']}")
                    kept_matches.append(match)
                    used_positions.update(match_range)
                else:
                    log.debug(f"Filtered overlapping skill: {match['skill']}")
                    
        # We don't need to sort again because we process all matches together
        
        # We no longer need this section as all processing is now done in the sorted order loop above
                    
        # Additional deduplication for identical skill names that don't overlap
        skill_to_idx = {}
        for i, m in enumerate(kept_matches):
            skill_lower = m["skill"].lower()
            token_count = len(skill_lower.split())
            
            if skill_lower in skill_to_idx:
                # Compare token count, keep the longer version
                existing_idx = skill_to_idx[skill_lower]
                existing_tokens = len(kept_matches[existing_idx]["skill"].lower().split())
                
                if token_count > existing_tokens:
                    # Replace with longer version
                    kept_matches[existing_idx] = m
            else:
                skill_to_idx[skill_lower] = i
        
        # Convert to final output list
        out = [m for i, m in enumerate(kept_matches) if i in set(skill_to_idx.values())]
        
        # Apply aggressive filtering to remove common false positives
        # Filter out standalone common words that got through the earlier checks
        stop_words = {"is", "was", "be", "in", "on", "at", "to", "or", "and", "but", 
                     "of", "for", "a", "an", "it", "its", "by", "as", "are", "were",
                     "am", "can", "will", "do", "does", "did", "has", "have", "had", "tools"}
        
        # Extract parts of slash skills to block them from matching independently
        slash_parts = set()
        common_generic_terms = {
            # Basic generic terms
            "field", "force", "system", "process", "model", "tool", "method", 
            "analysis", "design", "management", "development", "service",
            "technology", "solution", "application", "framework", "platform",
            
            # Additional generic terms
            "approach", "strategy", "technique", "function", "procedure", "resource",
            "capability", "feature", "concept", "structure", "interface", "module",
            "component", "element", "format", "standard", "protocol", "unit", "phase",
            "stage", "level", "layer", "stack", "domain", "area", "sector", "industry",
            "category", "group", "class", "type", "form", "mode", "style", "pattern",
            
            # Generic technical terms
            "algorithm", "code", "language", "program", "software", "hardware", "network",
            "database", "data", "file", "document", "record", "report", "information",
            "content", "material", "object", "entity", "device", "equipment", "machine",
            
            # Basic single words that need context
            "plan", "test", "build", "run", "set", "track", "check", "review", "audit",
            "assess", "evaluate", "monitor", "measure", "control", "manage", "direct",
            "lead", "guide", "support", "assist", "help", "enable", "facilitate",
            
            # Operations-related terms
            "operation", "workflow", "pipeline", "chain", "sequence", "series", "cycle",
            "iteration", "version", "release", "update", "maintenance", "support",
            
            # Misc general terms
            "quality", "performance", "efficiency", "effectiveness", "productivity",
            "improvement", "enhancement", "optimization", "innovation", "transformation"
        }
        
        for m in out:
            if '/' in m["skill"]:
                # Extract parts to block them
                parts = [p.strip().lower() for p in m["skill"].split('/')]
                for part in parts:
                    if len(part) > 0:
                        slash_parts.add(part)
                        
                        # Also add individual words from each part if they're generic terms
                        # This prevents matching just "field" from "Gravitational Force / Field"
                        words = part.split()
                        for word in words:
                            if word.lower() in common_generic_terms:
                                slash_parts.add(word.lower())
                                log.debug(f"Adding generic term '{word}' from slash skill '{m['skill']}' to blocked terms")
                    
        # Final filter to remove common false positives
        final_out = []
        for m in out:
            skill_lower = m["skill"].lower()
            
            # Always keep skills with slashes
            if '/' in skill_lower:
                final_out.append(m)
                continue
                
            # Block common stop words
            if skill_lower in stop_words:
                log.debug(f"Final filter removing '{m['skill']}' (common word)")
                continue
                
            # Block parts of slash skills
            if skill_lower in slash_parts:
                log.debug(f"Final filter removing '{m['skill']}' (part of slash-skill)")
                continue
                
            # More aggressively filter common generic terms
            # Only keep them if they have multiple words or are part of multi-word skills
            words = skill_lower.split()
            if len(words) == 1 and words[0] in common_generic_terms:
                # Check the context to see if this is part of a larger phrase
                # by looking at surrounding tokens
                match_start, match_end = m["start"], m["end"]
                context_window = 10  # Look at 10 tokens before and after
                
                # If this appears to be a standalone generic term, filter it out
                log.debug(f"Final filter removing '{m['skill']}' (standalone generic term)")
                continue
                
            # Keep everything else
            final_out.append(m)
        
        # Create two versions of the output:
        # 1. Sorted by position (for highlighting in the text)
        text_ordered = sorted(final_out, key=lambda x: x["start"])
        
        # 2. Sorted alphabetically by skill name (for the skills list display)
        alpha_ordered = sorted(out, key=lambda x: x["skill"].lower())
        
        log.info(f"Found {len(out)} unique skill matches in {time.time()-t0:.2f}s")
        return {"text_ordered": text_ordered, "alpha_ordered": alpha_ordered}
    
    except Exception as e:
        log.error(f"Error during skill extraction: {e}")
        log.debug(traceback.format_exc())
        return []

# ———————— Text Output Generation ———————————————————————————————————
def generate_text_output(matches_dict, output_file="skill_ids.txt"):
    """Generate text output with just skill IDs"""
    try:
        # Use Downloads folder for output if relative path provided
        if not os.path.isabs(output_file):
            downloads_dir = Path(os.path.expanduser("~")) / "Downloads"
            output_file = str(downloads_dir / Path(output_file).name)
            
        log.info(f"Generating text output with skill IDs to {output_file}...")
        t0 = time.time()
        
        # Extract the matches (use alpha_ordered for consistent output)
        alpha_ordered_matches = matches_dict.get("alpha_ordered", [])
        
        # Generate text output with one skill ID per line
        lines = []
        for match_item in alpha_ordered_matches:
            # Get skill ID if available, otherwise use the skill name
            skill_id = match_item.get("id", "unknown")
            skill_name = match_item.get("skill", "")
            
            if skill_id != "unknown":
                lines.append(f"{skill_id}")
            else:
                # If no ID is available, log a warning and use the skill name
                log.warning(f"No ID found for skill: {skill_name}")
                lines.append(f"no_id:{skill_name}")
        
        # Join lines with newlines
        text_output = "\n".join(lines)
        
        # Write output with robust error handling
        try:
            Path(output_file).write_text(text_output, encoding="utf-8")
        except Exception as e:
            # Try an alternative location as fallback
            alt_path = f"skill_ids_{int(time.time())}.txt"
            log.warning(f"Error writing to {output_file}, trying alternate: {alt_path}")
            Path(alt_path).write_text(text_output, encoding="utf-8")
            output_file = alt_path
                
        log.info(f"Text output generated successfully in {time.time()-t0:.2f}s")
        return output_file
        
    except Exception as e:
        log.error(f"Error generating text output: {e}")
        log.debug(traceback.format_exc())
        # Create emergency output in the current directory
        try:
            emergency_file = f"skill_ids_emergency_{int(time.time())}.txt"
            with open(emergency_file, 'w', encoding="utf-8") as f:
                f.write("Error generating skill IDs output\n")
                for m in matches_dict.get("alpha_ordered", []):
                    f.write(f"{m.get('skill', 'unknown')}\n")
            log.warning(f"Created emergency text output: {emergency_file}")
            return emergency_file
        except Exception:
            log.error("Failed to create emergency text output")
            return output_file

# ———————— HTML Output Generation ———————————————————————————————————
def generate_html_output(text, matches_dict, output_file="skills_output.html"):
    """Generate HTML output with error handling and memory-efficient processing"""
    # Try to use Downloads folder for output
    try:
        downloads_dir = Path(os.path.expanduser("~")) / "Downloads"
        output_file = str(downloads_dir / Path(output_file).name)
    except Exception:
        output_file = str(output_file)
        
    log.info(f"Generating HTML output to {output_file}…")
    t0 = time.time()
    
    try:
        # Extract our two differently sorted match lists
        text_ordered_matches = matches_dict["text_ordered"]  # For text highlighting (position order)
        alpha_ordered_matches = matches_dict["alpha_ordered"]  # For skills list (alphabetical order)
        
        # Generate highlighted text in chunks for memory efficiency
        def generate_highlighted_chunks():
            last_pos = 0
            # Use position-sorted matches for text highlighting
            for i, match_item in enumerate(text_ordered_matches):
                s, e = match_item["start"], match_item["end"]
                # Text before match
                if s > last_pos:
                    yield text[last_pos:s]
                # Highlighted match
                # Store the alpha index for proper linking to the alphabetized list
                alpha_idx = next((j for j, am in enumerate(alpha_ordered_matches) 
                                 if am["start"] == match_item["start"] and am["end"] == match_item["end"]), i)
                yield (
                    f'<mark id="match-{i}" class="highlight" '
                    f'data-alpha-idx="{alpha_idx}" '
                    f'onclick="highlightMatch(\'match-{i}\', {alpha_idx})">{text[s:e]}</mark>'
                )
                last_pos = e
            # Remaining text after last match
            if last_pos < len(text):
                yield text[last_pos:]
                
        # Join highlighted chunks
        hl = "".join(generate_highlighted_chunks())
        
        # Generate skills list (alphabetically ordered)
        skills_list = "".join(
            f'<li id="list-{i}" class="skill-item" '
            f'data-text-idx="{next((j for j, tm in enumerate(text_ordered_matches) if tm["start"] == match_item["start"] and tm["end"] == match_item["end"]), i)}" '
            f'onclick="highlightMatchFromList(\'list-{i}\')">'
            f'<strong>{match_item["skill"]}</strong> '
            f'<em>({match_item["match_type"]})</em> '
            f'<span>{match_item["classification"]}</span></li>'
            for i, match_item in enumerate(alpha_ordered_matches)
        )
        
        # Generate statistics
        match_types_count, classification_count = {}, {}
        for match_item in text_ordered_matches:  # Either list works for stats
            match_types_count[match_item["match_type"]] = match_types_count.get(match_item["match_type"], 0) + 1
            c = match_item["classification"]
            classification_count[c] = classification_count.get(c, 0) + 1
            
        type_summary = "".join(f'<div><b>{k}:</b> {v}</div>' for k, v in match_types_count.items())
        class_summary = "".join(f'<div><b>{k}:</b> {v}</div>' for k, v in classification_count.items())
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skill Extraction Result</title><style>
  body{{display:flex;margin:0;font-family:Arial, sans-serif}}
  .pane{{width:50%;padding:20px;box-sizing:border-box;overflow-y:auto;height:100vh}}
  .left{{background:#f9f9f9;border-right:2px solid #ddd}}
  .right{{background:#fff}}
  mark.highlight{{background:#ffffcc;padding:2px;border-radius:3px;cursor:pointer}}
  mark.highlight.active{{background:#ffcc00!important}}
  .skill-item{{cursor:pointer;padding:8px;margin:4px 0;border-radius:4px}}
  .skill-item.active{{background:#ffcc00!important;color:#000}}
  .stats{{background:#eef2f7;padding:15px;border-radius:4px;margin-bottom:20px}}
</style></head><body>
  <div class="pane left"><h2>Resume Text</h2><p>{hl}</p></div>
  <div class="pane right"><h2>Extracted Skills</h2><div class="stats">
    <div><strong>Total Skills:</strong> {len(alpha_ordered_matches)}</div>
    <h3>Match Types:</h3>{type_summary}
    <h3>Classifications:</h3>{class_summary}
  </div><ul>{skills_list}</ul></div>
<script>
function highlightMatch(id, alphaIdx){{
  // Clear all highlights
  document.querySelectorAll('.highlight, .skill-item').forEach(el=>el.classList.remove('active'));
  
  // Highlight the text match
  let mk = document.getElementById(id);
  if(mk){{
    mk.classList.add('active');
    mk.scrollIntoView({{behavior:'smooth',block:'center'}});
  }}
  
  // Highlight the corresponding list item (in alphabetical list)
  let li = document.getElementById('list-' + alphaIdx);
  if(li){{
    li.classList.add('active');
    li.scrollIntoView({{behavior:'smooth',block:'center'}});
  }}
}}

function highlightMatchFromList(listId){{
  // Clear all highlights
  document.querySelectorAll('.highlight, .skill-item').forEach(el=>el.classList.remove('active'));
  
  // Highlight the list item
  let li = document.getElementById(listId);
  if(li){{
    li.classList.add('active');
    
    // Get the text index from the list item
    let textIdx = li.getAttribute('data-text-idx');
    
    // Highlight corresponding text match
    let mk = document.getElementById('match-' + textIdx);
    if(mk){{
      mk.classList.add('active');
      mk.scrollIntoView({{behavior:'smooth',block:'center'}});
      
      // After scrolling to the text, scroll back to the list item
      setTimeout(() => {{
        li.scrollIntoView({{behavior:'smooth',block:'center'}});
      }}, 1000);
    }}
  }}
}}

// Initialize with first skill highlighted
document.addEventListener('DOMContentLoaded', () => {{
  if(document.querySelectorAll('.skill-item').length > 0) {{
    // Get the alpha index of the first match
    let firstMark = document.getElementById('match-0');
    let alphaIdx = firstMark ? firstMark.getAttribute('data-alpha-idx') : 0;
    highlightMatch('match-0', alphaIdx);
  }}
}});
</script></body></html>"""
        
        # Write output with robust error handling
        try:
            Path(output_file).write_text(html, encoding="utf-8")
        except OSError as e:
            if e.errno == 9:  # EBADF
                log.warning("Path.write_text EBADF; falling back to built-in open()")
                with open(output_file, 'w', encoding="utf-8") as f:
                    f.write(html)
            else:
                # Try an alternative location as fallback
                alt_path = f"skills_output_{int(time.time())}.html"
                log.warning(f"Error writing to {output_file}, trying alternate: {alt_path}")
                with open(alt_path, 'w', encoding="utf-8") as f:
                    f.write(html)
                output_file = alt_path
                
        log.info(f"HTML output generated successfully in {time.time()-t0:.2f}s")
        return output_file
        
    except Exception as e:
        log.error(f"Error generating HTML output: {e}")
        log.debug(traceback.format_exc())
        
        # Create a minimal emergency output as fallback
        try:
            emergency_file = f"skills_emergency_{int(time.time())}.html"
            minimal_html = f"""<!DOCTYPE html>
<html><head><title>Skills (Emergency Output)</title></head><body>
<h1>Skills Extracted</h1>
<p>Error generating formatted output: {str(e)}</p>
<ul>{''.join(f'<li>{m["skill"]}</li>' for m in matches_dict.get("alpha_ordered", []))}</ul>
</body></html>"""
            with open(emergency_file, 'w', encoding="utf-8") as f:
                f.write(minimal_html)
            log.warning(f"Created emergency output: {emergency_file}")
            return emergency_file
        except Exception:
            log.error("Failed to create emergency output")
            return output_file

# ———————— Main Entry Point ——————————————————————————————————————
def main():
    """Main entry point with comprehensive error handling"""
    try:
        parser = argparse.ArgumentParser(description="Extract skills from a résumé")
        parser.add_argument("--preprocessed", type=str, required=True, help="Path to .skillsdb")
        parser.add_argument("--resume", type=str, required=True, help="Path to résumé .docx")
        parser.add_argument("--output", type=str, default="skills_output.html", help="Output HTML file")
        parser.add_argument("--text-output", action="store_true", default=False, help="Generate text output with skill IDs")
        parser.add_argument("--text-file", type=str, default="skill_ids.txt", help="Output text file for skill IDs")
        parser.add_argument("--no-open", action="store_false", dest="open", default=True, help="Do not open browser")
        parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging")
        parser.add_argument("--config", type=str, help="Path to custom skill patterns configuration (JSON format)")
        
        args = parser.parse_args()
        
        global DEBUG_MODE, SKILL_PATTERNS
        DEBUG_MODE = args.debug
        if DEBUG_MODE:
            logging.getLogger().setLevel(logging.DEBUG)
            log.info("Debug mode enabled")
            
        # Load custom configuration if specified
        if args.config:
            SKILL_PATTERNS = load_skill_patterns(args.config)
            log.info(f"Loaded custom skill patterns from {args.config}")
            
        # Time the overall process
        overall_start = time.time()
            
        # Verify resume file exists
        resume_path = Path(args.resume).absolute()
        if not resume_path.exists():
            log.error(f"Resume not found: {resume_path}")
            return 1
            
        # Load skills database
        try:
            exact, std, lemma, is_std, qcw, classes = load_preprocessed_skills(Path(args.preprocessed))
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            log.error(f"Skills database error: {e}")
            return 1
            
        # Extract text from resume
        try:
            text = extract_text_from_docx(resume_path)
            if not text.strip():
                log.error("No text extracted from resume")
                return 1
        except Exception as e:
            log.error(f"Text extraction error: {e}")
            return 1
            
        # Extract skills
        matches_dict = extract_skills(text, exact, std, lemma, is_std, qcw, classes)
        
        # Ensure skill ID is included in each match if available
        for match_list in [matches_dict["text_ordered"], matches_dict["alpha_ordered"]]:
            for match in match_list:
                skill_name = match["skill"]
                # Check if the skill has an ID in the exact match trie
                norm_skill = normalize_text(skill_name)
                if norm_skill in exact and isinstance(exact[norm_skill], dict) and "id" in exact[norm_skill]:
                    match["id"] = exact[norm_skill]["id"]
        
        # Generate output
        try:
            # Always generate HTML output
            out = generate_html_output(text, matches_dict, args.output)
            
            # Generate text output if requested
            if args.text_output:
                text_out = generate_text_output(matches_dict, args.text_file)
                log.info(f"Text output with skill IDs saved to: {text_out}")
        except Exception as e:
            log.error(f"Output generation error: {e}")
            return 1
            
        # Open browser if requested
        if args.open:
            try:
                import webbrowser
                webbrowser.open(f"file://{Path(out).absolute()}")
            except Exception as e:
                log.error(f"Error opening browser: {e}")
                
        overall_time = time.time() - overall_start
        log.info(f"Done. HTML results saved to: {out}")
        if args.text_output:
            log.info(f"Skill IDs text output saved to: {text_out}")
        log.info(f"Total processing time: {overall_time:.2f} seconds")
        return 0
        
    except KeyboardInterrupt:
        log.info("Process interrupted by user")
        return 130
    except Exception as e:
        log.error(f"Unhandled error: {e}")
        log.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())