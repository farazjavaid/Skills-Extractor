# extractor/views.py - Complete file with overlap prevention

import re
import string
from urllib import response
from venv import logger
from extractor.utils import load_skills_from_db
from extractorv6 import TrieMatcher, tokenize_with_indices, get_contiguous_segments
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.db.models import Q
from django.db import connection
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import traceback
from extractor.models import Skill, SkillsDemand
from extractor.skill_overlap_service import SkillOverlapAnalyzer
from urllib.parse import unquote
import os
import PyPDF2
import docx
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import tempfile

from extractorv6 import (
    tokenize_with_indices,
    load_preprocessed_skills,
    get_contiguous_segments
)
from .models import BatchResumeAnalysis, ResumeMatchResult, Skill, SkillSimilarity, SkillsDemand, Folder, File

# Load DB once on startup
# exact, std, lemma, qcw, classes = load_skills_from_db()

from django.core.cache import cache
import threading

from extractor import models

_db_cache = {}
_db_lock = threading.Lock()

def get_cached_skills_db():
    """Get cached skills database with thread-safe loading"""
    global _db_cache
    
    if not _db_cache:
        with _db_lock:
            if not _db_cache:  # Double-check pattern
                print("🔄 Loading skills database (one-time)...")
                _db_cache['data'] = load_skills_from_db()
                print("✅ Skills database cached")
    
    return _db_cache['data']


def extract_parenthetical_skills(text):
    """
    Extract individual skills from parenthetical lists and VALIDATE against original text
    """
    parenthetical_skills = []
    
    # 1. PARENTHETICAL LISTS: word/phrase followed by parentheses with comma-separated items
    paren_pattern = r'(\w+(?:\s+\w+)*)\s*\(([^)]+)\)'
    
    for match in re.finditer(paren_pattern, text, re.IGNORECASE):
        base_phrase = match.group(1).strip()
        paren_content = match.group(2).strip()
        
        # Add the base phrase as a skill
        base_start = match.start(1)
        base_end = match.end(1)
        parenthetical_skills.append((base_phrase, base_start, base_end))
        print(f"📝 PARENTHETICAL BASE: Found '{base_phrase}' at {base_start}-{base_end}")
        
        # Split and add individual items from parentheses
        items = [item.strip() for item in paren_content.split(',')]
        
        # Calculate positions for each item
        paren_start = match.start(2)
        current_offset = 0
        
        for item in items:
            if item and len(item.strip()) > 1:
                item = item.strip()
                
                # Find the item's position within the parenthetical content
                item_pos_in_content = paren_content.find(item, current_offset)
                if item_pos_in_content >= 0:
                    item_start = paren_start + item_pos_in_content
                    item_end = item_start + len(item)
                    parenthetical_skills.append((item, item_start, item_end))
                    current_offset = item_pos_in_content + len(item)
                    print(f"📝 PARENTHETICAL ITEM: Found '{item}' at {item_start}-{item_end}")
    
    # 2. COMPOUND SKILLS: Only extract if EXACTLY present in text
    compound_patterns = [
        (r'\b3d\s+modell?ing\b', '3D modelling'),
        (r'\bteamwork\s*&\s*collaboration\b', 'Teamwork & Collaboration'),
        (r'\bproblem[-\s]solving\s*(?:and|&)?\s*critical\s+thinking\b', 'Problem-solving and Critical Thinking'),
        (r'\bcopywriting\s*&\s*content\s+writing\b', 'Copywriting & Content Writing'),
        (r'\bui/ux\b', 'UI/UX'),
        (r'\buser\s+interface\s*/\s*user\s+experience\b', 'UI/UX'),
        (r'\bfinancial\s+analysis\b', 'Financial analysis'),
        (r'\bweb\s+development\b', 'Web Development'),
        (r'\bvideo\s+editing\b', 'Video Editing'),
        (r'\bdigital\s+marketing\b', 'Digital Marketing'),
        (r'\bcontent\s+writing\b', 'Content Writing'),
        (r'\bcloud\s+computing\b', 'Cloud Computing'),
        (r'\bmachine\s+learning\b', 'Machine Learning'),
    ]
    
    for pattern, skill_name in compound_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            parenthetical_skills.append((skill_name, start, end))
            print(f"📝 COMPOUND SKILL: Found '{skill_name}' at {start}-{end}")
    
    # 3. REMOVE STANDALONE SKILLS SECTION - This was causing false matches
    # We should only extract skills that are explicitly present, not inferred
    
    # 4. AMPERSAND SKILLS: Skills connected with & - but validate they exist
    ampersand_pattern = r'(\w+(?:\s+\w+)*)\s*&\s*(\w+(?:\s+\w+)*)'
    for match in re.finditer(ampersand_pattern, text, re.IGNORECASE):
        full_match = match.group(0).strip()
        skill1 = match.group(1).strip()
        skill2 = match.group(2).strip()
        
        # Add the combined skill
        start, end = match.span()
        parenthetical_skills.append((full_match, start, end))
        print(f"📝 AMPERSAND COMBINED: Found '{full_match}' at {start}-{end}")
        
        # DON'T add individual parts automatically - this causes false matches
        # Only add if they appear as standalone skills elsewhere in text
    
    return parenthetical_skills

def validate_skill_in_text(skill_name, original_text, start_pos=None, end_pos=None):
    """
    STRICT validation that a skill actually exists in the original text
    NO plural/singular conversions allowed - must match exactly what's in text
    """
    if not skill_name or not original_text:
        return False
    
    # If we have exact positions, validate the text at those positions
    if start_pos is not None and end_pos is not None:
        extracted_text = original_text[start_pos:end_pos].lower().strip()
        skill_lower = skill_name.lower().strip()
        
        # STRICT: Must be exact match
        if extracted_text == skill_lower:
            return True
            
        # DO NOT allow any plural/singular conversions
        # The skill name must match exactly what's in the text
        return False
    
    # If no exact positions, check if skill exists anywhere in text with EXACT matching
    skill_lower = skill_name.lower().strip()
    text_lower = original_text.lower()
    
    # Check for exact word boundary match ONLY
    import re
    pattern = r'\b' + re.escape(skill_lower) + r'\b'
    if re.search(pattern, text_lower):
        return True
    
    # DO NOT allow plural variations - be strict
    # If the database has "Software Developers" but text has "software developer"
    # then reject it - only extract what's actually in the text
    
    return False

def get_exact_skill_from_database(text_phrase, exact, std, lemma):
    """
    Get the skill from database that EXACTLY matches the text phrase
    Prioritizes exact form over variations
    """
    text_phrase_lower = text_phrase.lower().strip()
    
    # First try exact match in each trie
    if text_phrase_lower in exact:
        info = exact[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Exact Match"
        }
    
    if text_phrase_lower in std:
        info = std[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Standardized Match"
        }
    
    if text_phrase_lower in lemma:
        info = lemma[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Lemma Match"
        }
    
    # If no direct match, try database lookup but ONLY return if it matches the text exactly
    try:
        # Try exact case-insensitive match first
        skill_obj = Skill.objects.filter(skill_name__iexact=text_phrase).first()
        if skill_obj:
            return {
                "name": skill_obj.skill_name,
                "definition": skill_obj.skill_definition or "Definition not available",
                "id": skill_obj.id,
                "classification": skill_obj.classification or "Unknown",
                "match_type": "Database Match"
            }
    except Exception as e:
        print(f"Error in database lookup: {e}")
    
    return None

def find_exact_skill_form_in_text(text, potential_skills):
    """
    Given a list of potential skill names, find which one EXACTLY matches text
    Returns the exact form that exists in the text, or Nones
    """
    import re
    text_lower = text.lower()
    
    for skill in potential_skills:
        skill_lower = skill.lower().strip()
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        if re.search(pattern, text_lower):
            return skill
    
    return None


def extract_skills_from_compound_phrases(text):
    """
    Extract skills from compound phrases like "copywriting & content writing"
    but DON'T break them down into individual parts unless they exist separately
    """
    compound_skills = []
    
    # Look for & patterns but be conservative
    ampersand_pattern = r'(\w+(?:\s+\w+)*)\s*&\s*(\w+(?:\s+\w+)*)'
    for match in re.finditer(ampersand_pattern, text, re.IGNORECASE):
        full_match = match.group(0).strip()
        start, end = match.span()
        
        # Only add the full compound skill, not the parts
        compound_skills.append((full_match, start, end))
        print(f"🔗 COMPOUND: Found '{full_match}' at {start}-{end}")
    
    return compound_skills

@lru_cache(maxsize=32768)
def find_matching_skills(phrase: str):
    phrase = phrase.lower()
    for trie in [exact, lemma]:
        if phrase in trie:
            info = trie[phrase]
            skill_name = info.get("name", phrase)
            skill_definition = info.get("definition", "No definition available")
            skill_id = info.get("id", None)
            return {
                "name": skill_name,
                "definition": skill_definition,
                "id": skill_id
            }
    return None

def remove_overlapping_skills(matched_skills_with_positions):
    """
    Remove overlapping skills, prioritizing longer/more specific matches.
    Input: List of tuples (skill_info, start_pos, end_pos, original_phrase)
    Output: List of non-overlapping skills
    """
    if not matched_skills_with_positions:
        return []
    
    # Sort by multiple criteria to prioritize the best matches
    sorted_matches = sorted(matched_skills_with_positions, 
                          key=lambda x: (
                              len(x[3].split()),  # Length of original phrase in words
                              len(x[0].get("name", "").split()),  # Length of skill name in words
                              bool(x[0].get("id")),  # Whether it has an ID
                              x[0].get("name", "").lower()  # Alphabetical as final tiebreaker
                          ), 
                          reverse=True)
    
    final_matches = []
    used_positions = set()
    
    for skill_info, start_pos, end_pos, original_phrase in sorted_matches:
        try:
            # Check if this match overlaps with any already selected match
            current_positions = set(range(start_pos, end_pos))
            
            # Allow partial overlap if the overlap is small relative to the match size
            overlap_size = len(current_positions.intersection(used_positions))
            match_size = end_pos - start_pos
            
            # If no overlap or minimal overlap (less than 25% of the match), include it
            if overlap_size == 0 or (overlap_size < match_size * 0.25):
                final_matches.append(skill_info)
                used_positions.update(current_positions)
                print(f"✅ KEPT: {skill_info.get('name', 'Unknown')} at {start_pos}-{end_pos}")
            else:
                print(f"🚫 OVERLAP: Removed {skill_info.get('name', 'Unknown')} due to overlap with existing match")
                
        except Exception as e:
            print(f"⚠️ Error processing overlap for skill {skill_info}: {e}")
            # Keep the skill anyway as fallback
            final_matches.append(skill_info)
    
    return final_matches

# ===== ORIGINAL SKILLS EXTRACTION API (Updated) =====
@csrf_exempt
def extract(request):
    if request.method == "POST":
        data = json.loads(request.body)
        text = data.get("text", "")

        tokens = tokenize_with_indices(text)
        segments = get_contiguous_segments(tokens)
        matcher = TrieMatcher(
        exact, std, lemma,
        is_std=False,
        qcw=qcw,
        classes=classes,
        full_text=text
)


        all_matches_with_positions = []
        max_ngram_size = 6

        def process_segment(segment):
            local_matches = []
            words = [tok.lower() for tok, _, _ in segment]
            n = len(words)
            
            all_matches_with_positions = []
            used_positions = {}
            matched_phrases = set()

            for segment in segments:
                matches = matcher.find_matches(segment, used_positions, matched_phrases)
                for m in matches:
                    all_matches_with_positions.append((
                        {
                            "name": m["skill"],
                            "definition": "",  # You’ll add real definition next
                            "id": m.get("id")
                        },
                        m["start"],
                        m["end"],
                        m["phrase"]
                    ))

            
            return local_matches

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_segment, seg) for seg in segments]
            for future in as_completed(futures):
                local_matches = future.result()
                all_matches_with_positions.extend(local_matches)

        # Remove overlapping matches
        final_skills = remove_overlapping_skills(all_matches_with_positions)
        for skill in final_skills:
            try:
                skill_obj = Skill.objects.filter(skill_name__iexact=skill["name"]).first()
                if skill_obj:
                    skill["definition"] = skill_obj.skill_definition
            except:
                pass

        
        # Sort by skill name
        final_skills.sort(key=lambda x: x["name"])

        return JsonResponse(final_skills, safe=False)
    return JsonResponse({"error": "Only POST allowed"}, status=400)

# ===== ENHANCED SKILLS EXTRACTION WITH DEMAND/RISK DATA (Updated) =====

def validate_skill_in_text(skill_name, original_text, start_pos=None, end_pos=None):
    """
    STRICT validation that a skill actually exists in the original text
    NO plural/singular conversions allowed - must match exactly what's in text
    """
    if not skill_name or not original_text:
        return False
    
    # If we have exact positions, validate the text at those positions
    if start_pos is not None and end_pos is not None:
        extracted_text = original_text[start_pos:end_pos].lower().strip()
        skill_lower = skill_name.lower().strip()
        
        # STRICT: Must be exact match
        if extracted_text == skill_lower:
            return True
            
        # DO NOT allow any plural/singular conversions
        # The skill name must match exactly what's in the text
        return False
    
    # If no exact positions, check if skill exists anywhere in text with EXACT matching
    skill_lower = skill_name.lower().strip()
    text_lower = original_text.lower()
    
    # Check for exact word boundary match ONLY
    import re
    pattern = r'\b' + re.escape(skill_lower) + r'\b'
    if re.search(pattern, text_lower):
        return True
    
    # DO NOT allow plural variations - be strict
    # If the database has "Software Developers" but text has "software developer"
    # then reject it - only extract what's actually in the text
    
    return False


def get_exact_skill_from_database(text_phrase, exact, std, lemma):
    """
    Get the skill from database that EXACTLY matches the text phrase
    Prioritizes exact form over variations
    """
    text_phrase_lower = text_phrase.lower().strip()
    
    # First try exact match in each trie
    if text_phrase_lower in exact:
        info = exact[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Exact Match"
        }
    
    if text_phrase_lower in std:
        info = std[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Standardized Match"
        }
    
    if text_phrase_lower in lemma:
        info = lemma[text_phrase_lower]
        skill_name = info.get("name", text_phrase) if isinstance(info, dict) else info
        return {
            "name": skill_name,
            "definition": info.get("definition", "Definition not available") if isinstance(info, dict) else "Definition not available",
            "id": info.get("id") if isinstance(info, dict) else None,
            "classification": info.get("classification", "Unknown") if isinstance(info, dict) else "Unknown",
            "match_type": "Lemma Match"
        }
    
    # If no direct match, try database lookup but ONLY return if it matches the text exactly
    try:
        # Try exact case-insensitive match first
        skill_obj = Skill.objects.filter(skill_name__iexact=text_phrase).first()
        if skill_obj:
            return {
                "name": skill_obj.skill_name,
                "definition": skill_obj.skill_definition or "Definition not available",
                "id": skill_obj.id,
                "classification": skill_obj.classification or "Unknown",
                "match_type": "Database Match"
            }
    except Exception as e:
        print(f"Error in database lookup: {e}")
    
    return None


@csrf_exempt
def extract_with_demand(request):
    """Enhanced skills extraction with strict validation and comprehensive error handling"""
    if request.method == "POST":
        start_time = time.time()
        
        try:
            data = json.loads(request.body)
            text = data.get("text", "")
            
            if not text.strip():
                return JsonResponse({"error": "No text provided"}, status=400)

            # Load skills from DB and build tries
            exact, std, lemma, qcw, classes = load_skills_from_db()
            print(f"🔍 Starting extraction on text length: {len(text)}")

            # Extract skills from parenthetical lists and special patterns
            parenthetical_skills = extract_parenthetical_skills(text)
            print(f"📝 Found {len(parenthetical_skills)} parenthetical/pattern skills")


            # Add parenthetical skills to the matches if they exist in our database
            verified_parenthetical_matches = []
            for skill_text, start, end in parenthetical_skills:
                skill_norm = skill_text.lower().strip()
                
                # Get the EXACT text from the original document at these positions
                actual_text_in_document = text[start:end].lower().strip()
                
                # 🆕 HANDLE & SPLITTING HERE - Split compound skills immediately
                if '&' in skill_norm:
                    parts = [part.strip() for part in skill_norm.split('&')]
                    print(f"🔄 SPLITTING COMPOUND SKILL: '{skill_text}' -> {parts}")
                    
                    for part in parts:
                        if len(part) > 1:  # Skip empty parts
                            # STRICT: Only add if this exact part exists in database AND matches text
                            skill_info = get_exact_skill_from_database(part, exact, std, lemma)
                            
                            if skill_info:
                                # VALIDATE: Does this skill name exactly match what's in the text?
                                if validate_skill_in_text(skill_info["name"], text, start, end):
                                    verified_parenthetical_matches.append((skill_info, start, end, part))
                                    print(f"✅ COMPOUND PART VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')}) from part '{part}'")
                                else:
                                    print(f"❌ COMPOUND PART VALIDATION FAILED: '{skill_info['name']}' doesn't exactly match text")
                            else:
                                print(f"🚫 COMPOUND PART SKIPPED: '{part}' not found in database")
                else:
                    # Handle non-compound skills with STRICT matching
                    # STRICT: Get exact skill that matches the text
                    skill_info = get_exact_skill_from_database(skill_text, exact, std, lemma)
                    
                    if skill_info:
                        # STRICT VALIDATION: Must exactly match what's in the text
                        if validate_skill_in_text(skill_info["name"], text, start, end):
                            verified_parenthetical_matches.append((skill_info, start, end, skill_text))
                            print(f"✅ SINGLE SKILL VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')}) in database")
                        else:
                            print(f"❌ SINGLE SKILL VALIDATION FAILED: '{skill_info['name']}' doesn't exactly match text '{actual_text_in_document}'")
                    else:
                        print(f"🚫 SINGLE SKILL SKIPPED: '{skill_text}' not found in database")

            # Continue with regular tokenization and matching
            tokens = tokenize_with_indices(text)
            segments = get_contiguous_segments(tokens)

            matcher = TrieMatcher(
                exact=exact, std=std, lemma=lemma, is_std=False,
                qcw=qcw, classes=classes, full_text=text
            )

            all_matches_with_positions = []
            used_positions = {}
            matched_phrases = set()

            for segment in segments:
                matches = matcher.find_matches(segment, used_positions, matched_phrases)
                for m in matches:
                    start_pos = m["start"]
                    end_pos = m["end"]
                    actual_text_span = text[start_pos:end_pos].lower().strip()
                    
                    skill_name = m["skill"]
                    
                    # STRICT VALIDATION: Must exactly match what's in the text
                    if not validate_skill_in_text(skill_name, text, start_pos, end_pos):
                        print(f"❌ REJECTED REGULAR MATCH: '{skill_name}' - doesn't exactly match text at {start_pos}-{end_pos}")
                        print(f"   Text at position: '{text[start_pos:end_pos]}'")
                        continue
                    
                    # Handle compound skills but with STRICT validation
                    if '&' in skill_name:
                        parts = [part.strip() for part in skill_name.split('&')]
                        print(f"🔄 SPLITTING REGULAR COMPOUND SKILL: '{skill_name}' -> {parts}")
                        
                        # First add the complete compound skill if it exactly matches
                        if validate_skill_in_text(skill_name, text, start_pos, end_pos):
                            skill_info_dict = {
                                "name": skill_name,
                                "definition": "Definition not available",
                                "id": m.get("id"),
                                "classification": m.get("classification", "Unknown")
                            }
                            all_matches_with_positions.append((
                                skill_info_dict,
                                m["start"], m["end"], skill_name
                            ))
                        
                        # DON'T add individual parts - this causes the plural issue
                        # Only add parts if they exist as separate skills in the text
                        
                    else:
                        # Handle non-compound skills with database lookup for exact form
                        exact_skill_info = get_exact_skill_from_database(actual_text_span, exact, std, lemma)
                        
                        if exact_skill_info and validate_skill_in_text(exact_skill_info["name"], text, start_pos, end_pos):
                            all_matches_with_positions.append((
                                exact_skill_info,
                                m["start"], m["end"], actual_text_span
                            ))
                            print(f"✅ VALIDATED REGULAR: Added '{exact_skill_info['name']}' (ID: {exact_skill_info.get('id')})")
                        else:
                            print(f"❌ REJECTED: No exact database match for '{actual_text_span}' or validation failed")

            # Combine regular matches with validated parenthetical matches
            all_matches_with_positions.extend(verified_parenthetical_matches)
            print(f"🔄 Total matches before deduplication: {len(all_matches_with_positions)}")

            # Remove overlapping matches
            final_skills = remove_overlapping_skills(all_matches_with_positions)
            
            # FINAL VALIDATION - Remove any skills that don't actually exist in text
            truly_final_skills = []
            for skill in final_skills:
                skill_name = skill.get("name", "")
                if validate_skill_in_text(skill_name, text):
                    truly_final_skills.append(skill)
                    print(f"✅ FINAL VALIDATION PASSED: '{skill_name}'")
                else:
                    print(f"❌ FINAL VALIDATION FAILED: '{skill_name}' - removing from results")
            
            final_skills = truly_final_skills
            print(f"✅ Final skills after validation: {len(final_skills)}")

            # Add definitions from DB and ensure all fields are present
            for skill in final_skills:
                try:
                    if "definition" not in skill or skill["definition"] == "Definition not available":
                        skill_obj = Skill.objects.filter(skill_name__iexact=skill["name"]).first()
                        if not skill_obj:
                            skill_obj = Skill.objects.filter(skill_name__icontains=skill["name"]).first()
                        
                        if skill_obj:
                            skill["definition"] = skill_obj.skill_definition or "Definition not available"
                            skill["classification"] = skill_obj.classification or skill.get("classification", "Unknown")
                            if not skill.get("id"):
                                skill["id"] = skill_obj.id

                    # Ensure all required fields
                    skill.setdefault("definition", "Definition not available")
                    skill.setdefault("classification", "Unknown")
                    skill.setdefault("id", None)
                    skill.setdefault("name", "Unknown Skill")
                            
                except Exception as e:
                    print(f"⚠️ Error fetching skill data for '{skill.get('name', 'Unknown')}': {e}")
                    skill.setdefault("definition", "Definition not available")
                    skill.setdefault("classification", "Unknown")
                    skill.setdefault("id", None)
                    skill.setdefault("name", "Unknown Skill")

            # Attach demand and risk metadata
            enhanced_skills = []
            for skill in final_skills:
                try:
                    skill_data = {
                        "id": skill.get("id"),
                        "name": skill.get("name", "Unknown Skill"),
                        "definition": skill.get("definition", "Definition not available"),
                        "classification": skill.get("classification", "Unknown"),
                        "demand": None, 
                        "demand_rationale": None,
                        "risk": None, 
                        "risk_rationale": None
                    }

                    # Try to add demand data
                    try:
                        demand_data = SkillsDemand.objects.filter(skill__iexact=skill_data["name"]).first()
                        if demand_data:
                            skill_data.update({
                                "demand": demand_data.demand,
                                "demand_rationale": demand_data.demand_rationale,
                                "risk": demand_data.risk,
                                "risk_rationale": demand_data.risk_rationale
                            })
                    except Exception as demand_error:
                        print(f"⚠️ Demand data error for '{skill_data['name']}': {demand_error}")

                    enhanced_skills.append(skill_data)
                    
                except Exception as e:
                    print(f"⚠️ Error creating skill data for {skill}: {e}")
                    fallback_skill = {
                        "id": None,
                        "name": str(skill) if not isinstance(skill, dict) else skill.get("name", "Unknown Skill"),
                        "definition": "Definition not available",
                        "classification": "Unknown",
                        "demand": None, "demand_rationale": None,
                        "risk": None, "risk_rationale": None
                    }
                    enhanced_skills.append(fallback_skill)

            # Remove duplicates based on skill name and ID
            seen_skills = set()
            deduplicated_skills = []
            
            for skill in enhanced_skills:
                try:
                    skill_identifier = (skill.get("name", "").lower(), skill.get("id"))
                    
                    if skill_identifier not in seen_skills:
                        seen_skills.add(skill_identifier)
                        deduplicated_skills.append(skill)
                    else:
                        print(f"🧹 REMOVED DUPLICATE: {skill.get('name', 'Unknown')} (ID: {skill.get('id', 'None')})")
                except Exception as dedup_error:
                    print(f"⚠️ Error deduplicating skill {skill}: {dedup_error}")
                    deduplicated_skills.append(skill)
            
            enhanced_skills = deduplicated_skills

            # Group skills by classification
            classification_mapping = {
                "Highly Specialized Technical Skills": [],
                "Core Job Skills": [],
                "Industry-Specific Job Skills": [],
                "General Professional Skills (Soft Skills)": [],
                "Specialized Industry Concepts": [],
                "Other": []
            }
            
            for skill in enhanced_skills:
                try:
                    classification = skill.get("classification", "Unknown")
                    if classification in classification_mapping:
                        classification_mapping[classification].append(skill)
                    else:
                        classification_mapping["Other"].append(skill)
                except Exception as classification_error:
                    print(f"⚠️ Error classifying skill {skill}: {classification_error}")
                    classification_mapping["Other"].append(skill)

            # Remove empty classifications
            filtered_classifications = {k: v for k, v in classification_mapping.items() if v}

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000

            print(f"🎉 Extraction completed in {processing_time:.2f}ms")
            print(f"📊 FINAL SKILL COUNT: {len(enhanced_skills)} skills total")

            return JsonResponse({
                "skills": filtered_classifications,
                "total_count": len(enhanced_skills),
                "processing_time_ms": round(processing_time, 2)
            }, safe=False)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            traceback_str = traceback.format_exc()
            print("EXTRACT ERROR TRACEBACK:\n", traceback_str)
            return JsonResponse({"error": f"Processing error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)



# ===== ALTERNATIVE IMPLEMENTATION USING WORD-BASED OVERLAP =====
@csrf_exempt
def extract_with_demand_v2(request):
    """Alternative implementation using word-based overlap detection"""
    if request.method == "POST":
        start_time = time.time()
        
        try:
            data = json.loads(request.body)
            text = data.get("text", "")
            
            if not text.strip():
                return JsonResponse({"error": "No text provided"}, status=400)

            tokens = tokenize_with_indices(text)
            segments = get_contiguous_segments(tokens)

            matched_skills_with_phrases = []
            max_ngram_size = 6

            def process_segment(segment):
                local_matches = []
                words = [tok.lower() for tok, _, _ in segment]
                n = len(words)
                
                for i in range(n):
                    for j in range(i + 1, min(i + max_ngram_size + 1, n + 1)):
                        phrase = " ".join(words[i:j])
                        match = find_matching_skills(phrase)
                        if match:
                            local_matches.append((match, phrase))
                
                return local_matches

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_segment, seg) for seg in segments]
                for future in as_completed(futures):
                    local_matches = future.result()
                    matched_skills_with_phrases.extend(local_matches)

            # Remove overlapping matches using word-based approach
            final_skills = remove_overlapping_skills_by_words(matched_skills_with_phrases)
            
            # Sort by skill name
            final_skills.sort(key=lambda x: x["name"])

            # Enhance skills with demand and risk data
            enhanced_skills = []
            for skill in final_skills:
                skill_data = {
                    "name": skill["name"],
                    "definition": skill["definition"],
                    "id": skill["id"],
                    "demand": None,
                    "demand_rationale": None,
                    "risk": None,
                    "risk_rationale": None
                }
                
                # Look up demand data
                try:
                    demand_data = SkillsDemand.objects.filter(skill__iexact=skill["name"]).first()
                    if demand_data:
                        skill_data.update({
                            "demand": demand_data.demand,
                            "demand_rationale": demand_data.demand_rationale,
                            "risk": demand_data.risk,
                            "risk_rationale": demand_data.risk_rationale
                        })
                except Exception as e:
                    print(f"Error looking up demand data for {skill['name']}: {e}")
                    pass
                
                enhanced_skills.append(skill_data)

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            return JsonResponse({
                "skills": enhanced_skills,
                "total_count": len(enhanced_skills),
                "processing_time_ms": round(processing_time, 2)
            }, safe=False)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Processing error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)

# ===== REST OF THE APIS (unchanged) =====
@csrf_exempt
def skills_demand_search(request):
    """Search skills demand data"""
    if request.method == "GET":
        skill_name = request.GET.get('skill', '').strip()
        demand = request.GET.get('demand', '').strip()
        risk = request.GET.get('risk', '').strip()
        limit = min(int(request.GET.get('limit', 50)), 100)
        
        try:
            queryset = SkillsDemand.objects.all()
            
            if skill_name:
                queryset = queryset.filter(skill__icontains=skill_name)
            
            if demand:
                queryset = queryset.filter(demand__icontains=demand)
            
            if risk:
                queryset = queryset.filter(risk__icontains=risk)
            
            queryset = queryset.order_by('skill')[:limit]
            
            results = []
            for record in queryset:
                results.append({
                    "id": str(record.id),
                    "skill": record.skill,
                    "demand": record.demand,
                    "demand_rationale": record.demand_rationale,
                    "risk": record.risk,
                    "risk_rationale": record.risk_rationale,
                    "created_at": record.created_at.isoformat()
                })
            
            return JsonResponse({
                "skills_demand": results,
                "total_count": len(results),
                "filters": {
                    "skill": skill_name,
                    "demand": demand,
                    "risk": risk
                }
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Search error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def skill_demand_by_name(request, skill_name):
    """Get demand data for a specific skill"""
    if request.method == "GET":
        try:
            record = SkillsDemand.objects.filter(skill__iexact=skill_name).first()
            
            if not record:
                return JsonResponse({"error": "Skill not found"}, status=404)
            
            result = {
                "id": str(record.id),
                "skill": record.skill,
                "demand": record.demand,
                "demand_rationale": record.demand_rationale,
                "risk": record.risk,
                "risk_rationale": record.risk_rationale,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat()
            }
            
            return JsonResponse({
                "skill_name": skill_name,
                "demand_data": result
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def all_skills_demand(request):
    """Get all skills demand data"""
    if request.method == "GET":
        try:
            records = SkillsDemand.objects.all().order_by('skill')
            
            results = []
            for record in records:
                results.append({
                    "skill": record.skill,
                    "demand": record.demand,
                    "demand_rationale": record.demand_rationale,
                    "risk": record.risk,
                    "risk_rationale": record.risk_rationale
                })
            
            return JsonResponse({
                "skills_demand": results,
                "total_count": len(results)
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

# ===== FOLDER MANAGEMENT APIs =====
@csrf_exempt
def create_folder(request):
    """Create a new folder"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            folder_name = data.get("name", "").strip()
            
            if not folder_name:
                return JsonResponse({"error": "Folder name is required"}, status=400)
            
            if Folder.objects.filter(name=folder_name).exists():
                return JsonResponse({"error": "Folder with this name already exists"}, status=400)
            
            folder = Folder.objects.create(name=folder_name)
            
            return JsonResponse({
                "success": True,
                "folder": {
                    "id": folder.id,
                    "name": folder.name,
                    "created_at": folder.created_at.isoformat()
                }
            }, status=201)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Error creating folder: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def get_folders(request):
    """Get all folders"""
    if request.method == "GET":
        try:
            folders = Folder.objects.all().order_by('name')
            
            folders_data = []
            for folder in folders:
                folders_data.append({
                    "id": folder.id,
                    "name": folder.name,
                    "files_count": folder.files.count(),
                    "created_at": folder.created_at.isoformat(),
                    "updated_at": folder.updated_at.isoformat()
                })
            
            return JsonResponse({
                "folders": folders_data,
                "total_count": len(folders_data)
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Error fetching folders: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def get_folder_by_id(request, folder_id):
    """Get a specific folder by ID"""
    if request.method == "GET":
        try:
            folder = Folder.objects.get(id=folder_id)
            
            files = folder.files.all().order_by('filename')
            files_data = []
            for file in files:
                files_data.append({
                    "id": file.id,
                    "filename": file.filename,
                    "has_skill_profile": bool(file.skill_profile),
                    "created_at": file.created_at.isoformat()
                })
            
            return JsonResponse({
                "folder": {
                    "id": folder.id,
                    "name": folder.name,
                    "created_at": folder.created_at.isoformat(),
                    "updated_at": folder.updated_at.isoformat()
                },
                "files": files_data,
                "files_count": len(files_data)
            })
            
        except Folder.DoesNotExist:
            return JsonResponse({"error": "Folder not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def delete_folder(request, folder_id):
    """Delete a folder and all its files"""
    if request.method == "DELETE":
        try:
            folder = Folder.objects.get(id=folder_id)
            folder_name = folder.name
            files_count = folder.files.count()
            
            folder.delete()
            
            return JsonResponse({
                "success": True,
                "message": f"Folder '{folder_name}' and {files_count} files deleted successfully"
            })
            
        except Folder.DoesNotExist:
            return JsonResponse({"error": "Folder not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error deleting folder: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only DELETE allowed"}, status=405)

# ===== FILE MANAGEMENT APIs =====
@csrf_exempt
def create_file(request):
    """Create a new file with skill profile"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            filename = data.get("filename", "").strip()
            folder_id = data.get("folder_id")
            skill_profile = data.get("skill_profile", {})
            
            if not filename:
                return JsonResponse({"error": "Filename is required"}, status=400)
            
            if not folder_id:
                return JsonResponse({"error": "Folder ID is required"}, status=400)
            
            try:
                folder = Folder.objects.get(id=folder_id)
            except Folder.DoesNotExist:
                return JsonResponse({"error": "Folder not found"}, status=404)
            
            if File.objects.filter(filename=filename, folder=folder).exists():
                return JsonResponse({"error": "File with this name already exists in this folder"}, status=400)
            
            file = File.objects.create(
                filename=filename,
                folder=folder,
                skill_profile=skill_profile
            )
            
            return JsonResponse({
                "success": True,
                "file": {
                    "id": file.id,
                    "filename": file.filename,
                    "folder_id": file.folder.id,
                    "folder_name": file.folder.name,
                    "skill_profile": file.skill_profile,
                    "full_path": file.full_path,
                    "created_at": file.created_at.isoformat()
                }
            }, status=201)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Error creating file: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def update_file(request, file_id):
    """Update file data including skill profile"""
    if request.method == "PUT":
        try:
            data = json.loads(request.body)
            
            try:
                file = File.objects.get(id=file_id)
            except File.DoesNotExist:
                return JsonResponse({"error": "File not found"}, status=404)
            
            if "filename" in data:
                new_filename = data["filename"].strip()
                if new_filename:
                    if File.objects.filter(filename=new_filename, folder=file.folder).exclude(id=file_id).exists():
                        return JsonResponse({"error": "File with this name already exists in this folder"}, status=400)
                    file.filename = new_filename
            
            if "folder_id" in data:
                try:
                    new_folder = Folder.objects.get(id=data["folder_id"])
                    if File.objects.filter(filename=file.filename, folder=new_folder).exclude(id=file_id).exists():
                        return JsonResponse({"error": "File with this name already exists in the target folder"}, status=400)
                    file.folder = new_folder
                except Folder.DoesNotExist:
                    return JsonResponse({"error": "Target folder not found"}, status=404)
            
            if "skill_profile" in data:
                if isinstance(data["skill_profile"], dict):
                    file.skill_profile.update(data["skill_profile"])
                else:
                    file.skill_profile = data["skill_profile"]
            
            file.save()
            
            return JsonResponse({
                "success": True,
                "file": {
                    "id": file.id,
                    "filename": file.filename,
                    "folder_id": file.folder.id,
                    "folder_name": file.folder.name,
                    "skill_profile": file.skill_profile,
                    "full_path": file.full_path,
                    "updated_at": file.updated_at.isoformat()
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Error updating file: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only PUT allowed"}, status=405)

@csrf_exempt
def get_file_by_id(request, file_id):
    """Get a specific file by ID"""
    if request.method == "GET":
        try:
            file = File.objects.select_related('folder').get(id=file_id)
            
            return JsonResponse({
                "file": {
                    "id": file.id,
                    "filename": file.filename,
                    "folder_id": file.folder.id,
                    "folder_name": file.folder.name,
                    "skill_profile": file.skill_profile,
                    "full_path": file.full_path,
                    "created_at": file.created_at.isoformat(),
                    "updated_at": file.updated_at.isoformat()
                }
            })
            
        except File.DoesNotExist:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def get_files(request):
    """Get all files with optional filtering"""
    if request.method == "GET":
        try:
            folder_id = request.GET.get('folder_id')
            limit = min(int(request.GET.get('limit', 100)), 500)
            
            queryset = File.objects.select_related('folder').all()
            
            if folder_id:
                queryset = queryset.filter(folder_id=folder_id)
            
            files = queryset.order_by('folder__name', 'filename')[:limit]
            
            files_data = []
            for file in files:
                files_data.append({
                    "id": file.id,
                    "filename": file.filename,
                    "folder_id": file.folder.id,
                    "folder_name": file.folder.name,
                    "has_skill_profile": bool(file.skill_profile),
                    "full_path": file.full_path,
                    "created_at": file.created_at.isoformat()
                })
            
            return JsonResponse({
                "files": files_data,
                "total_count": len(files_data),
                "filters": {
                    "folder_id": folder_id
                }
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Error fetching files: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def delete_file(request, file_id):
    """Delete a specific file"""
    if request.method == "DELETE":
        try:
            file = File.objects.get(id=file_id)
            filename = file.filename
            folder_name = file.folder.name
            
            file.delete()
            
            return JsonResponse({
                "success": True,
                "message": f"File '{filename}' from folder '{folder_name}' deleted successfully"
            })
            
        except File.DoesNotExist:
            return JsonResponse({"error": "File not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error deleting file: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only DELETE allowed"}, status=405)


@csrf_exempt
def skill_suggestions(request):
    """Get skill suggestions with demand/risk data for autocomplete"""
    if request.method == "GET":
        try:
            query = request.GET.get('q', '').strip()
            limit = min(int(request.GET.get('limit', 10)), 50)
            
            if not query or len(query) < 2:
                return JsonResponse({
                    "suggestions": [],
                    "message": "Query too short. Minimum 2 characters required."
                })
            
            # Search skills by name (case-insensitive, contains)
            skills_queryset = Skill.objects.filter(
                skill_name__icontains=query
            ).order_by('skill_name')[:limit]
            
            suggestions = []
            for skill in skills_queryset:
                skill_data = {
                    "id": skill.id,
                    "name": skill.skill_name,
                    "definition": skill.skill_definition or "Definition not available",
                    "classification": skill.classification or "Unknown",
                    "demand": None,
                    "demand_rationale": None,
                    "risk": None,
                    "risk_rationale": None
                }
                
                # Try to get demand/risk data
                try:
                    demand_data = SkillsDemand.objects.filter(skill__iexact=skill.skill_name).first()
                    if demand_data:
                        skill_data.update({
                            "demand": demand_data.demand,
                            "demand_rationale": demand_data.demand_rationale,
                            "risk": demand_data.risk,
                            "risk_rationale": demand_data.risk_rationale
                        })
                except Exception as e:
                    print(f"Error fetching demand data for {skill.skill_name}: {e}")
                
                suggestions.append(skill_data)
            
            return JsonResponse({
                "suggestions": suggestions,
                "query": query,
                "total_count": len(suggestions)
            })
            
        except Exception as e:
            return JsonResponse({"error": f"Search error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)


# Add these functions to your extractor/views.py

@csrf_exempt
def get_all_folders_with_analytics(request):
    """Get all folders with aggregated skill analytics"""
    if request.method == "GET":
        try:
            folders = Folder.objects.prefetch_related('files').all().order_by('name')
            
            result = {}
            
            for folder in folders:
                folder_data = {}
                
                for file in folder.files.all():
                    # Initialize all possible categories with 0 counts
                    file_analytics = {
                        "competencyLevel": {
                            "Skill Gap": 0,
                            "Upskilling Needed": 0,
                            "Skilled": 0
                        },
                        "classification": {
                            "Highly Specialized Technical Skills": 0,
                            "Core Job Skills": 0,
                            "Industry-Specific Job Skills": 0,
                            "General Professional Skills (Soft Skills)": 0,
                            "Specialized Industry Concepts": 0,
                            "Other": 0
                        },
                        "marketDemand": {
                            "Critical Shortage": 0,
                            "High Demand": 0,
                            "Stable Demand": 0,
                            "Low Demand": 0
                        },
                        "obsolescenceRisk": {
                            "High Risk": 0,
                            "Medium Risk": 0,
                            "Low Risk": 0
                        }
                    }
                    
                    # Parse skill_profile and count occurrences
                    if file.skill_profile and 'skills' in file.skill_profile:
                        skills = file.skill_profile['skills']
                        
                        for skill in skills:
                            # Count competencyLevel
                            competency = skill.get('competencyLevel', 'Unknown')
                            if competency in file_analytics['competencyLevel']:
                                file_analytics['competencyLevel'][competency] += 1
                            
                            # Count classification
                            classification = skill.get('classification', 'Unknown')
                            if classification in file_analytics['classification']:
                                file_analytics['classification'][classification] += 1
                            else:
                                file_analytics['classification']['Other'] += 1
                            
                            # Count marketDemand
                            market_demand = skill.get('marketDemand')
                            if market_demand and market_demand in file_analytics['marketDemand']:
                                file_analytics['marketDemand'][market_demand] += 1
                            
                            # Count obsolescenceRisk - handle different variations
                            risk = skill.get('obsolescenceRisk')
                            if risk:
                                if risk == 'Low':
                                    file_analytics['obsolescenceRisk']['Low Risk'] += 1
                                elif risk == 'Medium':
                                    file_analytics['obsolescenceRisk']['Medium Risk'] += 1
                                elif risk == 'High':
                                    file_analytics['obsolescenceRisk']['High Risk'] += 1
                                elif risk in file_analytics['obsolescenceRisk']:
                                    file_analytics['obsolescenceRisk'][risk] += 1
                    
                    # Only add file data if it has skills (check if any count > 0)
                    has_skills = (file.skill_profile and 'skills' in file.skill_profile and 
                                 len(file.skill_profile['skills']) > 0)
                    if has_skills:
                        folder_data[file.filename] = file_analytics
                
                # Only add folder data if it has files with skills
                if folder_data:
                    result[folder.name] = folder_data
            
            result = {folder.name: folder_data}
            
            return JsonResponse({"data": result}, safe=False)
            
        except Exception as e:
            return JsonResponse({"error": f"Error fetching analytics: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)


@csrf_exempt
def get_folder_analytics(request, folder_id):
    """Get analytics for a specific folder"""
    if request.method == "GET":
        try:
            folder = Folder.objects.prefetch_related('files').get(id=folder_id)
            
            folder_data = {}
            
            for file in folder.files.all():
                # Initialize all possible categories with 0 counts
                file_analytics = {
                    "competencyLevel": {
                        "Skill Gap": 0,
                        "Upskilling Needed": 0,
                        "Skilled": 0
                    },
                    "classification": {
                        "Highly Specialized Technical Skills": 0,
                        "Core Job Skills": 0,
                        "Industry-Specific Job Skills": 0,
                        "General Professional Skills (Soft Skills)": 0,
                        "Specialized Industry Concepts": 0,
                        "Other": 0
                    },
                    "marketDemand": {
                        "Critical Shortage": 0,
                        "High Demand": 0,
                        "Stable Demand": 0,
                        "Low Demand": 0
                    },
                    "obsolescenceRisk": {
                        "High Risk": 0,
                        "Medium Risk": 0,
                        "Low Risk": 0
                    }
                }
                
                # Parse skill_profile and count occurrences
                if file.skill_profile and 'skills' in file.skill_profile:
                    skills = file.skill_profile['skills']
                    
                    for skill in skills:
                        # Count competencyLevel
                        competency = skill.get('competencyLevel', 'Unknown')
                        if competency in file_analytics['competencyLevel']:
                            file_analytics['competencyLevel'][competency] += 1
                        
                        # Count classification
                        classification = skill.get('classification', 'Unknown')
                        if classification in file_analytics['classification']:
                            file_analytics['classification'][classification] += 1
                        else:
                            file_analytics['classification']['Other'] += 1
                        
                        # Count marketDemand
                        market_demand = skill.get('marketDemand')
                        if market_demand and market_demand in file_analytics['marketDemand']:
                            file_analytics['marketDemand'][market_demand] += 1
                        
                        # Count obsolescenceRisk - handle different variations
                        risk = skill.get('obsolescenceRisk')
                        if risk:
                            if risk == 'Low':
                                file_analytics['obsolescenceRisk']['Low Risk'] += 1
                            elif risk == 'Medium':
                                file_analytics['obsolescenceRisk']['Medium Risk'] += 1
                            elif risk == 'High':
                                file_analytics['obsolescenceRisk']['High Risk'] += 1
                            elif risk in file_analytics['obsolescenceRisk']:
                                file_analytics['obsolescenceRisk'][risk] += 1
                
                # Only add file data if it has skills (check if any count > 0)
                has_skills = (file.skill_profile and 'skills' in file.skill_profile and 
                             len(file.skill_profile['skills']) > 0)
                if has_skills:
                    folder_data[file.filename] = file_analytics
            
            return JsonResponse({"data": result}, safe=False)
            
        except Folder.DoesNotExist:
            return JsonResponse({"error": "Folder not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error fetching analytics: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)


# Alternative version that provides more detailed analytics with metadata
@csrf_exempt
def get_detailed_folder_analytics(request, folder_id=None):
    """Get detailed analytics with metadata"""
    if request.method == "GET":
        try:
            if folder_id:
                folders = [Folder.objects.prefetch_related('files').get(id=folder_id)]
            else:
                folders = Folder.objects.prefetch_related('files').all().order_by('name')
            
            result = {}
            total_files = 0
            total_skills = 0
            
            for folder in folders:
                folder_data = {}
                folder_files_count = 0
                folder_skills_count = 0
                
                for file in folder.files.all():
                    # Initialize all possible categories with 0 counts
                    file_analytics = {
                        "competencyLevel": {
                            "Skill Gap": 0,
                            "Upskilling Needed": 0,
                            "Skilled": 0
                        },
                        "classification": {
                            "Highly Specialized Technical Skills": 0,
                            "Core Job Skills": 0,
                            "Industry-Specific Job Skills": 0,
                            "General Professional Skills (Soft Skills)": 0,
                            "Specialized Industry Concepts": 0,
                            "Other": 0
                        },
                        "marketDemand": {
                            "Critical Shortage": 0,
                            "High Demand": 0,
                            "Stable Demand": 0,
                            "Low Demand": 0
                        },
                        "obsolescenceRisk": {
                            "High Risk": 0,
                            "Medium Risk": 0,
                            "Low Risk": 0
                        },
                        "metadata": {
                            "totalSkills": 0,
                            "lastUpdated": file.updated_at.isoformat() if file.updated_at else None
                        }
                    }
                    
                    # Parse skill_profile and count occurrences
                    if file.skill_profile and 'skills' in file.skill_profile:
                        skills = file.skill_profile['skills']
                        file_analytics["metadata"]["totalSkills"] = len(skills)
                        folder_skills_count += len(skills)
                        
                        for skill in skills:
                            # Count competencyLevel
                            competency = skill.get('competencyLevel', 'Unknown')
                            if competency in file_analytics['competencyLevel']:
                                file_analytics['competencyLevel'][competency] += 1
                            
                            # Count classification
                            classification = skill.get('classification', 'Unknown')
                            if classification in file_analytics['classification']:
                                file_analytics['classification'][classification] += 1
                            else:
                                file_analytics['classification']['Other'] += 1
                            
                            # Count marketDemand
                            market_demand = skill.get('marketDemand')
                            if market_demand and market_demand in file_analytics['marketDemand']:
                                file_analytics['marketDemand'][market_demand] += 1
                            
                            # Count obsolescenceRisk - handle different variations
                            risk = skill.get('obsolescenceRisk')
                            if risk:
                                if risk == 'Low':
                                    file_analytics['obsolescenceRisk']['Low Risk'] += 1
                                elif risk == 'Medium':
                                    file_analytics['obsolescenceRisk']['Medium Risk'] += 1
                                elif risk == 'High':
                                    file_analytics['obsolescenceRisk']['High Risk'] += 1
                                elif risk in file_analytics['obsolescenceRisk']:
                                    file_analytics['obsolescenceRisk'][risk] += 1
                    
                    # Only add file data if it has skills
                    if file_analytics["metadata"]["totalSkills"] > 0:
                        folder_data[file.filename] = file_analytics
                        folder_files_count += 1
                
                # Add folder metadata
                if folder_data:
                    result[folder.name] = {
                        "files": folder_data,
                        "metadata": {
                            "totalFiles": folder_files_count,
                            "totalSkills": folder_skills_count,
                            "folderCreated": folder.created_at.isoformat() if folder.created_at else None
                        }
                    }
                    total_files += folder_files_count
                    total_skills += folder_skills_count
            
            # Return with overall metadata wrapped in data object
            response_data = {
                "data": {
                    "folders": result,
                    "overallMetadata": {
                        "totalFolders": len([f for f in result.keys()]),
                        "totalFiles": total_files,
                        "totalSkills": total_skills,
                        "generatedAt": time.timezone
                    }
                }
            }
            
            return JsonResponse(response_data, safe=False)
            
        except Folder.DoesNotExist:
            return JsonResponse({"error": "Folder not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": f"Error fetching detailed analytics: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

# Add these APIs to your extractor/views.py file

from extractor.models import SkillsJobTitles, SkillJobFunction
from django.core.paginator import Paginator
from django.db.models import Q, Count

# SKILLS BUILDER

@csrf_exempt
def get_skills_job_titles(request):
    """Enhanced skills search with predictive search and multiple selections"""
    if request.method == "GET":
        try:
            # Get query parameters
            search = request.GET.get('search', '').strip()
            search_mode = request.GET.get('search_mode', 'job_titles')  # 'job_titles' or 'job_functions'
            relevance_level = int(request.GET.get('relevance_level', 50))  # 0-100 scale
            page = int(request.GET.get('page', 1))
            page_size = min(int(request.GET.get('page_size', 20)), 100)
            
            # Multiple selection filters - Updated to handle array parameters correctly
            selected_industries = request.GET.getlist('industries[]') or request.GET.getlist('industries')
            selected_job_functions = request.GET.getlist('job_functions[]') or request.GET.getlist('job_functions') 
            selected_skills = request.GET.getlist('skills[]') or request.GET.getlist('skills')
            
            # Additional filters
            skill_category = request.GET.get('skill_category', '').strip()
            demand_levels = request.GET.getlist('demand_level')
            risk_levels = request.GET.getlist('risk_level')
            classification = request.GET.get('classification', '').strip()
            
            print(f"🔍 Search: '{search}', Mode: {search_mode}, Relevance: {relevance_level}")
            print(f"🏭 Selected Industries: {selected_industries}")
            print(f"💼 Selected Job Functions: {selected_job_functions}")
            print(f"🎯 Selected Skills: {selected_skills}")
            
            # Validate search mode
            if search_mode not in ['job_titles', 'job_functions']:
                return JsonResponse({"error": "Invalid search_mode. Use 'job_titles' or 'job_functions'"}, status=400)
            
            # Start with all skills
            queryset = SkillsJobTitles.objects.all()
            
            # Apply basic filters first
            if skill_category:
                queryset = queryset.filter(skill_category__icontains=skill_category)
            
            # FIXED: Core search logic - find skills based on job titles or job functions
            matching_skill_names = set()
            
            if search:
                search_lower = search.lower()
                
                if search_mode == 'job_titles':
                    # FIXED: Search in job titles based on relevance level
                    # Determine which job title levels to search based on relevance
                    if relevance_level >= 81:
                        max_levels = 1
                    elif relevance_level >= 61:
                        max_levels = 2
                    elif relevance_level >= 41:
                        max_levels = 3
                    elif relevance_level >= 21:
                        max_levels = 4
                    else:
                        max_levels = 5
                    
                    # Build Q objects for job title search
                    q_objects = Q()
                    
                    # Always include skill name search
                    q_objects |= Q(skill__icontains=search)
                    
                    # Add job title searches based on relevance level
                    if max_levels >= 1:
                        q_objects |= Q(job_title_1__icontains=search)
                    if max_levels >= 2:
                        q_objects |= Q(job_title_2__icontains=search)
                    if max_levels >= 3:
                        q_objects |= Q(job_title_3__icontains=search)
                    if max_levels >= 4:
                        q_objects |= Q(job_title_4__icontains=search)
                    if max_levels >= 5:
                        q_objects |= Q(job_title_5__icontains=search)
                    
                    # Get all skills that match the search in job titles
                    matching_skills = SkillsJobTitles.objects.filter(q_objects).values_list('skill', flat=True)
                    matching_skill_names.update(matching_skills)
                    
                elif search_mode == 'job_functions':
                    # FIXED: Search in job functions with relevance level
                    # First, search in skill names from SkillsJobTitles
                    skill_name_matches = SkillsJobTitles.objects.filter(
                        skill__icontains=search
                    ).values_list('skill', flat=True)
                    matching_skill_names.update(skill_name_matches)
                    
                    # Then search in SkillJobFunction table based on relevance
                    q_objects = Q()
                    
                    # Always search in skill name
                    q_objects |= Q(skill__icontains=search)
                    
                    # Add fields based on relevance level
                    if relevance_level >= 80:  # Very strict - only primary job function
                        q_objects |= Q(primary_job_function__icontains=search)
                    elif relevance_level >= 60:  # Strict - primary + secondary job functions
                        q_objects |= Q(primary_job_function__icontains=search)
                        q_objects |= Q(secondary_job_function__icontains=search)
                    elif relevance_level >= 40:  # Medium - functions + industries
                        q_objects |= Q(primary_job_function__icontains=search)
                        q_objects |= Q(secondary_job_function__icontains=search)
                        q_objects |= Q(primary_industry__icontains=search)
                        q_objects |= Q(secondary_industry__icontains=search)
                    elif relevance_level >= 20:  # Broad - include some related skills
                        q_objects |= Q(primary_job_function__icontains=search)
                        q_objects |= Q(secondary_job_function__icontains=search)
                        q_objects |= Q(primary_industry__icontains=search)
                        q_objects |= Q(secondary_industry__icontains=search)
                        q_objects |= Q(related_skill_1__icontains=search)
                        q_objects |= Q(related_skill_2__icontains=search)
                    else:  # Very broad - all fields
                        q_objects |= Q(primary_job_function__icontains=search)
                        q_objects |= Q(secondary_job_function__icontains=search)
                        q_objects |= Q(primary_industry__icontains=search)
                        q_objects |= Q(secondary_industry__icontains=search)
                        q_objects |= Q(related_skill_1__icontains=search)
                        q_objects |= Q(related_skill_2__icontains=search)
                        q_objects |= Q(related_skill_3__icontains=search)
                    
                    # Get matching skills from SkillJobFunction
                    job_function_matches = SkillJobFunction.objects.filter(q_objects).values_list('skill', flat=True)
                    matching_skill_names.update(job_function_matches)
            
            # Apply selected filters (industries, job functions, skills)
            if selected_industries or selected_job_functions or selected_skills:
                filter_skill_names = set()
                
                # Industry-based filtering
                if selected_industries:
                    for industry in selected_industries:
                        industry_skills = SkillJobFunction.objects.filter(
                            Q(primary_industry__iexact=industry) | 
                            Q(secondary_industry__iexact=industry)
                        ).values_list('skill', flat=True)
                        filter_skill_names.update(industry_skills)
                
                # Job function-based filtering
                if selected_job_functions:
                    for job_function in selected_job_functions:
                        function_skills = SkillJobFunction.objects.filter(
                            Q(primary_job_function__iexact=job_function) | 
                            Q(secondary_job_function__iexact=job_function)
                        ).values_list('skill', flat=True)
                        filter_skill_names.update(function_skills)
                
                # Direct skill filtering
                if selected_skills:
                    filter_skill_names.update(selected_skills)
                
                # If we have search results, intersect with filter results
                if search and matching_skill_names:
                    matching_skill_names = matching_skill_names.intersection(filter_skill_names)
                elif filter_skill_names:
                    # No search, just use filter results
                    matching_skill_names = filter_skill_names
            
            # Now filter the queryset to only include matching skills
            if search or selected_industries or selected_job_functions or selected_skills:
                if matching_skill_names:
                    queryset = queryset.filter(skill__in=list(matching_skill_names))
                else:
                    # No matches found
                    queryset = queryset.none()
            
            # Apply additional filters
            filtered_skills = []
            for skill_record in queryset:
                include_skill = True
                
                # Get additional data for filtering
                skill_data = {
                    "skill_record": skill_record,
                    "demand": None,
                    "risk": None,
                    "classification": None
                }
                
                # Get demand and risk data
                try:
                    demand_data = SkillsDemand.objects.filter(skill__iexact=skill_record.skill).first()
                    if demand_data:
                        skill_data["demand"] = demand_data.demand
                        skill_data["risk"] = demand_data.risk
                except:
                    pass
                
                # Get classification
                try:
                    from extractor.models import Skill
                    skill_obj = Skill.objects.filter(skill_name__iexact=skill_record.skill).first()
                    if skill_obj:
                        skill_data["classification"] = skill_obj.classification
                except:
                    pass
                
                # Filter by demand levels
                if demand_levels and include_skill:
                    if not skill_data["demand"] or skill_data["demand"] not in demand_levels:
                        include_skill = False
                
                # Filter by risk levels
                if risk_levels and include_skill:
                    if not skill_data["risk"] or skill_data["risk"] not in risk_levels:
                        include_skill = False
                
                # Filter by classification
                if classification and include_skill:
                    if not skill_data["classification"] or not (classification.lower() in skill_data["classification"].lower()):
                        include_skill = False
                
                if include_skill:
                    filtered_skills.append(skill_data)
            
            print(f"📊 Total matching skills: {len(filtered_skills)}")
            
            # Sort by skill name
            filtered_skills.sort(key=lambda x: x['skill_record'].skill.lower())
            
            # Pagination
            total_count = len(filtered_skills)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_skills = filtered_skills[start_idx:end_idx]
            
            # Build response data
            skills_data = []
            for item in paginated_skills:
                skill_record = item['skill_record']
                
                skill_data = {
                    "id": skill_record.id,
                    "skill": skill_record.skill,
                    "skill_category": skill_record.skill_category,
                    "job_titles": skill_record.get_all_job_titles(),
                    "created_at": skill_record.created_at.isoformat()
                }
                
                # Add additional data from related models
                try:
                    # Get job function data
                    job_function_data = SkillJobFunction.objects.filter(skill__iexact=skill_record.skill).first()
                    if job_function_data:
                        skill_data.update({
                            "skill_type": job_function_data.skill_type,
                            "primary_industry": job_function_data.primary_industry,
                            "secondary_industry": job_function_data.secondary_industry,
                            "primary_job_function": job_function_data.primary_job_function,
                            "secondary_job_function": job_function_data.secondary_job_function,
                            "related_skills": job_function_data.get_all_related_skills(),
                            "skill_definition": job_function_data.skill_definition
                        })
                    
                    # Get demand data
                    demand_data = SkillsDemand.objects.filter(skill__iexact=skill_record.skill).first()
                    if demand_data:
                        skill_data.update({
                            "demand": demand_data.demand,
                            "demand_rationale": demand_data.demand_rationale,
                            "risk": demand_data.risk,
                            "risk_rationale": demand_data.risk_rationale
                        })
                    
                    # Get classification from main Skill model
                    from extractor.models import Skill
                    skill_obj = Skill.objects.filter(skill_name__iexact=skill_record.skill).first()
                    if skill_obj:
                        skill_data.update({
                            "classification": skill_obj.classification,
                            "skill_definition": skill_obj.skill_definition or skill_data.get("skill_definition")
                        })
                
                except Exception as e:
                    print(f"Error fetching additional data for skill {skill_record.skill}: {e}")
                
                skills_data.append(skill_data)
            
            # Calculate pagination info
            total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
            
            return JsonResponse({
                "skills": skills_data,
                "pagination": {
                    "current_page": page,
                    "total_pages": total_pages,
                    "total_count": total_count,
                    "page_size": page_size,
                    "has_next": page < total_pages,
                    "has_previous": page > 1
                },
                "search_info": {
                    "search_term": search,
                    "search_mode": search_mode,
                    "relevance_level": relevance_level,
                    "matching_skills_count": len(matching_skill_names) if search else 0
                },
                "filters": {
                    "skill_category": skill_category,
                    "demand_levels": demand_levels,
                    "risk_levels": risk_levels,
                    "classification": classification,
                    "selected_industries": selected_industries,
                    "selected_job_functions": selected_job_functions,
                    "selected_skills": selected_skills
                }
            })
            
        except Exception as e:
            import traceback
            print(f"Error in get_skills_job_titles: {e}")
            print(traceback.format_exc())
            return JsonResponse({"error": f"Error fetching skills: {str(e)}"}, status=500)

# Updated views.py

def predictive_search(request):
    """
    Returns predictive search suggestions for job titles or job functions
    """
    query = request.GET.get('query', '').strip().lower()
    search_type = request.GET.get('type', '').strip().lower()
    limit = int(request.GET.get('limit', 50))
    relevance_level = int(request.GET.get('relevance_level', 50))
    
    if not query or len(query) < 1:
        return JsonResponse({'suggestions': []})
    
    suggestions = []
    
    if search_type == 'job_title':
        # Dictionary to store unique job titles and their skill counts
        title_to_skills = {}
        all_unique_skills = set()  # Track all unique skills across all titles
        
        # Apply relevance filtering to the query
        if relevance_level >= 80:
            # Strictest - only job_title_1
            matching_rows = SkillsJobTitles.objects.filter(
                job_title_1__icontains=query
            )
        elif relevance_level >= 60:
            # Strict - job_title_1 and job_title_2
            matching_rows = SkillsJobTitles.objects.filter(
                Q(job_title_1__icontains=query) |
                Q(job_title_2__icontains=query)
            )
        elif relevance_level >= 40:
            # Moderate - first 3 title fields
            matching_rows = SkillsJobTitles.objects.filter(
                Q(job_title_1__icontains=query) |
                Q(job_title_2__icontains=query) |
                Q(job_title_3__icontains=query)
            )
        elif relevance_level >= 20:
            # Broad - first 4 title fields
            matching_rows = SkillsJobTitles.objects.filter(
                Q(job_title_1__icontains=query) |
                Q(job_title_2__icontains=query) |
                Q(job_title_3__icontains=query) |
                Q(job_title_4__icontains=query)
            )
        else:
            # Broadest - all 5 title fields
            matching_rows = SkillsJobTitles.objects.filter(
                Q(job_title_1__icontains=query) |
                Q(job_title_2__icontains=query) |
                Q(job_title_3__icontains=query) |
                Q(job_title_4__icontains=query) |
                Q(job_title_5__icontains=query)
            )
        
        # Build a map of job titles to their associated skills
        for row in matching_rows:
            skill_name = row.skill  # Assuming the field is called 'skill'
            
            # Check each field based on relevance level
            for field_num, field in enumerate(['job_title_1', 'job_title_2', 'job_title_3', 'job_title_4', 'job_title_5'], 1):
                # Skip fields that are beyond the relevance level
                if relevance_level >= 80 and field_num > 1:
                    break
                elif relevance_level >= 60 and field_num > 2:
                    break
                elif relevance_level >= 40 and field_num > 3:
                    break
                elif relevance_level >= 20 and field_num > 4:
                    break
                
                title = getattr(row, field, None)
                if title and query in title.lower():
                    if title not in title_to_skills:
                        title_to_skills[title] = set()
                    title_to_skills[title].add(skill_name)
                    all_unique_skills.add(skill_name)  # Add to total unique skills
        
        # The total count should be all unique skills found
        total_unique_count = len(all_unique_skills)
        
        # Convert to suggestions with actual skill counts
        for title, skills in sorted(title_to_skills.items(), key=lambda x: (-len(x[1]), x[0]))[:limit]:
            suggestions.append({
                'value': title,
                'count': len(skills)  # Count of unique skills for this specific title
            })
        
        # Return both suggestions and the total count
        return JsonResponse({
            'suggestions': suggestions,
            'total_count': total_unique_count  # This is what should be used for "All matching"
        })
    
    elif search_type == 'job_function':
        # Dictionary to store unique job functions and their skill counts
        function_to_skills = {}
        all_unique_skills = set()  # Track all unique skills
        
        # Apply relevance filtering
        if relevance_level >= 60:
            # Only primary job functions
            matching_rows = SkillJobFunction.objects.filter(
                primary_job_function__icontains=query
            )
        else:
            # Both primary and secondary
            matching_rows = SkillJobFunction.objects.filter(
                Q(primary_job_function__icontains=query) |
                Q(secondary_job_function__icontains=query)
            )
        
        # Build a map of job functions to their associated skills
        for row in matching_rows:
            skill_name = row.skill  # Assuming the field is called 'skill'
            
            if row.primary_job_function and query in row.primary_job_function.lower():
                if row.primary_job_function not in function_to_skills:
                    function_to_skills[row.primary_job_function] = set()
                function_to_skills[row.primary_job_function].add(skill_name)
                all_unique_skills.add(skill_name)
            
            # Only include secondary if relevance < 60
            if relevance_level < 60 and row.secondary_job_function and query in row.secondary_job_function.lower():
                if row.secondary_job_function not in function_to_skills:
                    function_to_skills[row.secondary_job_function] = set()
                function_to_skills[row.secondary_job_function].add(skill_name)
                all_unique_skills.add(skill_name)
        
        # The total count should be all unique skills found
        total_unique_count = len(all_unique_skills)
        
        # Convert to suggestions with actual skill counts
        for function, skills in sorted(function_to_skills.items(), key=lambda x: (-len(x[1]), x[0]))[:limit]:
            suggestions.append({
                'value': function,
                'count': len(skills)  # Count of unique skills for this specific function
            })
        
        # Return both suggestions and the total count
        return JsonResponse({
            'suggestions': suggestions,
            'total_count': total_unique_count
        })
    
    return JsonResponse({'suggestions': []})

def search_skills_enhanced(request):
    """
    Enhanced search with multiple search terms and relevance filtering
    """
    search_terms = request.GET.getlist('search_terms[]')
    search_type = request.GET.get('search_type', '').strip().lower()
    relevance_level = int(request.GET.get('relevance_level', 50))
    
    if not search_terms or search_type not in ['job_title', 'job_function']:
        return JsonResponse({'error': 'Invalid parameters'}, status=400)
    
    all_matched_skills = set()
    
    # Process each search term
    for search_term in search_terms:
        search_term = search_term.strip()  # Don't lowercase to preserve exact matches
        
        if search_type == 'job_function':
            # For job functions, apply relevance filtering
            if relevance_level >= 60:
                # Only primary job functions
                matched_job_functions = SkillJobFunction.objects.filter(
                    primary_job_function__icontains=search_term
                ).values_list('skill', flat=True)
            else:
                # Both primary and secondary
                matched_job_functions = SkillJobFunction.objects.filter(
                    Q(primary_job_function__icontains=search_term) |
                    Q(secondary_job_function__icontains=search_term)
                ).values_list('skill', flat=True)
            all_matched_skills.update(matched_job_functions)
            
        elif search_type == 'job_title':
            # Check if this is a general search (e.g., "All matching...")
            if search_term.startswith('All matching'):
                # Extract the actual search term from quotes
                import re
                match = re.search(r'"([^"]*)"', search_term)
                if match:
                    actual_term = match.group(1).lower()
                    # Apply relevance filtering even for "All matching" queries
                    if relevance_level >= 80:
                        # Strictest - only matches in job_title_1
                        matched_titles = SkillsJobTitles.objects.filter(
                            job_title_1__icontains=actual_term
                        ).values_list('skill', flat=True)
                    elif relevance_level >= 60:
                        # Strict - matches in job_title_1 and job_title_2
                        matched_titles = SkillsJobTitles.objects.filter(
                            Q(job_title_1__icontains=actual_term) |
                            Q(job_title_2__icontains=actual_term)
                        ).values_list('skill', flat=True)
                    elif relevance_level >= 40:
                        # Moderate - matches in first 3 title fields
                        matched_titles = SkillsJobTitles.objects.filter(
                            Q(job_title_1__icontains=actual_term) |
                            Q(job_title_2__icontains=actual_term) |
                            Q(job_title_3__icontains=actual_term)
                        ).values_list('skill', flat=True)
                    elif relevance_level >= 20:
                        # Broad - matches in first 4 title fields
                        matched_titles = SkillsJobTitles.objects.filter(
                            Q(job_title_1__icontains=actual_term) |
                            Q(job_title_2__icontains=actual_term) |
                            Q(job_title_3__icontains=actual_term) |
                            Q(job_title_4__icontains=actual_term)
                        ).values_list('skill', flat=True)
                    else:
                        # Broadest - matches in all 5 title fields
                        matched_titles = SkillsJobTitles.objects.filter(
                            Q(job_title_1__icontains=actual_term) |
                            Q(job_title_2__icontains=actual_term) |
                            Q(job_title_3__icontains=actual_term) |
                            Q(job_title_4__icontains=actual_term) |
                            Q(job_title_5__icontains=actual_term)
                        ).values_list('skill', flat=True)
                    all_matched_skills.update(matched_titles)
            else:
                # Apply relevance-based filtering for specific job titles
                search_term_lower = search_term.lower()
                if relevance_level >= 80:
                    # Strictest - only matches in job_title_1
                    matched_titles = SkillsJobTitles.objects.filter(
                        job_title_1__icontains=search_term_lower
                    ).values_list('skill', flat=True)
                elif relevance_level >= 60:
                    # Strict - matches in job_title_1 and job_title_2
                    matched_titles = SkillsJobTitles.objects.filter(
                        Q(job_title_1__icontains=search_term_lower) |
                        Q(job_title_2__icontains=search_term_lower)
                    ).values_list('skill', flat=True)
                elif relevance_level >= 40:
                    # Moderate - matches in first 3 title fields
                    matched_titles = SkillsJobTitles.objects.filter(
                        Q(job_title_1__icontains=search_term_lower) |
                        Q(job_title_2__icontains=search_term_lower) |
                        Q(job_title_3__icontains=search_term_lower)
                    ).values_list('skill', flat=True)
                elif relevance_level >= 20:
                    # Broad - matches in first 4 title fields
                    matched_titles = SkillsJobTitles.objects.filter(
                        Q(job_title_1__icontains=search_term_lower) |
                        Q(job_title_2__icontains=search_term_lower) |
                        Q(job_title_3__icontains=search_term_lower) |
                        Q(job_title_4__icontains=search_term_lower)
                    ).values_list('skill', flat=True)
                else:
                    # Broadest - matches in all 5 title fields
                    matched_titles = SkillsJobTitles.objects.filter(
                        Q(job_title_1__icontains=search_term_lower) |
                        Q(job_title_2__icontains=search_term_lower) |
                        Q(job_title_3__icontains=search_term_lower) |
                        Q(job_title_4__icontains=search_term_lower) |
                        Q(job_title_5__icontains=search_term_lower)
                    ).values_list('skill', flat=True)
                
                all_matched_skills.update(matched_titles)
    
    # Convert to list to ensure uniqueness is maintained
    matched_skills_list = list(all_matched_skills)
    
    # Fetch skill details
    skills_data = Skill.objects.filter(skill_name__in=matched_skills_list)
    demand_data = SkillsDemand.objects.filter(skill__in=matched_skills_list)
    job_func_data = SkillJobFunction.objects.filter(skill__in=matched_skills_list)
    
    # Create lookup dictionaries
    skills_dict = {s.skill_name: s for s in skills_data}
    demand_dict = {d.skill: d for d in demand_data}
    job_func_dict = {j.skill: j for j in job_func_data}
    
    # Build response
    response_data = []
    for skill_name in matched_skills_list:
        skill_obj = skills_dict.get(skill_name)
        demand_obj = demand_dict.get(skill_name)
        job_func_obj = job_func_dict.get(skill_name)
        
        response_data.append({
            'skill_name': skill_name,
            'classification': skill_obj.classification if skill_obj else '',
            'definition': skill_obj.skill_definition if skill_obj else '',
            'demand_level': demand_obj.demand if demand_obj else '',
            'obsolescence_risk': demand_obj.risk if demand_obj else '',
            'industry': (job_func_obj.primary_industry or job_func_obj.secondary_industry) if job_func_obj else '',
            'job_function': (job_func_obj.primary_job_function or job_func_obj.secondary_job_function) if job_func_obj else '',
        })
    
    return JsonResponse({
        'count': len(response_data),  # This ensures the count matches the actual unique results
        'results': response_data
    })

def search_skills(request):
    search_term = request.GET.get('search_term', '').strip().lower()
    search_type = request.GET.get('search_type', '').strip().lower()
    
    if not search_term or search_type not in ['job_title', 'job_function']:
        return JsonResponse({'error': 'Invalid parameters'}, status=400)
    
    matched_skills = set()
    
    # Fetch Matched Skills
    if search_type == 'job_function':
        matched_job_functions = SkillJobFunction.objects.filter(
            Q(primary_job_function__icontains=search_term) |
            Q(secondary_job_function__icontains=search_term)
        ).values_list('skill', flat=True)
        matched_skills.update(matched_job_functions)
    elif search_type == 'job_title':
        matched_titles = SkillsJobTitles.objects.filter(
            Q(job_title_1__icontains=search_term) |
            Q(job_title_2__icontains=search_term) |
            Q(job_title_3__icontains=search_term) |
            Q(job_title_4__icontains=search_term) |
            Q(job_title_5__icontains=search_term)
        ).values_list('skill', flat=True)
        matched_skills.update(matched_titles)
    
    # Convert to list
    matched_skills_list = list(matched_skills)
    
    # If you just need the count for the predictive search
    if request.GET.get('count_only') == 'true':
        return JsonResponse({'count': len(matched_skills_list)})
    
    # Bulk Fetch All Related Models
    skills_data = Skill.objects.filter(skill_name__in=matched_skills_list)
    demand_data = SkillsDemand.objects.filter(skill__in=matched_skills_list)
    job_func_data = SkillJobFunction.objects.filter(skill__in=matched_skills_list)
    
    # Create Dicts for Fast Access
    skills_dict = {s.skill_name: s for s in skills_data}
    demand_dict = {d.skill: d for d in demand_data}
    job_func_dict = {j.skill: j for j in job_func_data}
    
    # Compose Response Efficiently
    response_data = []
    for skill_name in matched_skills_list:
        skill_obj = skills_dict.get(skill_name)
        demand_obj = demand_dict.get(skill_name)
        job_func_obj = job_func_dict.get(skill_name)
        
        response_data.append({
            'skill_name': skill_name,
            'classification': skill_obj.classification if skill_obj else '',
            'definition': skill_obj.skill_definition if skill_obj else '',
            'demand_level': demand_obj.demand if demand_obj else '',
            'obsolescence_risk': demand_obj.risk if demand_obj else '',
            'industry': (job_func_obj.primary_industry or job_func_obj.secondary_industry) if job_func_obj else '',
            'job_function': (job_func_obj.primary_job_function or job_func_obj.secondary_job_function) if job_func_obj else '',
        })
    
    return JsonResponse({
        'count': len(response_data),
        'results': response_data
    })

@csrf_exempt
def get_industries(request):
    """Get unique industries for predictive search"""
    if request.method == "GET":
        try:
            query = request.GET.get('query', '').strip().lower()
            
            # Get all unique industries from SkillJobFunction
            all_industries = set()
            
            # Query both primary and secondary industries
            if query:
                primary_industries = SkillJobFunction.objects.filter(
                    primary_industry__icontains=query
                ).values_list('primary_industry', flat=True).distinct()
                
                secondary_industries = SkillJobFunction.objects.filter(
                    secondary_industry__icontains=query
                ).exclude(secondary_industry='').values_list('secondary_industry', flat=True).distinct()
                
                all_industries.update(primary_industries)
                all_industries.update(secondary_industries)
            else:
                # If no query, return top industries
                primary_industries = SkillJobFunction.objects.exclude(
                    primary_industry=''
                ).values_list('primary_industry', flat=True).distinct()[:20]
                
                secondary_industries = SkillJobFunction.objects.exclude(
                    secondary_industry=''
                ).values_list('secondary_industry', flat=True).distinct()[:20]
                
                all_industries.update(primary_industries)
                all_industries.update(secondary_industries)
            
            # Convert to list and sort
            industries_list = sorted(list(all_industries))[:50]  # Limit to 50 results
            
            # Format for frontend
            industries_data = [{'value': industry} for industry in industries_list if industry]
            
            return JsonResponse({'industries': industries_data})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def analyze_skill_overlap(request):
    """
    Analyze skill overlap between two positions
    
    Expected POST body:
    {
        "position_a_skills": [123, 456, 789],  // List of skill IDs
        "position_b_skills": [123, 999, 888],  // List of skill IDs
        "include_details": true  // Optional, default false
    }
    """
    if request.method == "POST":
        try:
            start_time = time.time()
            
            # Parse request data
            data = json.loads(request.body)
            position_a_skills = data.get("position_a_skills", [])
            position_b_skills = data.get("position_b_skills", [])
            include_details = data.get("include_details", False)
            
            if not position_a_skills or not position_b_skills:
                return JsonResponse({
                    "error": "Both position_a_skills and position_b_skills are required"
                }, status=400)
            
            # Convert to sets of strings for processing
            position_a_set = set(str(skill_id) for skill_id in position_a_skills)
            position_b_set = set(str(skill_id) for skill_id in position_b_skills)
            
            # Initialize analyzer
            analyzer = SkillOverlapAnalyzer()
            
            # Load skill information
            all_skill_ids = list(set(position_a_skills + position_b_skills))
            skill_info = analyzer.load_skill_info_from_db(all_skill_ids)
            
            # Build similarity map from database
            similarity_map = analyzer.build_similarity_map_from_db(
                position_a_set, position_b_set
            )
            
            # Find overlaps in both directions
            a_to_b_results, a_to_b_matches, a_to_b_similarity_sum = analyzer.find_skill_overlap(
                position_a_set, position_b_set, similarity_map, "A→B", skill_info
            )
            
            b_to_a_results, b_to_a_matches, b_to_a_similarity_sum = analyzer.find_skill_overlap(
                position_b_set, position_a_set, similarity_map, "B→A", skill_info
            )
            
            # Calculate similarity metrics
            jaccard_similarity = analyzer.calculate_jaccard_similarity(
                a_to_b_results, b_to_a_results, position_a_set, position_b_set
            )
            
            weighted_similarity = analyzer.calculate_weighted_jaccard(
                a_to_b_results, b_to_a_results, position_a_set, position_b_set
            )
            
            # Calculate percentages
            a_match_percentage = (a_to_b_matches / len(position_a_set) * 100) if position_a_set else 0
            b_match_percentage = (b_to_a_matches / len(position_b_set) * 100) if position_b_set else 0
            
            # Build response
            response_data = {
                "summary": {
                    "position_a_skills_count": len(position_a_set),
                    "position_b_skills_count": len(position_b_set),
                    "matches_a_to_b": a_to_b_matches,
                    "matches_b_to_a": b_to_a_matches,
                    "match_percentage_a": round(a_match_percentage, 1),
                    "match_percentage_b": round(b_match_percentage, 1),
                    "jaccard_similarity": round(jaccard_similarity, 3),
                    "weighted_similarity": round(weighted_similarity, 3),
                    "processing_time_seconds": round(time.time() - start_time, 3)
                }
            }
            
            # Include detailed results if requested
            if include_details:
                response_data["details"] = {
                    "a_to_b_matches": [
                        {
                            "source_id": result[0],
                            "source_name": result[3],
                            "source_classification": result[4],
                            "target_id": result[1],
                            "target_name": result[5],
                            "target_classification": result[6],
                            "similarity_score": result[2]
                        }
                        for result in a_to_b_results
                    ],
                    "b_to_a_matches": [
                        {
                            "source_id": result[0],
                            "source_name": result[3],
                            "source_classification": result[4],
                            "target_id": result[1],
                            "target_name": result[5],
                            "target_classification": result[6],
                            "similarity_score": result[2]
                        }
                        for result in b_to_a_results
                    ]
                }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error in skill overlap analysis: {str(e)}", exc_info=True)
            return JsonResponse({"error": f"Analysis error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)


@csrf_exempt
def get_skill_similarities(request, skill_id):
    """Get similarity data for a specific skill"""
    if request.method == "GET":
        try:
            # Get threshold parameter if provided
            threshold = request.GET.get('threshold', None)
            
            # Fetch similarity data
            similarity = SkillSimilarity.objects.filter(skill_id=skill_id).first()
            
            if not similarity:
                return JsonResponse({"error": "No similarity data found for this skill"}, status=404)
            
            # Get skill info
            skill = Skill.objects.filter(id=skill_id).first()
            
            response_data = {
                "skill_id": skill_id,
                "skill_name": skill.skill_name if skill else f"Skill {skill_id}",
                "thresholds_available": list(similarity.similarity_thresholds.keys())
            }
            
            if threshold:
                # Return specific threshold data
                threshold_data = similarity.similarity_thresholds.get(str(threshold), [])
                response_data["threshold"] = threshold
                response_data["similar_skills"] = threshold_data
                response_data["similar_skills_count"] = len(threshold_data)
            else:
                # Return all threshold data
                response_data["all_thresholds"] = similarity.similarity_thresholds
                response_data["total_similar_skills"] = sum(
                    len(skills) for skills in similarity.similarity_thresholds.values()
                )
            
            return JsonResponse(response_data)
            
        except Exception as e:
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def analyze_profile_overlap(request):
    """
    Analyze skill overlap between two profiles/files using their IDs
    
    Expected POST body:
    {
        "profile_1": 12,
        "profile_2": 54
    }
    """
    if request.method == "POST":
        try:
            start_time = time.time()
            
            # Parse request data
            data = json.loads(request.body)
            profile_1_id = data.get("profile_1")
            profile_2_id = data.get("profile_2")
            
            if not profile_1_id or not profile_2_id:
                return JsonResponse({
                    "error": "Both profile_1 and profile_2 are required"
                }, status=400)
            
            # Get files from database
            try:
                file_1 = File.objects.get(id=profile_1_id)
                file_2 = File.objects.get(id=profile_2_id)
            except File.DoesNotExist as e:
                return JsonResponse({
                    "error": f"Profile not found: {str(e)}"
                }, status=404)
            
            # Extract skill IDs and create skill maps for classifications and other fields
            skills_1 = []
            skills_2 = []
            skill_classifications = {}  # Map skill_id -> classification
            skill_details_1 = {}  # Map skill_id -> full skill details for profile 1
            skill_details_2 = {}  # Map skill_id -> full skill details for profile 2
            
            # Extract from profile 1
            if file_1.skill_profile and 'skills' in file_1.skill_profile:
                for skill in file_1.skill_profile['skills']:
                    if skill.get('id'):
                        skill_id = skill['id']
                        skills_1.append(skill_id)
                        # Store all skill details
                        skill_details_1[str(skill_id)] = {
                            'definition': skill.get('definition', ''),
                            'demand': skill.get('marketDemand', ''),
                            'risk': skill.get('obsolescenceRisk', ''),
                            'demandRationale': skill.get('demandRationale', ''),
                            'riskRationale': skill.get('riskRationale', ''),
                            'classification': skill.get('classification', '')
                        }
            
            # Extract from profile 2 and build classification map
            if file_2.skill_profile and 'skills' in file_2.skill_profile:
                for skill in file_2.skill_profile['skills']:
                    if skill.get('id'):
                        skill_id = skill['id']
                        skills_2.append(skill_id)
                        # Store classification for profile 2 skills
                        if skill.get('classification'):
                            skill_classifications[str(skill_id)] = skill['classification']
                        # Store all skill details
                        skill_details_2[str(skill_id)] = {
                            'definition': skill.get('definition', ''),
                            'demand': skill.get('marketDemand', ''),
                            'risk': skill.get('obsolescenceRisk', ''),
                            'demandRationale': skill.get('demandRationale', ''),
                            'riskRationale': skill.get('riskRationale', ''),
                            'classification': skill.get('classification', '')
                        }
            
            if not skills_1:
                return JsonResponse({
                    "error": f"No skills found in profile {profile_1_id} ({file_1.filename})"
                }, status=400)
                
            if not skills_2:
                return JsonResponse({
                    "error": f"No skills found in profile {profile_2_id} ({file_2.filename})"
                }, status=400)
            
            # Convert to sets of strings for processing
            position_a_set = set(str(skill_id) for skill_id in skills_1)
            position_b_set = set(str(skill_id) for skill_id in skills_2)
            
            # Initialize analyzer
            analyzer = SkillOverlapAnalyzer()
            
            # Load skill information
            all_skill_ids = list(set(skills_1 + skills_2))
            skill_info = analyzer.load_skill_info_from_db(all_skill_ids)
            
            # Build similarity map from database
            similarity_map = analyzer.build_similarity_map_from_db(
                position_a_set, position_b_set
            )
            
            # Find overlaps in both directions
            a_to_b_results, a_to_b_matches, a_to_b_similarity_sum = analyzer.find_skill_overlap(
                position_a_set, position_b_set, similarity_map, "Profile 1→Profile 2", skill_info
            )
            
            b_to_a_results, b_to_a_matches, b_to_a_similarity_sum = analyzer.find_skill_overlap(
                position_b_set, position_a_set, similarity_map, "Profile 2→Profile 1", skill_info
            )
            
            # Calculate similarity metrics
            jaccard_similarity = analyzer.calculate_jaccard_similarity(
                a_to_b_results, b_to_a_results, position_a_set, position_b_set
            )
            
            weighted_similarity = analyzer.calculate_weighted_jaccard(
                a_to_b_results, b_to_a_results, position_a_set, position_b_set
            )
            
            # Calculate percentages
            a_match_percentage = (a_to_b_matches / len(position_a_set) * 100) if position_a_set else 0
            b_match_percentage = (b_to_a_matches / len(position_b_set) * 100) if position_b_set else 0
            
            # Build response
            response_data = {
                "profile_info": {
                    "profile_1": {
                        "id": profile_1_id,
                        "filename": file_1.filename,
                        "folder": file_1.folder.name,
                        "skills_count": len(skills_1)
                    },
                    "profile_2": {
                        "id": profile_2_id,
                        "filename": file_2.filename,
                        "folder": file_2.folder.name,
                        "skills_count": len(skills_2)
                    }
                },
                "overlap_summary": {
                    "total_unique_skills": len(set(skills_1 + skills_2)),
                    "common_skills": len(set(skills_1) & set(skills_2)),
                    "profile_1_match_percentage": round(a_match_percentage, 1),
                    "profile_2_match_percentage": round(b_match_percentage, 1),
                    "jaccard_similarity": round(jaccard_similarity, 3),
                    "weighted_similarity": round(weighted_similarity, 3)
                },
                "detailed_results": {
                    "profile_1_to_profile_2": {
                        "matched_skills": a_to_b_matches,
                        "unmatched_skills": len(skills_1) - a_to_b_matches,
                        "match_percentage": round(a_match_percentage, 1)
                    },
                    "profile_2_to_profile_1": {
                        "matched_skills": b_to_a_matches,
                        "unmatched_skills": len(skills_2) - b_to_a_matches,
                        "match_percentage": round(b_match_percentage, 1)
                    }
                },
                "processing_time_seconds": round(time.time() - start_time, 3)
            }
            
            # Include ALL skills from both directions, including unmatched ones
            all_matches_a_to_b = []
            all_matches_b_to_a = []
            
            # Get ALL A to B results INCLUDING those with no match (similarity = 0)
            for result in a_to_b_results:
                # Extract skill IDs
                skill_id_from = result[0]
                skill_id_to = result[1] if result[1] else None
                
                # Get skill details from profile 1
                from_details = skill_details_1.get(str(skill_id_from), {})
                
                # Get skill details from profile 2 if there's a match
                to_details = {}
                if skill_id_to:
                    to_details = skill_details_2.get(str(skill_id_to), {})
                
                # Include ALL results, even those with similarity = 0
                all_matches_a_to_b.append({
                    "from_profile_1": result[3],  # skill name
                    "from_profile_1_details": from_details,  # Add skill details
                    "to_profile_2": result[5] if result[1] else "",  # matched skill name or empty
                    "to_profile_2_details": to_details if skill_id_to else {},  # Add matched skill details
                    "similarity": round(result[2], 2),
                    "direction": "A_to_B"
                })
            
            # Get ALL B to A results INCLUDING those with no match (similarity = 0)
            for result in b_to_a_results:
                # Extract skill IDs
                skill_id_from = result[0]
                skill_id_to = result[1] if result[1] else None
                
                # Get classification and details for profile 2 skill
                classification = skill_classifications.get(str(skill_id_from), "Unknown")
                from_details = skill_details_2.get(str(skill_id_from), {})
                
                # Get skill details from profile 1 if there's a match
                to_details = {}
                if skill_id_to:
                    to_details = skill_details_1.get(str(skill_id_to), {})
                
                # Include ALL results, even those with similarity = 0
                all_matches_b_to_a.append({
                    "from_profile_2": result[3],  # skill name
                    "from_profile_2_details": from_details,  # Add skill details
                    "to_profile_1": result[5] if result[1] else "",  # matched skill name or empty
                    "to_profile_1_details": to_details if skill_id_to else {},  # Add matched skill details
                    "similarity": round(result[2], 2),
                    "direction": "B_to_A",
                    "classification": classification  # Add classification for profile 2 skills
                })
            
            # Add all matches to response
            response_data["all_matches"] = {
                "a_to_b": all_matches_a_to_b,
                "b_to_a": all_matches_b_to_a
            }
            
            # Keep top_matches for backward compatibility (top 10 A to B)
            top_matches = []
            for result in a_to_b_results[:10]:  # Top 10 matches
                if result[1] and result[2] > 0:  # Has a match
                    top_matches.append({
                        "from_profile_1": result[3],  # skill name
                        "to_profile_2": result[5],    # matched skill name
                        "similarity": round(result[2], 2)
                    })
            
            if top_matches:
                response_data["top_matches"] = top_matches
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error in profile overlap analysis: {str(e)}", exc_info=True)
            return JsonResponse({"error": f"Analysis error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def get_skill_detail(request, skill_identifier):
    """
    Get comprehensive skill details by ID or slug (skill name)
    Accepts either numeric ID or skill name as identifier
    """
    if request.method == "GET":
        try:
            # URL decode the identifier
            skill_identifier = unquote(skill_identifier).strip()
            
            # Try to parse as ID first
            try:
                skill_id = int(skill_identifier)
                skill = Skill.objects.filter(id=skill_id).first()
            except ValueError:
                # If not a number, treat as skill name (slug)
                skill = Skill.objects.filter(skill_name__iexact=skill_identifier).first()
            
            if not skill:
                return JsonResponse({"error": "Skill not found"}, status=404)
            
            # Get related data
            skill_job_titles = SkillsJobTitles.objects.filter(skill__iexact=skill.skill_name).first()
            skill_job_function = SkillJobFunction.objects.filter(skill__iexact=skill.skill_name).first()
            skill_demand = SkillsDemand.objects.filter(skill__iexact=skill.skill_name).first()
            
            # Build response data
            response_data = {
                "skill_id": skill.id,
                "skill_name": skill.skill_name,
                "skill_definition": skill.skill_definition or "Definition not available",
                "classification": skill.classification or "Unknown",
                
                # Industries
                "industries": {
                    "primary": skill_job_function.primary_industry if skill_job_function else None,
                    "secondary": skill_job_function.secondary_industry if skill_job_function and skill_job_function.secondary_industry else None,
                },
                
                # Job Functions
                "job_functions": {
                    "primary": [],
                    "secondary": []
                },
                
                # Relevant Jobs (from job titles)
                "relevant_jobs": [],
                
                # Labor Market Intelligence
                "labor_market_intel": {
                    "demand": skill_demand.demand if skill_demand else "Data not available",
                    "demand_rationale": skill_demand.demand_rationale if skill_demand else "Labor market analysis is not available for this skill at the moment. Check back later for updated insights.",
                },
                
                # Obsolescence Risk
                "obsolescence_risk": {
                    "risk": skill_demand.risk if skill_demand else "Data not available",
                    "risk_rationale": skill_demand.risk_rationale if skill_demand else "Risk assessment data is not available for this skill at the moment. Check back later for updated insights.",
                },
                
                # Related Skills
                "related_skills": []
            }
            
            # Populate job functions
            # Populate job functions
            if skill_job_function:
                if skill_job_function.primary_job_function:
                    response_data["job_functions"]["primary"].append(skill_job_function.primary_job_function)
                if skill_job_function.secondary_job_function:
                    response_data["job_functions"]["secondary"].append(skill_job_function.secondary_job_function)

            
            # Get relevant jobs from job titles
            if skill_job_titles:
                job_titles = [
                    skill_job_titles.job_title_1,
                    skill_job_titles.job_title_2,
                    skill_job_titles.job_title_3,
                    skill_job_titles.job_title_4,
                    skill_job_titles.job_title_5,
                ]
                # Filter out empty titles and duplicates
                unique_jobs = []
                seen = set()
                for job in job_titles:
                    if job and job not in seen:
                        unique_jobs.append(job)
                        seen.add(job)
                response_data["relevant_jobs"] = unique_jobs[:5]
            
            # Get related skills
            if skill_job_function:
                related_skills_names = skill_job_function.get_all_related_skills()
                related_skills_data = []
                
                for related_skill_name in related_skills_names[:5]:  # Limit to 5
                    if related_skill_name:  # Check if not empty
                        related_skill = Skill.objects.filter(skill_name__iexact=related_skill_name).first()
                        if related_skill:
                            related_skills_data.append({
                                "id": related_skill.id,
                                "name": related_skill.skill_name,
                                "classification": related_skill.classification or "Unknown"
                            })
                
                response_data["related_skills"] = related_skills_data
            
            # If no related skills found through job function, find similar skills by classification
            if not response_data["related_skills"] and skill.classification:
                similar_skills = Skill.objects.filter(
                    classification=skill.classification
                ).exclude(id=skill.id)[:5]
                
                for similar_skill in similar_skills:
                    response_data["related_skills"].append({
                        "id": similar_skill.id,
                        "name": similar_skill.skill_name,
                        "classification": similar_skill.classification
                    })
            
            return JsonResponse(response_data)
            
        except Exception as e:
            import traceback
            print(f"Error in get_skill_detail: {e}")
            print(traceback.format_exc())
            return JsonResponse({"error": f"Error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)

@csrf_exempt
def search_skills_autocomplete(request):
    """
    Autocomplete search for skills
    Returns suggestions based on partial skill name match
    """
    if request.method == "GET":
        try:
            query = request.GET.get('q', '').strip()
            # limit = min(int(request.GET.get()), 20)
            
            if not query or len(query) < 2:
                return JsonResponse({"suggestions": []})
            
            # Search skills by name (case-insensitive)
            skills = Skill.objects.filter(
                skill_name__icontains=query
            ).order_by('skill_name')[:]
            
            suggestions = []
            for skill in skills:
                # Get classification for additional context
                suggestions.append({
                    "id": skill.id,
                    "name": skill.skill_name,
                    "classification": skill.classification or "Unknown",
                    "display": f"{skill.skill_name} ({skill.classification or 'General'})"
                })
            
            return JsonResponse({
                "suggestions": suggestions,
                "query": query,
                "total": len(suggestions)
            })
            
        except Exception as e:
            print(f"Error in search_skills_autocomplete: {e}")
            return JsonResponse({"error": f"Search error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only GET allowed"}, status=405)


# SKILL MATCH ANALYZER
# SKILL MATCH ANALYZER
# SKILL MATCH ANALYZER
@csrf_exempt
def extract_and_analyze_overlap(request):
    """
    Extract skills from two documents and analyze their overlap
    
    Expected POST body:
    {
        "document_a": "text content for document A...",
        "document_b": "text content for document B..."
    }
    """
    if request.method == "POST":
        start_time = time.time()
        
        try:
            data = json.loads(request.body)
            document_a_text = data.get("document_a", "")
            document_b_text = data.get("document_b", "")
            
            if not document_a_text.strip() or not document_b_text.strip():
                return JsonResponse({
                    "error": "Both document_a and document_b text are required"
                }, status=400)

            # Load skills from DB and build tries (shared for both documents)
            exact, std, lemma, qcw, classes = load_skills_from_db()
            print(f"🔍 Starting extraction for two documents")
            print(f"📄 Document A length: {len(document_a_text)}")
            print(f"📄 Document B length: {len(document_b_text)}")

            # Function to extract skills from a single document
            def extract_skills_from_document(text, doc_name):
                print(f"\n🔍 Extracting skills from {doc_name}")
                
                # Extract skills from parenthetical lists and special patterns
                parenthetical_skills = extract_parenthetical_skills(text)
                print(f"📝 Found {len(parenthetical_skills)} parenthetical/pattern skills in {doc_name}")

                # Add parenthetical skills to the matches if they exist in our database
                verified_parenthetical_matches = []
                for skill_text, start, end in parenthetical_skills:
                    skill_norm = skill_text.lower().strip()
                    
                    # Handle & splitting
                    if '&' in skill_norm:
                        parts = [part.strip() for part in skill_norm.split('&')]
                        print(f"🔄 SPLITTING COMPOUND SKILL: '{skill_text}' -> {parts}")
                        
                        for part in parts:
                            if len(part) > 1:
                                skill_info = get_exact_skill_from_database(part, exact, std, lemma)
                                
                                if skill_info:
                                    if validate_skill_in_text(skill_info["name"], text, start, end):
                                        verified_parenthetical_matches.append((skill_info, start, end, part))
                                        print(f"✅ COMPOUND PART VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')})")
                    else:
                        skill_info = get_exact_skill_from_database(skill_text, exact, std, lemma)
                        
                        if skill_info:
                            if validate_skill_in_text(skill_info["name"], text, start, end):
                                verified_parenthetical_matches.append((skill_info, start, end, skill_text))
                                print(f"✅ SINGLE SKILL VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')})")

                # Continue with regular tokenization and matching
                tokens = tokenize_with_indices(text)
                segments = get_contiguous_segments(tokens)

                matcher = TrieMatcher(
                    exact=exact, std=std, lemma=lemma, is_std=False,
                    qcw=qcw, classes=classes, full_text=text
                )

                all_matches_with_positions = []
                used_positions = {}
                matched_phrases = set()

                for segment in segments:
                    matches = matcher.find_matches(segment, used_positions, matched_phrases)
                    for m in matches:
                        start_pos = m["start"]
                        end_pos = m["end"]
                        actual_text_span = text[start_pos:end_pos].lower().strip()
                        
                        skill_name = m["skill"]
                        
                        if not validate_skill_in_text(skill_name, text, start_pos, end_pos):
                            continue
                        
                        if '&' in skill_name:
                            parts = [part.strip() for part in skill_name.split('&')]
                            
                            if validate_skill_in_text(skill_name, text, start_pos, end_pos):
                                skill_info_dict = {
                                    "name": skill_name,
                                    "definition": "Definition not available",
                                    "id": m.get("id"),
                                    "classification": m.get("classification", "Unknown")
                                }
                                all_matches_with_positions.append((
                                    skill_info_dict,
                                    m["start"], m["end"], skill_name
                                ))
                        else:
                            exact_skill_info = get_exact_skill_from_database(actual_text_span, exact, std, lemma)
                            
                            if exact_skill_info and validate_skill_in_text(exact_skill_info["name"], text, start_pos, end_pos):
                                all_matches_with_positions.append((
                                    exact_skill_info,
                                    m["start"], m["end"], actual_text_span
                                ))

                # Combine regular matches with validated parenthetical matches
                all_matches_with_positions.extend(verified_parenthetical_matches)
                
                # Remove overlapping matches
                final_skills = remove_overlapping_skills(all_matches_with_positions)
                
                # Final validation
                truly_final_skills = []
                for skill in final_skills:
                    skill_name = skill.get("name", "")
                    if validate_skill_in_text(skill_name, text):
                        truly_final_skills.append(skill)
                
                final_skills = truly_final_skills
                print(f"✅ Final skills for {doc_name}: {len(final_skills)}")

                # Add definitions and metadata
                for skill in final_skills:
                    try:
                        if "definition" not in skill or skill["definition"] == "Definition not available":
                            skill_obj = Skill.objects.filter(skill_name__iexact=skill["name"]).first()
                            if not skill_obj:
                                skill_obj = Skill.objects.filter(skill_name__icontains=skill["name"]).first()
                            
                            if skill_obj:
                                skill["definition"] = skill_obj.skill_definition or "Definition not available"
                                skill["classification"] = skill_obj.classification or skill.get("classification", "Unknown")
                                if not skill.get("id"):
                                    skill["id"] = skill_obj.id

                        skill.setdefault("definition", "Definition not available")
                        skill.setdefault("classification", "Unknown")
                        skill.setdefault("id", None)
                        skill.setdefault("name", "Unknown Skill")
                                
                    except Exception as e:
                        print(f"⚠️ Error fetching skill data for '{skill.get('name', 'Unknown')}': {e}")

                # Attach demand and risk metadata
                enhanced_skills = []
                for skill in final_skills:
                    try:
                        skill_data = {
                            "id": skill.get("id"),
                            "name": skill.get("name", "Unknown Skill"),
                            "definition": skill.get("definition", "Definition not available"),
                            "classification": skill.get("classification", "Unknown"),
                            "demand": None, 
                            "demand_rationale": None,
                            "risk": None, 
                            "risk_rationale": None
                        }

                        try:
                            demand_data = SkillsDemand.objects.filter(skill__iexact=skill_data["name"]).first()
                            if demand_data:
                                skill_data.update({
                                    "demand": demand_data.demand,
                                    "demand_rationale": demand_data.demand_rationale,
                                    "risk": demand_data.risk,
                                    "risk_rationale": demand_data.risk_rationale
                                })
                        except Exception as demand_error:
                            print(f"⚠️ Demand data error for '{skill_data['name']}': {demand_error}")

                        enhanced_skills.append(skill_data)
                        
                    except Exception as e:
                        print(f"⚠️ Error creating skill data for {skill}: {e}")

                # Remove duplicates
                seen_skills = set()
                deduplicated_skills = []
                
                for skill in enhanced_skills:
                    try:
                        skill_identifier = (skill.get("name", "").lower(), skill.get("id"))
                        
                        if skill_identifier not in seen_skills:
                            seen_skills.add(skill_identifier)
                            deduplicated_skills.append(skill)
                    except Exception as dedup_error:
                        print(f"⚠️ Error deduplicating skill {skill}: {dedup_error}")
                        deduplicated_skills.append(skill)
                
                return deduplicated_skills

            # Extract skills from both documents
            skills_a = extract_skills_from_document(document_a_text, "Document A")
            skills_b = extract_skills_from_document(document_b_text, "Document B")

            # Group skills by classification for both documents
            def group_by_classification(skills):
                classification_mapping = {
                    "Highly Specialized Technical Skills": [],
                    "Core Job Skills": [],
                    "Industry-Specific Job Skills": [],
                    "General Professional Skills (Soft Skills)": [],
                    "Specialized Industry Concepts": [],
                    "Other": []
                }
                
                for skill in skills:
                    classification = skill.get("classification", "Unknown")
                    if classification in classification_mapping:
                        classification_mapping[classification].append(skill)
                    else:
                        classification_mapping["Other"].append(skill)
                
                # Remove empty classifications
                return {k: v for k, v in classification_mapping.items() if v}

            classified_skills_a = group_by_classification(skills_a)
            classified_skills_b = group_by_classification(skills_b)

            # Extract skill IDs for overlap analysis
            skill_ids_a = [skill['id'] for skill in skills_a if skill.get('id')]
            skill_ids_b = [skill['id'] for skill in skills_b if skill.get('id')]

            # ADD THIS: Calculate match percentage using only skills with IDs
            overall_match_percentage = 0
            
            # Perform overlap analysis if both documents have skills
            overlap_results = {}
            if skill_ids_a and skill_ids_b:
                print(f"\n🔄 Starting overlap analysis")
                print(f"📊 Document A has {len(skill_ids_a)} skills with IDs")
                print(f"📊 Document B has {len(skill_ids_b)} skills with IDs")
                
                # Convert to sets for processing
                position_a_set = set(str(skill_id) for skill_id in skill_ids_a)
                position_b_set = set(str(skill_id) for skill_id in skill_ids_b)
                
                # Initialize analyzer
                analyzer = SkillOverlapAnalyzer()
                
                # Load skill information
                all_skill_ids = list(set(skill_ids_a + skill_ids_b))
                skill_info = analyzer.load_skill_info_from_db(all_skill_ids)
                
                # Build similarity map from database
                similarity_map = analyzer.build_similarity_map_from_db(
                    position_a_set, position_b_set
                )
                
                # Find overlaps in both directions
                a_to_b_results, a_to_b_matches, a_to_b_similarity_sum = analyzer.find_skill_overlap(
                    position_a_set, position_b_set, similarity_map, "Document A→Document B", skill_info
                )
                
                b_to_a_results, b_to_a_matches, b_to_a_similarity_sum = analyzer.find_skill_overlap(
                    position_b_set, position_a_set, similarity_map, "Document B→Document A", skill_info
                )
                
                # ADD THIS: Calculate overall match percentage (B to A - Job Description skills matched with Resume)
                # Using the same default threshold as frontend (50%)
                SIMILARITY_THRESHOLD = 0.5
                matched_with_threshold = 0
                
                for result in b_to_a_results:
                    if result[1] and result[2] >= SIMILARITY_THRESHOLD:  # Has a match and meets threshold
                        matched_with_threshold += 1
                
                # Calculate match percentage using filtered matches
                overall_match_percentage = round((matched_with_threshold / len(skill_ids_b) * 100)) if skill_ids_b else 0
                
                # Calculate similarity metrics
                jaccard_similarity = analyzer.calculate_jaccard_similarity(
                    a_to_b_results, b_to_a_results, position_a_set, position_b_set
                )
                
                weighted_similarity = analyzer.calculate_weighted_jaccard(
                    a_to_b_results, b_to_a_results, position_a_set, position_b_set
                )
                
                # Calculate percentages
                a_match_percentage = (a_to_b_matches / len(position_a_set) * 100) if position_a_set else 0
                b_match_percentage = (b_to_a_matches / len(position_b_set) * 100) if position_b_set else 0
                
                # Collect all matches
                all_matches_a_to_b = []
                all_matches_b_to_a = []
                
                for result in a_to_b_results:
                    if result[1] and result[2] > 0:  # Has a match
                        all_matches_a_to_b.append({
                            "from_document_a": result[3],
                            "to_document_b": result[5],
                            "similarity": round(result[2], 2),
                            "direction": "A_to_B"
                        })
                
                for result in b_to_a_results:
                    if result[1] and result[2] > 0:  # Has a match
                        all_matches_b_to_a.append({
                            "from_document_b": result[3],
                            "to_document_a": result[5],
                            "similarity": round(result[2], 2),
                            "direction": "B_to_A"
                        })
                
                overlap_results = {
                    "summary": {
                        "total_unique_skills": len(set(skill_ids_a + skill_ids_b)),
                        "common_skills": len(set(skill_ids_a) & set(skill_ids_b)),
                        "document_a_match_percentage": round(a_match_percentage, 1),
                        "document_b_match_percentage": round(b_match_percentage, 1),
                        "jaccard_similarity": round(jaccard_similarity, 3),
                        "weighted_similarity": round(weighted_similarity, 3),
                        "overall_match_score": overall_match_percentage  # ADD THIS
                    },
                    "detailed_results": {
                        "document_a_to_document_b": {
                            "matched_skills": a_to_b_matches,
                            "unmatched_skills": len(skill_ids_a) - a_to_b_matches,
                            "match_percentage": round(a_match_percentage, 1)
                        },
                        "document_b_to_document_a": {
                            "matched_skills": b_to_a_matches,
                            "unmatched_skills": len(skill_ids_b) - b_to_a_matches,
                            "match_percentage": round(b_match_percentage, 1),
                            "matched_with_threshold": matched_with_threshold  # ADD THIS
                        }
                    },
                    "all_matches": {
                        "a_to_b": all_matches_a_to_b,
                        "b_to_a": all_matches_b_to_a
                    }
                }

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000

            # Build final response - FILTER OUT SKILLS WITHOUT IDs
            response_data = {
                "document_a": {
                    "skills": group_by_classification([s for s in skills_a if s.get('id')]),  # FILTER HERE
                    "total_count": len([s for s in skills_a if s.get('id')])  # COUNT ONLY WITH IDs
                },
                "document_b": {
                    "skills": group_by_classification([s for s in skills_b if s.get('id')]),  # FILTER HERE
                    "total_count": len([s for s in skills_b if s.get('id')])  # COUNT ONLY WITH IDs
                },
                "overlap_analysis": overlap_results if overlap_results else {
                    "error": "No overlap analysis performed - one or both documents have no skills with IDs"
                },
                "processing_time_ms": round(processing_time, 2)
            }

            print(f"\n🎉 Extraction and analysis completed in {processing_time:.2f}ms")
            print(f"📊 Document A: {len(skills_a)} total skills, {len(skill_ids_a)} with IDs")
            print(f"📊 Document B: {len(skills_b)} total skills, {len(skill_ids_b)} with IDs")
            if overlap_results:
                print(f"🔄 Overlap analysis complete")
                print(f"📊 Overall match score: {overall_match_percentage}%")

            return JsonResponse(response_data, safe=False)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            traceback_str = traceback.format_exc()
            print("EXTRACT AND ANALYZE ERROR TRACEBACK:\n", traceback_str)
            return JsonResponse({"error": f"Processing error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Only POST allowed"}, status=405)


# BATCH RESUME ANALYSIS

def extract_skills_from_document_for_batch(text, doc_name, exact, std, lemma, qcw, classes):
    """
    Extract skills from a document for batch processing
    (Simplified version without overlap analysis)
    """
    print(f"\n🔍 Extracting skills from {doc_name}")
    
    # Extract skills from parenthetical lists and special patterns
    parenthetical_skills = extract_parenthetical_skills(text)
    print(f"📝 Found {len(parenthetical_skills)} parenthetical/pattern skills in {doc_name}")

    # Add parenthetical skills to the matches if they exist in our database
    verified_parenthetical_matches = []
    for skill_text, start, end in parenthetical_skills:
        skill_norm = skill_text.lower().strip()
        
        # Handle & splitting
        if '&' in skill_norm:
            parts = [part.strip() for part in skill_norm.split('&')]
            print(f"🔄 SPLITTING COMPOUND SKILL: '{skill_text}' -> {parts}")
            
            for part in parts:
                if len(part) > 1:
                    skill_info = get_exact_skill_from_database(part, exact, std, lemma)
                    
                    if skill_info:
                        if validate_skill_in_text(skill_info["name"], text, start, end):
                            verified_parenthetical_matches.append((skill_info, start, end, part))
                            print(f"✅ COMPOUND PART VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')})")
        else:
            skill_info = get_exact_skill_from_database(skill_text, exact, std, lemma)
            
            if skill_info:
                if validate_skill_in_text(skill_info["name"], text, start, end):
                    verified_parenthetical_matches.append((skill_info, start, end, skill_text))
                    print(f"✅ SINGLE SKILL VERIFIED: Found '{skill_info['name']}' (ID: {skill_info.get('id')})")

    # Continue with regular tokenization and matching
    tokens = tokenize_with_indices(text)
    segments = get_contiguous_segments(tokens)

    matcher = TrieMatcher(
        exact=exact, std=std, lemma=lemma, is_std=False,
        qcw=qcw, classes=classes, full_text=text
    )

    all_matches_with_positions = []
    used_positions = {}
    matched_phrases = set()

    for segment in segments:
        matches = matcher.find_matches(segment, used_positions, matched_phrases)
        for m in matches:
            start_pos = m["start"]
            end_pos = m["end"]
            actual_text_span = text[start_pos:end_pos].lower().strip()
            
            skill_name = m["skill"]
            
            if not validate_skill_in_text(skill_name, text, start_pos, end_pos):
                continue
            
            if '&' in skill_name:
                parts = [part.strip() for part in skill_name.split('&')]
                
                if validate_skill_in_text(skill_name, text, start_pos, end_pos):
                    skill_info_dict = {
                        "name": skill_name,
                        "definition": "Definition not available",
                        "id": m.get("id"),
                        "classification": m.get("classification", "Unknown")
                    }
                    all_matches_with_positions.append((
                        skill_info_dict,
                        m["start"], m["end"], skill_name
                    ))
            else:
                exact_skill_info = get_exact_skill_from_database(actual_text_span, exact, std, lemma)
                
                if exact_skill_info and validate_skill_in_text(exact_skill_info["name"], text, start_pos, end_pos):
                    all_matches_with_positions.append((
                        exact_skill_info,
                        m["start"], m["end"], actual_text_span
                    ))

    # Combine regular matches with validated parenthetical matches
    all_matches_with_positions.extend(verified_parenthetical_matches)
    
    # Remove overlapping matches
    final_skills = remove_overlapping_skills(all_matches_with_positions)
    
    # Final validation
    truly_final_skills = []
    for skill in final_skills:
        skill_name = skill.get("name", "")
        if validate_skill_in_text(skill_name, text):
            truly_final_skills.append(skill)
    
    final_skills = truly_final_skills
    print(f"✅ Final skills for {doc_name}: {len(final_skills)}")

    # Add definitions and metadata
    for skill in final_skills:
        try:
            if "definition" not in skill or skill["definition"] == "Definition not available":
                skill_obj = Skill.objects.filter(skill_name__iexact=skill["name"]).first()
                if not skill_obj:
                    skill_obj = Skill.objects.filter(skill_name__icontains=skill["name"]).first()
                
                if skill_obj:
                    skill["definition"] = skill_obj.skill_definition or "Definition not available"
                    skill["classification"] = skill_obj.classification or skill.get("classification", "Unknown")
                    if not skill.get("id"):
                        skill["id"] = skill_obj.id

            skill.setdefault("definition", "Definition not available")
            skill.setdefault("classification", "Unknown")
            skill.setdefault("id", None)
            skill.setdefault("name", "Unknown Skill")
                    
        except Exception as e:
            print(f"⚠️ Error fetching skill data for '{skill.get('name', 'Unknown')}': {e}")

    # Attach demand and risk metadata
    enhanced_skills = []
    for skill in final_skills:
        try:
            skill_data = {
                "id": skill.get("id"),
                "name": skill.get("name", "Unknown Skill"),
                "definition": skill.get("definition", "Definition not available"),
                "classification": skill.get("classification", "Unknown"),
                "demand": None, 
                "demand_rationale": None,
                "risk": None, 
                "risk_rationale": None
            }

            try:
                demand_data = SkillsDemand.objects.filter(skill__iexact=skill_data["name"]).first()
                if demand_data:
                    skill_data.update({
                        "demand": demand_data.demand,
                        "demand_rationale": demand_data.demand_rationale,
                        "risk": demand_data.risk,
                        "risk_rationale": demand_data.risk_rationale
                    })
            except Exception as demand_error:
                print(f"⚠️ Demand data error for '{skill_data['name']}': {demand_error}")

            enhanced_skills.append(skill_data)
            
        except Exception as e:
            print(f"⚠️ Error creating skill data for {skill}: {e}")

    # Remove duplicates
    seen_skills = set()
    deduplicated_skills = []
    
    for skill in enhanced_skills:
        try:
            skill_identifier = (skill.get("name", "").lower(), skill.get("id"))
            
            if skill_identifier not in seen_skills:
                seen_skills.add(skill_identifier)
                deduplicated_skills.append(skill)
        except Exception as dedup_error:
            print(f"⚠️ Error deduplicating skill {skill}: {dedup_error}")
            deduplicated_skills.append(skill)
    
    return deduplicated_skills

@csrf_exempt
def batch_resume_analysis(request):
    """
    Batch process multiple resumes against a single job description
    
    Expected POST body (multipart/form-data):
    - job_description: single file (PDF/DOCX/TXT)
    - resumes: multiple files (PDF/DOCX/TXT) - max 10
    """
    if request.method == "POST":
        start_time = time.time()
        
        try:
            # Validate files
            if 'job_description' not in request.FILES:
                return JsonResponse({"error": "Job description file is required"}, status=400)
            
            if 'resumes' not in request.FILES:
                return JsonResponse({"error": "At least one resume file is required"}, status=400)
            
            job_description_file = request.FILES['job_description']
            resume_files = request.FILES.getlist('resumes')
            
            # Validate number of resumes
            if len(resume_files) > 10:
                return JsonResponse({"error": "Maximum 10 resumes allowed"}, status=400)
            
            if len(resume_files) == 0:
                return JsonResponse({"error": "At least one resume is required"}, status=400)
            
            # Helper function to extract text from files
            def extract_text_from_file(file):
                """Extract text from PDF, DOCX, or TXT files"""
                file_extension = os.path.splitext(file.name)[1].lower()
                
                try:
                    if file_extension == '.pdf':
                        # Save temporarily and extract PDF text
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            for chunk in file.chunks():
                                tmp_file.write(chunk)
                            tmp_file_path = tmp_file.name
                        
                        text = ""
                        with open(tmp_file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                        
                        os.remove(tmp_file_path)
                        return text.strip()
                    
                    elif file_extension == '.docx':
                        # Extract DOCX text
                        doc = docx.Document(file)
                        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                        return text.strip()
                    
                    elif file_extension == '.txt':
                        # Extract plain text
                        return file.read().decode('utf-8', errors='ignore').strip()
                    
                    else:
                        raise ValueError(f"Unsupported file format: {file_extension}")
                        
                except Exception as e:
                    print(f"Error extracting text from {file.name}: {str(e)}")
                    raise
            
            # Extract job description text
            print(f"📄 Extracting text from job description: {job_description_file.name}")
            jd_text = extract_text_from_file(job_description_file)
            
            if not jd_text.strip():
                return JsonResponse({"error": "Job description file is empty"}, status=400)
            
            # Create batch analysis record
            batch_analysis = BatchResumeAnalysis.objects.create(
                job_description_filename=job_description_file.name,
                job_description_text=jd_text,
                total_resumes=len(resume_files),
                status='processing'
            )
            
            # Extract skills from job description once
            print(f"🔍 Extracting skills from job description")
            exact, std, lemma, qcw, classes = load_skills_from_db()
            
            # Use the existing extraction logic
            jd_skills = extract_skills_from_document_for_batch(jd_text, "Job Description", exact, std, lemma, qcw, classes)
            jd_skill_ids = [skill['id'] for skill in jd_skills if skill.get('id')]
            
            print(f"✅ Found {len(jd_skills)} skills in job description")
            
            # Process each resume
            results = []
            for idx, resume_file in enumerate(resume_files):
                try:
                    print(f"\n📄 Processing resume {idx + 1}/{len(resume_files)}: {resume_file.name}")
                    resume_start_time = time.time()
                    
                    # Extract resume text
                    resume_text = extract_text_from_file(resume_file)
                    
                    if not resume_text.strip():
                        # Create error result
                        ResumeMatchResult.objects.create(
                            batch_analysis=batch_analysis,
                            resume_filename=resume_file.name,
                            resume_text="",
                            match_score=0,
                            skills_matched=0,
                            total_skills_in_jd=len(jd_skills),
                            error_message="Resume file is empty"
                        )
                        results.append({
                            "filename": resume_file.name,
                            "match_score": 0,
                            "error": "Resume file is empty"
                        })
                        continue
                    
                    # Extract skills from resume
                    resume_skills = extract_skills_from_document_for_batch(
                        resume_text, f"Resume {idx + 1}", exact, std, lemma, qcw, classes
                    )
                    resume_skill_ids = [skill['id'] for skill in resume_skills if skill.get('id')]
                    
                    # Calculate match score using existing overlap logic
                    match_score = 0
                    skills_matched = 0
                    match_details = {}
                    
                    if jd_skill_ids and resume_skill_ids:
                        # Use simplified matching logic
                        jd_set = set(str(skill_id) for skill_id in jd_skill_ids)
                        resume_set = set(str(skill_id) for skill_id in resume_skill_ids)
                        
                        # Initialize analyzer
                        analyzer = SkillOverlapAnalyzer()
                        
                        # Build similarity map
                        similarity_map = analyzer.build_similarity_map_from_db(jd_set, resume_set)
                        
                        # Find matches from JD to Resume
                        skill_info = analyzer.load_skill_info_from_db(jd_skill_ids + resume_skill_ids)
                        jd_to_resume_results, matched_count, similarity_sum = analyzer.find_skill_overlap(
                            jd_set, resume_set, similarity_map, "JD→Resume", skill_info
                        )
                        
                        # Calculate match score (percentage of JD skills found in resume)
                        skills_matched = matched_count
                        match_score = (matched_count / len(jd_skill_ids) * 100) if jd_skill_ids else 0
                        
                        # Store detailed match information
                        match_details = {
                            "jd_skills_count": len(jd_skills),
                            "resume_skills_count": len(resume_skills),
                            "matched_skills_count": matched_count,
                            "match_percentage": round(match_score, 2)
                        }
                    
                    resume_processing_time = (time.time() - resume_start_time) * 1000
                    
                    # Save result to database
                    ResumeMatchResult.objects.create(
                        batch_analysis=batch_analysis,
                        resume_filename=resume_file.name,
                        resume_text=resume_text[:5000],  # Store first 5000 chars
                        match_score=round(match_score, 2),
                        skills_matched=skills_matched,
                        total_skills_in_jd=len(jd_skills),
                        processing_time_ms=round(resume_processing_time, 2),
                        match_details=match_details
                    )
                    
                    results.append({
                        "filename": resume_file.name,
                        "match_score": round(match_score, 2)
                    })
                    
                    print(f"✅ Resume processed: {match_score:.2f}% match")
                    
                except Exception as e:
                    print(f"❌ Error processing resume {resume_file.name}: {str(e)}")
                    
                    # Create error result
                    ResumeMatchResult.objects.create(
                        batch_analysis=batch_analysis,
                        resume_filename=resume_file.name,
                        resume_text="",
                        match_score=0,
                        skills_matched=0,
                        total_skills_in_jd=len(jd_skills),
                        error_message=str(e)
                    )
                    
                    results.append({
                        "filename": resume_file.name,
                        "match_score": 0,
                        "error": str(e)
                    })
            
            # Update batch analysis status
            batch_analysis.processed_resumes = len(results)
            batch_analysis.status = 'completed'
            batch_analysis.save()
            
            total_processing_time = (time.time() - start_time) * 1000
            
            # Sort results by match score (descending)
            results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            response_data = {
                "batch_id": batch_analysis.id,
                "job_description_file": job_description_file.name,
                "total_resumes": len(resume_files),
                "processed_resumes": len(results),
                "results": results,
                "processing_time_ms": round(total_processing_time, 2)
            }
            
            print(f"\n🎉 Batch analysis completed in {total_processing_time:.2f}ms")
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"❌ Batch analysis error: {str(e)}")
            traceback_str = traceback.format_exc()
            print("ERROR TRACEBACK:\n", traceback_str)
            
            # Update batch status if it exists
            if 'batch_analysis' in locals():
                batch_analysis.status = 'failed'
                batch_analysis.save()
            
            return JsonResponse({"error": f"Processing error: {str(e)}"}, status=500)
    
    return JsonResponse({"error": "Only POST allowed"}, status=405)
