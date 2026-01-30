import logging

from .models import Skill
from .extractorv6 import build_tries_from_skills, extract_quick_check_words

log = logging.getLogger(__name__)

def load_skills_from_db():
    """Load skills from database and build optimized tries with proper ID handling"""
    try:
        skills = Skill.objects.all()

        exact_skills = {}
        standardized_skills = {}
        lemmatized_skills = {}
        classification = {}
        skill_ids = {}

        log.info(f"Loading {skills.count()} skills from database...")

        for skill in skills:
            name = skill.skill_name.strip()
            name_lower = name.lower()

            # Store classification and ID mappings
            classification[name_lower] = skill.classification or "Unknown"
            skill_ids[name_lower] = skill.id

            # Exact skill with all required fields including ID
            exact_skills[name_lower] = {
                "name": name,
                "classification": skill.classification or "Unknown",
                "definition": skill.skill_definition or "",
                "id": skill.id,
            }

            # Variations (standardized)
            for var in skill.suggested_variations:
                if var and var.strip():  # Check that variation is not empty
                    var_lower = var.strip().lower()
                    standardized_skills[var_lower] = {
                        "name": name,  # Original skill name
                        "classification": skill.classification or "Unknown",
                        "definition": skill.skill_definition or "",
                        "id": skill.id,
                    }
                    # Also add to classification and skill_ids mapping
                    classification[var_lower] = skill.classification or "Unknown"
                    skill_ids[var_lower] = skill.id

            # Lemmatized variations
            for lemma in skill.lemmatized_skills:
                if lemma and lemma.strip():  # Check that lemma is not empty
                    lemma_lower = lemma.strip().lower()
                    lemmatized_skills[lemma_lower] = {
                        "name": name,  # Original skill name
                        "classification": skill.classification or "Unknown",
                        "definition": skill.skill_definition or "",
                        "id": skill.id,
                    }
                    # Also add to classification and skill_ids mapping
                    classification[lemma_lower] = skill.classification or "Unknown"
                    skill_ids[lemma_lower] = skill.id

        # Build the data structure for the trie builder
        data = {
            "exact_skills": exact_skills,
            "standardized_skills": standardized_skills,
            "lemmatized_skills": lemmatized_skills,
            "classification": classification,
            "skill_ids": skill_ids,
            "metadata": {
                "format": "standardized",
                "total_skills": len(exact_skills),
                "total_variations": len(standardized_skills),
                "total_lemmas": len(lemmatized_skills),
            },
        }

        log.info(
            f"Prepared skill data: "
            f"{len(exact_skills)} exact, {len(standardized_skills)} standardized, {len(lemmatized_skills)} lemmatized"
        )

        # Build tries
        exact, std, lemma = build_tries_from_skills(data)
        qcw = extract_quick_check_words(exact, std, lemma)

        log.info(
            f"Built tries successfully - exact: {len(exact)}, std: {len(std)}, lemma: {len(lemma)}"
        )
        log.info(f"Generated {len(qcw)} quick check words")

        return exact, std, lemma, qcw, classification

    except Exception as e:
        log.error(f"Error loading skills from database: {e}")
        # Return empty structures as fallback
        from pygtrie import StringTrie

        return StringTrie(), StringTrie(), StringTrie(), set(), {}
