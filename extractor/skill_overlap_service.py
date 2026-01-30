import logging
from typing import Dict, List, Tuple, Set, Optional
from django.db.models import Q
from extractor.models import Skill, SkillSimilarity
import time
from functools import lru_cache


logger = logging.getLogger(__name__)


class SkillOverlapAnalyzer:
    """Service for analyzing skill overlap between positions using database"""
    
    def __init__(self):
        self.similarity_cache = {}
        
    @lru_cache(maxsize=4096)
    def normalize_skill_id(self, skill_id):
        """Normalize a skill ID for consistent matching"""
        if not isinstance(skill_id, str):
            skill_id_str = str(skill_id)
        else:
            skill_id_str = skill_id
        
        skill_id_str = skill_id_str.strip()
        skill_id_str = skill_id_str.replace(',', '').replace('"', '').replace("'", "")
        
        return skill_id_str
    
    def load_skill_info_from_db(self, skill_ids: List[int]) -> Dict:
        """Load skill metadata from database"""
        skill_info = {}
        
        # Batch fetch skills from database
        skills = Skill.objects.filter(id__in=skill_ids).values(
            'id', 'skill_name', 'classification'
        )
        
        for skill in skills:
            skill_id_str = str(skill['id'])
            skill_info[skill_id_str] = {
                "name": skill['skill_name'],
                "classification": skill['classification'] or ""
            }
            
            # Also store with normalized ID
            normalized_id = self.normalize_skill_id(skill_id_str)
            if normalized_id != skill_id_str:
                skill_info[normalized_id] = skill_info[skill_id_str]
        
        return skill_info
    
    def build_similarity_map_from_db(self, position_a_skills: Set[str], 
                                   position_b_skills: Set[str]) -> Dict:
        """Build similarity map from database for relevant skills"""
        start_time = time.time()
        
        relevant_skills = position_a_skills.union(position_b_skills)
        normalized_skills = {self.normalize_skill_id(s) for s in relevant_skills}
        
        # Convert to integers for database query
        skill_ids = []
        for skill in normalized_skills:
            try:
                skill_ids.append(int(skill))
            except ValueError:
                logger.warning(f"Could not convert skill ID to integer: {skill}")
        
        logger.info(f"Building similarity map for {len(skill_ids)} skills")
        
        # Fetch similarities from database
        similarities = SkillSimilarity.objects.filter(
            skill_id__in=skill_ids
        ).values('skill_id', 'similarity_thresholds')
        
        similarity_map = {}
        
        for sim_record in similarities:
            skill_id = str(sim_record['skill_id'])
            normalized_id = self.normalize_skill_id(skill_id)
            
            thresholds_data = sim_record['similarity_thresholds']
            skill_entry = {}
            
            # Process each threshold
            for threshold_str, neighbors in thresholds_data.items():
                threshold = float(threshold_str)
                neighbor_list = []
                
                # Filter neighbors to only include relevant skills
                for neighbor_id in neighbors:
                    neighbor_normalized = self.normalize_skill_id(str(neighbor_id))
                    if neighbor_normalized in normalized_skills:
                        neighbor_list.append((str(neighbor_id), str(neighbor_id)))
                
                if neighbor_list:
                    skill_entry[threshold] = neighbor_list
            
            similarity_map[normalized_id] = skill_entry
        
        # Add missing skills with empty thresholds
        for skill in normalized_skills:
            if skill not in similarity_map:
                similarity_map[skill] = {}
        
        load_time = time.time() - start_time
        logger.info(f"Built similarity map with {len(similarity_map)} entries in {load_time:.3f} seconds")
        
        return similarity_map
    
    def find_skill_overlap(self, source_skills: Set[str], target_skills: Set[str], 
                          similarity_map: Dict, direction: str, skill_info: Dict) -> Tuple:
        """Find overlapping skills between source and target positions"""
        start_time = time.time()
        
        logger.info(f"Finding overlap: {direction}")
        logger.info(f"Source skills count: {len(source_skills)}")
        logger.info(f"Target skills count: {len(target_skills)}")
        
        # Convert to normalized lists
        source_skill_list = [self.normalize_skill_id(s) for s in source_skills]
        target_lookup = {self.normalize_skill_id(s): s for s in target_skills}
        
        results = []
        metrics = {
            "matched_count": 0,
            "similarity_sum": 0.0,
            "unmatched_count": 0,
            "missing_from_map_count": 0
        }
        
        for source_skill_id in sorted(source_skill_list):
            source_id_normalized = self.normalize_skill_id(source_skill_id)
            best_match = None
            
            # Check for exact match first
            if source_id_normalized in target_lookup:
                match_id = target_lookup[source_id_normalized]
                best_match = (match_id, 1.0)
            
            # If not exact match, check similarity map
            elif source_id_normalized in similarity_map:
                neighbors_dict = similarity_map[source_id_normalized]
                
                # Find best match across all thresholds
                for threshold, neighbors in sorted(neighbors_dict.items(), reverse=True):
                    for neighbor_id, _ in neighbors:
                        neighbor_normalized = self.normalize_skill_id(neighbor_id)
                        
                        if neighbor_normalized in target_lookup:
                            best_match = (target_lookup[neighbor_normalized], threshold)
                            break
                    
                    if best_match:
                        break
            
            # Record result
            if best_match:
                source_name = skill_info.get(source_skill_id, {}).get("name", source_skill_id)
                source_class = skill_info.get(source_skill_id, {}).get("classification", "")
                target_name = skill_info.get(best_match[0], {}).get("name", best_match[0])
                target_class = skill_info.get(best_match[0], {}).get("classification", "")
                
                results.append((
                    source_skill_id, best_match[0], best_match[1],
                    source_name, source_class, target_name, target_class
                ))
                metrics["matched_count"] += 1
                metrics["similarity_sum"] += best_match[1]
            else:
                source_name = skill_info.get(source_skill_id, {}).get("name", source_skill_id)
                source_class = skill_info.get(source_skill_id, {}).get("classification", "")
                
                results.append((
                    source_skill_id, "", 0,
                    source_name, source_class, "", ""
                ))
                
                if source_id_normalized in similarity_map:
                    metrics["unmatched_count"] += 1
                else:
                    metrics["missing_from_map_count"] += 1
        
        processing_time = time.time() - start_time
        logger.info(f"{direction}: Found {metrics['matched_count']} matches in {processing_time:.3f} seconds")
        
        return results, metrics["matched_count"], metrics["similarity_sum"]
    
    def calculate_jaccard_similarity(self, a_to_b_results, b_to_a_results, 
                                   position_a_skills, position_b_skills):
        """Calculate Jaccard similarity"""
        skills_a = {self.normalize_skill_id(skill) for skill in position_a_skills}
        skills_b = {self.normalize_skill_id(skill) for skill in position_b_skills}
        
        union_size = len(skills_a.union(skills_b))
        unique_matches = set()
        
        for source_id, matched_id, score, *_ in a_to_b_results:
            if matched_id and score:
                unique_matches.add((
                    self.normalize_skill_id(source_id),
                    self.normalize_skill_id(matched_id)
                ))
        
        for source_id, matched_id, score, *_ in b_to_a_results:
            if matched_id and score:
                unique_matches.add((
                    self.normalize_skill_id(matched_id),
                    self.normalize_skill_id(source_id)
                ))
        
        intersection_size = len(unique_matches)
        
        if union_size > 0:
            return intersection_size / union_size
        else:
            return 0.0
    
    def calculate_weighted_jaccard(self, a_to_b_results, b_to_a_results,
                                 position_a_skills, position_b_skills):
        """Calculate weighted Jaccard similarity"""
        skills_a = {self.normalize_skill_id(skill) for skill in position_a_skills}
        skills_b = {self.normalize_skill_id(skill) for skill in position_b_skills}
        
        union_size = len(skills_a.union(skills_b))
        weighted_matches = {}
        
        for source_id, matched_id, score, *_ in a_to_b_results:
            if matched_id and score:
                pair = (self.normalize_skill_id(source_id), 
                       self.normalize_skill_id(matched_id))
                weighted_matches[pair] = float(score)
        
        for source_id, matched_id, score, *_ in b_to_a_results:
            if matched_id and score:
                pair = (self.normalize_skill_id(matched_id),
                       self.normalize_skill_id(source_id))
                if pair in weighted_matches:
                    weighted_matches[pair] = max(weighted_matches[pair], float(score))
                else:
                    weighted_matches[pair] = float(score)
        
        weighted_sum = sum(weighted_matches.values())
        
        if union_size > 0:
            return weighted_sum / union_size
        else:
            return 0.0