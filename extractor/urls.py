# extractor/urls.py - Complete URLs configuration

from django.urls import path
from .views import (
    # Original skills extraction
    # count_skills_for_search_endpoint,
    analyze_profile_overlap,
    batch_resume_analysis,
    
    extract,
    extract_and_analyze_overlap,
    
    # Enhanced skills extraction with demand/risk
    extract_with_demand,
    get_industries,
    get_skill_detail,
    # get_dynamic_suggestions,
    predictive_search,
    search_skills,
    search_skills_autocomplete,
    search_skills_enhanced,
    skill_suggestions,
    
    # Skills demand APIs
    skills_demand_search, 
    skill_demand_by_name, 
    all_skills_demand,
    
    # Folder management APIs
    create_folder, 
    get_folders, 
    get_folder_by_id,
    delete_folder,
    
    # File management APIs
    create_file, 
    update_file, 
    get_file_by_id, 
    get_files,
    delete_file,

    get_all_folders_with_analytics,
    get_folder_analytics,
    get_detailed_folder_analytics,
    
    # New Skill Builder APIs
    get_skills_job_titles,
    # get_filter_options,
    # get_skills_statistics,
    # get_search_suggestions

    analyze_skill_overlap, 
    get_skill_similarities,

    get_skill_detail, 
    search_skills_autocomplete

    
)

urlpatterns = [
    # ===== SKILLS EXTRACTION APIs =====
    path('extract/', extract, name='extract_skills'),
    path('extract-with-demand/', extract_with_demand, name='extract_with_demand'),
    
    # ===== SKILLS DEMAND APIs =====
    path('skills-demand/', all_skills_demand, name='all_skills_demand'),
    path('skills-demand/search/', skills_demand_search, name='skills_demand_search'),
    path('skills-demand/<str:skill_name>/', skill_demand_by_name, name='skill_demand_by_name'),
    
    # ===== SKILL BUILDER APIs =====

    path('skill-builder/skills/', get_skills_job_titles, name='skill_builder_skills'),
    path("search-skills/",search_skills, name="search_skills"),
    path("search-skills-enhanced/",search_skills_enhanced, name="search_skills_enhanced"),
    path('predictive-search/', predictive_search, name='predictive_search'),
    path('get-industries/', get_industries, name='get_industries'),

    
    
    # ===== FOLDER MANAGEMENT APIs =====
    path('folders/', get_folders, name='get_folders'),
    path('folders/create/', create_folder, name='create_folder'),
    path('folders/<int:folder_id>/', get_folder_by_id, name='get_folder_by_id'),
    path('folders/<int:folder_id>/delete/', delete_folder, name='delete_folder'),
    
    # ===== FILE MANAGEMENT APIs =====
    path('files/', get_files, name='get_files'),
    path('files/create/', create_file, name='create_file'),
    path('files/<int:file_id>/', get_file_by_id, name='get_file_by_id'),
    path('files/<int:file_id>/update/', update_file, name='update_file'),
    path('files/<int:file_id>/delete/', delete_file, name='delete_file'),

    path('skills/suggestions/', skill_suggestions, name='skill_suggestions'),

    path('analytics/folders/', get_all_folders_with_analytics, name='get_all_folders_with_analytics'),
    path('analytics/folders/<int:folder_id>/', get_folder_analytics, name='get_folder_analytics'),
    path('analytics/detailed/', get_detailed_folder_analytics, name='get_detailed_all_folders_analytics'),
    path('analytics/detailed/<int:folder_id>/', get_detailed_folder_analytics, name='get_detailed_folder_analytics'),

    path('skills/overlap/', analyze_skill_overlap, name='analyze_skill_overlap'),
    path('skills/<int:skill_id>/similarities/', get_skill_similarities, name='get_skill_similarities'),
    path('profiles/overlap/', analyze_profile_overlap, name='analyze_profile_overlap'),
    path('document-overlap/', extract_and_analyze_overlap, name='extract_and_analyze_overlap'),

    # Skill Detail Page APIs
    path('skills/<str:skill_identifier>/detail/', get_skill_detail, name='get_skill_detail'),
    path('skills/search/autocomplete/', search_skills_autocomplete, name='search_skills_autocomplete'),

    # Add this to your urlpatterns
    path('batch-resume-analysis/', batch_resume_analysis, name='batch_resume_analysis'),
]