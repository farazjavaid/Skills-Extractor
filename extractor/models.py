# extractor/models.py
from django.db import models
from django.contrib.postgres.fields import ArrayField


# 1- SKILLS MODEL WITH DEFINITION
class Skill(models.Model):
    """
    Model to store skills with their metadata
    """
    # ID will be auto-incremented by default (Django's primary key)
    skill_name = models.CharField(
        max_length=255, 
        unique=True,
        help_text="Primary skill name"
    )
    
    lemmatized_skills = ArrayField(
        models.CharField(max_length=255),
        default=list,
        blank=True,
        help_text="Array of lemmatized variations of the skill"
    )
    
    skill_definition = models.TextField(
        blank=True,
        help_text="Definition or description of the skill"
    )
    
    classification = models.CharField(
        max_length=100,
        blank=True,
        help_text="Skill classification/category"
    )
    
    classification_explanation = models.TextField(
        blank=True,
        help_text="Explanation of why this skill belongs to this classification"
    )
    
    suggested_variations = ArrayField(
        models.CharField(max_length=255),
        default=list,
        blank=True,
        help_text="Array of suggested skill variations"
    )
    
    # Additional metadata fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['skill_name']
        verbose_name = "Skill"
        verbose_name_plural = "Skills"
        indexes = [
            models.Index(fields=['skill_name']),
            models.Index(fields=['classification']),
        ]
    
    def __str__(self):
        return self.skill_name
    
    def get_all_variations(self):
        """
        Get all variations of this skill (lemmatized + suggested)
        """
        variations = set(self.lemmatized_skills + self.suggested_variations)
        variations.add(self.skill_name)
        return list(variations)
    
# 2- SKILLS DEMAND MODEL WITH DEFINITION
class SkillsDemand(models.Model):
    """Skills demand data with 5 columns from Excel"""
    
    skill = models.CharField(max_length=255, help_text="Skill name")
    demand = models.CharField(max_length=100, blank=True, help_text="Demand level")
    demand_rationale = models.TextField(blank=True, help_text="Demand explanation")
    risk = models.CharField(max_length=100, blank=True, help_text="Risk level")
    risk_rationale = models.TextField(blank=True, help_text="Risk explanation")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'skills_demand'
        ordering = ['skill']
        unique_together = ['skill']  # One record per skill
    
    def __str__(self):
        return f"{self.skill} - {self.demand}"
    
# 3- FOLDERS
class Folder(models.Model):
    """
    Model to store folder information
    """
    name = models.CharField(
        max_length=255,
        help_text="Folder name"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        verbose_name = "Folder"
        verbose_name_plural = "Folders"
        indexes = [
            models.Index(fields=['name']),
        ]
    
    def __str__(self):
        return self.name

# 4- SKILL FILES
class File(models.Model):
    """
    Model to store file information with skill profiles
    """
    filename = models.CharField(
        max_length=255,
        help_text="Name of the file"
    )
    
    # Foreign key relationship to Folder
    folder = models.ForeignKey(
        Folder,
        on_delete=models.CASCADE,
        related_name='files',
        help_text="Folder this file belongs to"
    )
    
    # JSONB field for skill profile data
    skill_profile = models.JSONField(
        default=dict,
        blank=True,
        help_text="Skill profile data in JSON format"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['filename']
        verbose_name = "File"
        verbose_name_plural = "Files"
        indexes = [
            models.Index(fields=['filename']),
            models.Index(fields=['folder']),
        ]
        # Ensure unique filename within each folder
        unique_together = ['filename', 'folder']
    
    def __str__(self):
        return f"{self.folder.name}/{self.filename}"
    
    @property
    def full_path(self):
        """Get the full path of the file"""
        return f"{self.folder.name}/{self.filename}"
    
# SKILLS BUILDER
class SkillsJobTitles(models.Model):
    """
    Model to store skills with their associated job titles
    """
    skill = models.CharField(
        max_length=500,  # Some skill names might be long
        help_text="Skill name"
    )
    
    skill_category = models.CharField(
        max_length=100,
        help_text="Category of the skill (e.g., Software, Skill, etc.)"
    )
    
    job_title_1 = models.CharField(
        max_length=200,
        help_text="Primary job title associated with this skill"
    )
    
    job_title_2 = models.CharField(
        max_length=200,
        help_text="Second job title associated with this skill"
    )
    
    job_title_3 = models.CharField(
        max_length=200,
        help_text="Third job title associated with this skill"
    )
    
    job_title_4 = models.CharField(
        max_length=200,
        help_text="Fourth job title associated with this skill"
    )
    
    job_title_5 = models.CharField(
        max_length=200,
        help_text="Fifth job title associated with this skill"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'skills_job_titles'
        ordering = ['skill']
        verbose_name = "Skills Job Title"
        verbose_name_plural = "Skills Job Titles"
        indexes = [
            models.Index(fields=['skill']),
            models.Index(fields=['skill_category']),
            models.Index(fields=['job_title_1']),
        ]
    
    def __str__(self):
        return f"{self.skill} - {self.skill_category}"
    
    def get_all_job_titles(self):
        """
        Get all job titles as a list
        """
        return [
            self.job_title_1,
            self.job_title_2,
            self.job_title_3,
            self.job_title_4,
            self.job_title_5
        ]
    
    def get_unique_job_titles(self):
        """
        Get unique job titles (in case there are duplicates)
        """
        titles = self.get_all_job_titles()
        return list(dict.fromkeys(titles))  # Preserves order while removing duplicates
    
# Add this to your extractor/models.py file

class SkillJobFunction(models.Model):
    """
    Model to store skills with their job functions, industries, and related skills
    """
    skill = models.CharField(
        max_length=500,
        help_text="Skill name"
    )
    
    skill_type = models.CharField(
        max_length=100,
        help_text="Type of skill (e.g., Software, Document, etc.)"
    )
    
    primary_industry = models.CharField(
        max_length=200,
        help_text="Primary industry associated with this skill"
    )
    
    secondary_industry = models.CharField(
        max_length=200,
        blank=True,
        help_text="Secondary industry associated with this skill"
    )
    
    primary_job_function = models.CharField(
        max_length=200,
        help_text="Primary job function associated with this skill"
    )
    
    secondary_job_function = models.CharField(
        max_length=200,
        blank=True,
        help_text="Secondary job function associated with this skill"
    )
    
    related_skill_1 = models.CharField(
        max_length=300,
        blank=True,
        help_text="First related skill"
    )
    
    related_skill_2 = models.CharField(
        max_length=300,
        blank=True,
        help_text="Second related skill"
    )
    
    related_skill_3 = models.CharField(
        max_length=300,
        blank=True,
        help_text="Third related skill"
    )
    
    skill_definition = models.TextField(
        blank=True,
        help_text="Definition or description of the skill"
    )
    
    number = models.IntegerField(
        null=True,
        blank=True,
        help_text="Sequential number/ID from the original data"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'skill_job_function'
        ordering = ['skill']
        verbose_name = "Skill Job Function"
        verbose_name_plural = "Skill Job Functions"
        indexes = [
            models.Index(fields=['skill']),
            models.Index(fields=['skill_type']),
            models.Index(fields=['primary_industry']),
            models.Index(fields=['primary_job_function']),
            models.Index(fields=['number']),
        ]
    
    def __str__(self):
        return f"{self.skill} - {self.primary_job_function}"
    
    def get_all_related_skills(self):
        """
        Get all related skills as a list (excluding empty ones)
        """
        related_skills = [
            self.related_skill_1,
            self.related_skill_2,
            self.related_skill_3
        ]
        return [skill for skill in related_skills if skill and skill.strip()]
    
    def get_all_industries(self):
        """
        Get all industries as a list (excluding empty ones)
        """
        industries = [self.primary_industry]
        if self.secondary_industry and self.secondary_industry.strip():
            industries.append(self.secondary_industry)
        return industries
    
    def get_all_job_functions(self):
        """
        Get all job functions as a list (excluding empty ones)
        """
        job_functions = [self.primary_job_function]
        if self.secondary_job_function and self.secondary_job_function.strip():
            job_functions.append(self.secondary_job_function)
        return job_functions
    
# Add this to your existing models.py file

class SkillSimilarity(models.Model):
    """
    Model to store skill similarity data with thresholds
    """
    skill_id = models.IntegerField(
        help_text="ID of the primary skill"
    )
    
    # Store the complete thresholds data as JSON
    similarity_thresholds = models.JSONField(
        default=dict,
        help_text="Dictionary of similarity thresholds and their associated skill IDs"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'skill_similarities'
        ordering = ['skill_id']
        verbose_name = "Skill Similarity"
        verbose_name_plural = "Skill Similarities"
        indexes = [
            models.Index(fields=['skill_id']),
        ]
        # Ensure one record per skill_id
        unique_together = ['skill_id']
    
    def __str__(self):
        return f"Skill {self.skill_id} - {len(self.similarity_thresholds)} thresholds"
    
    def get_similar_skills_at_threshold(self, threshold):
        """Get all similar skills at a specific threshold"""
        threshold_str = str(threshold)
        return self.similarity_thresholds.get(threshold_str, [])


# Add these to your existing models.py

class BatchResumeAnalysis(models.Model):
    """Stores batch resume analysis session information"""
    job_description_filename = models.CharField(max_length=255)
    job_description_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    total_resumes = models.IntegerField(default=0)
    processed_resumes = models.IntegerField(default=0)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    
    class Meta:
        db_table = 'batch_resume_analysis'
        ordering = ['-created_at']

class ResumeMatchResult(models.Model):
    """Stores individual resume match results"""
    batch_analysis = models.ForeignKey(
        BatchResumeAnalysis, 
        on_delete=models.CASCADE, 
        related_name='resume_results'
    )
    resume_filename = models.CharField(max_length=255)
    resume_text = models.TextField()
    match_score = models.FloatField()
    skills_matched = models.IntegerField(default=0)
    total_skills_in_jd = models.IntegerField(default=0)
    processing_time_ms = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Store detailed match data as JSON
    match_details = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'resume_match_results'
        ordering = ['-match_score']