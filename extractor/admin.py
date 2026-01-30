# extractor/admin.py
from django.contrib import admin
from .models import Skill, SkillsDemand, Folder, File


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = [
        'id', 
        'skill_name', 
        'classification', 
        'created_at'
    ]
    list_filter = [
        'classification', 
        'created_at'
    ]
    search_fields = [
        'skill_name', 
        'skill_definition', 
        'classification'
    ]
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('skill_name', 'skill_definition')
        }),
        ('Classification', {
            'fields': ('classification', 'classification_explanation')
        }),
        ('Variations', {
            'fields': ('lemmatized_skills', 'suggested_variations')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        """Optimize queries"""
        return super().get_queryset(request).select_related()

# Update your extractor/admin.py - replace the SkillsDemandAdmin with this:

@admin.register(SkillsDemand)
class SkillsDemandAdmin(admin.ModelAdmin):
    list_display = ['skill', 'demand', 'risk', 'created_at']
    list_filter = ['demand', 'risk']
    search_fields = ['skill', 'demand_rationale', 'risk_rationale']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(Folder)
class FolderAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'files_count', 'created_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at']
    
    def files_count(self, obj):
        """Show number of files in this folder"""
        return obj.files.count()
    files_count.short_description = 'Files Count'
    
    fieldsets = (
        ('Folder Information', {
            'fields': ('name',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(File)
class FileAdmin(admin.ModelAdmin):
    list_display = ['id', 'filename', 'folder', 'has_skill_profile', 'created_at']
    list_filter = ['folder', 'created_at', 'updated_at']
    search_fields = ['filename', 'folder__name']
    readonly_fields = ['created_at', 'updated_at', 'full_path']
    
    def has_skill_profile(self, obj):
        """Check if file has skill profile data"""
        return bool(obj.skill_profile)
    has_skill_profile.boolean = True
    has_skill_profile.short_description = 'Has Profile'
    
    fieldsets = (
        ('File Information', {
            'fields': ('filename', 'folder', 'full_path')
        }),
        ('Skill Profile', {
            'fields': ('skill_profile',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    # Show files grouped by folder
    list_select_related = ['folder']