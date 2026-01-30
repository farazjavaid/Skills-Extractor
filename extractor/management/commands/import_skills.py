# extractor/management/commands/import_skills.py

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from extractor.models import Skill
import ast
import re

class Command(BaseCommand):
    help = 'Import skills from Excel file'

    def add_arguments(self, parser):
        parser.add_argument(
            'excel_file',
            type=str,
            help='Path to the Excel file containing skills data'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before importing',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Number of records to process in each batch (default: 1000)',
        )

    def handle(self, *args, **options):
        excel_file = options['excel_file']
        clear_data = options['clear']
        batch_size = options['batch_size']

        try:
            # Read Excel file
            self.stdout.write(f'Reading Excel file: {excel_file}')
            df = pd.read_excel(excel_file)
            
            # Display column information
            self.stdout.write(f'Columns found: {list(df.columns)}')
            self.stdout.write(f'Total rows: {len(df)}')
            
            # Expected columns (adjust these to match your Excel file)
            expected_columns = {
                'Skill ID': 'skill_id',
                'Skill Name': 'skill_name', 
                'Lemmatized Skills': 'lemmatized_skills',
                'Skill Definition': 'skill_definition',
                'Classification': 'classification',
                'Classification Explanation': 'classification_explanation',
                'Suggested Variations': 'suggested_variations'
            }
            
            # Check if required columns exist
            missing_columns = []
            for col in expected_columns.keys():
                if col not in df.columns:
                    # Try to find similar column names
                    similar_cols = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                    if similar_cols:
                        self.stdout.write(self.style.WARNING(f'Column "{col}" not found. Similar columns: {similar_cols}'))
                    else:
                        missing_columns.append(col)
            
            if missing_columns:
                self.stdout.write(self.style.ERROR(f'Missing columns: {missing_columns}'))
                self.stdout.write('Available columns:')
                for i, col in enumerate(df.columns):
                    self.stdout.write(f'  {i+1}. {col}')
                return
            
            # Clear existing data if requested
            if clear_data:
                self.stdout.write('Clearing existing skills data...')
                Skill.objects.all().delete()
                self.stdout.write(self.style.SUCCESS('Existing data cleared.'))
            
            # Process data in batches
            skills_to_create = []
            successful_imports = 0
            failed_imports = 0
            
            self.stdout.write(f'Processing {len(df)} records in batches of {batch_size}...')
            
            for index, row in df.iterrows():
                try:
                    # Parse array fields (lemmatized_skills and suggested_variations)
                    lemmatized_skills = self.parse_array_field(row['Lemmatized Skills'])
                    suggested_variations = self.parse_array_field(row['Suggested Variations'])
                    
                    # Create skill object
                    skill = Skill(
                        skill_name=str(row['Skill Name']).strip(),
                        lemmatized_skills=lemmatized_skills,
                        skill_definition=str(row['Skill Definition']) if pd.notna(row['Skill Definition']) else '',
                        classification=str(row['Classification']) if pd.notna(row['Classification']) else '',
                        classification_explanation=str(row['Classification Explanation']) if pd.notna(row['Classification Explanation']) else '',
                        suggested_variations=suggested_variations
                    )
                    
                    skills_to_create.append(skill)
                    
                    # Batch insert
                    if len(skills_to_create) >= batch_size:
                        self.bulk_create_skills(skills_to_create)
                        successful_imports += len(skills_to_create)
                        skills_to_create = []
                        self.stdout.write(f'Processed {successful_imports} records...')
                
                except Exception as e:
                    failed_imports += 1
                    self.stdout.write(
                        self.style.WARNING(f'Failed to process row {index + 1}: {str(e)}')
                    )
            
            # Insert remaining skills
            if skills_to_create:
                self.bulk_create_skills(skills_to_create)
                successful_imports += len(skills_to_create)
            
            # Summary
            self.stdout.write(self.style.SUCCESS(
                f'Import completed! Successfully imported: {successful_imports}, Failed: {failed_imports}'
            ))
            
        except FileNotFoundError:
            raise CommandError(f'Excel file not found: {excel_file}')
        except Exception as e:
            raise CommandError(f'Error importing data: {str(e)}')
    
    def parse_array_field(self, field_value):
        """Parse array field from Excel (could be string representation of list or comma-separated)"""
        if pd.isna(field_value) or field_value == '':
            return []
        
        field_str = str(field_value).strip()
        
        # If it's already a list (string representation)
        if field_str.startswith('[') and field_str.endswith(']'):
            try:
                return ast.literal_eval(field_str)
            except:
                # If literal_eval fails, try to parse manually
                items = re.findall(r"'([^']*)'", field_str)
                return items if items else []
        
        # If it's comma-separated values
        elif ',' in field_str:
            return [item.strip().strip('"\'') for item in field_str.split(',') if item.strip()]
        
        # If it's a single value
        else:
            return [field_str] if field_str else []
    
    def bulk_create_skills(self, skills):
        """Bulk create skills with error handling"""
        try:
            with transaction.atomic():
                Skill.objects.bulk_create(skills, ignore_conflicts=True)
        except Exception as e:
            # If bulk create fails, try individual creates
            self.stdout.write(self.style.WARNING(f'Bulk create failed, trying individual creates: {str(e)}'))
            for skill in skills:
                try:
                    skill.save()
                except Exception as individual_error:
                    self.stdout.write(
                        self.style.WARNING(f'Failed to create skill "{skill.skill_name}": {str(individual_error)}')
                    )