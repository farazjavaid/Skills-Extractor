# extractor/management/commands/import_skills_job_titles.py

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from extractor.models import SkillsJobTitles
import pandas as pd
from django.db import transaction


class Command(BaseCommand):
    help = 'Import skills and job titles data from Excel file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Path to the Excel file',
            default='skills_output_job_title combined.xlsx'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Number of records to process in each batch'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before importing'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        batch_size = options['batch_size']
        clear_data = options['clear']

        if not os.path.exists(file_path):
            self.stdout.write(
                self.style.ERROR(f'File not found: {file_path}')
            )
            return

        try:
            # Clear existing data if requested
            if clear_data:
                self.stdout.write('Clearing existing data...')
                SkillsJobTitles.objects.all().delete()
                self.stdout.write(
                    self.style.SUCCESS('Existing data cleared.')
                )

            # Read Excel file
            self.stdout.write(f'Reading Excel file: {file_path}')
            df = pd.read_excel(file_path, sheet_name='skills_output_job_title')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Verify expected columns
            expected_columns = [
                'Skill', 'Skill Category', 'Job Title 1', 'Job Title 2', 
                'Job Title 3', 'Job Title 4', 'Job Title 5'
            ]
            
            if not all(col in df.columns for col in expected_columns):
                self.stdout.write(
                    self.style.ERROR(f'Missing columns. Expected: {expected_columns}')
                )
                self.stdout.write(f'Found: {list(df.columns)}')
                return

            # Clean data
            df = df.fillna('')  # Replace NaN with empty strings
            df = df.astype(str)  # Convert all to strings
            
            total_rows = len(df)
            self.stdout.write(f'Total rows to import: {total_rows}')

            # Process in batches
            imported_count = 0
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Prepare batch data
                skills_job_titles_batch = []
                
                for _, row in batch_df.iterrows():
                    skills_job_title = SkillsJobTitles(
                        skill=row['Skill'].strip(),
                        skill_category=row['Skill Category'].strip(),
                        job_title_1=row['Job Title 1'].strip(),
                        job_title_2=row['Job Title 2'].strip(),
                        job_title_3=row['Job Title 3'].strip(),
                        job_title_4=row['Job Title 4'].strip(),
                        job_title_5=row['Job Title 5'].strip(),
                    )
                    skills_job_titles_batch.append(skills_job_title)
                
                # Bulk create with transaction
                with transaction.atomic():
                    SkillsJobTitles.objects.bulk_create(
                        skills_job_titles_batch,
                        batch_size=batch_size
                    )
                
                imported_count += len(skills_job_titles_batch)
                self.stdout.write(
                    f'Imported {imported_count}/{total_rows} records...'
                )

            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully imported {imported_count} skills and job titles!'
                )
            )

            # Display some statistics
            total_skills = SkillsJobTitles.objects.count()
            unique_skills = SkillsJobTitles.objects.values('skill').distinct().count()
            unique_categories = SkillsJobTitles.objects.values('skill_category').distinct().count()
            
            self.stdout.write('\nImport Statistics:')
            self.stdout.write(f'- Total records: {total_skills}')
            self.stdout.write(f'- Unique skills: {unique_skills}')
            self.stdout.write(f'- Unique categories: {unique_categories}')

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error importing data: {str(e)}')
            )
            raise