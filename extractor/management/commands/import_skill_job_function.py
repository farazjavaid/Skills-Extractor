# Create this file: extractor/management/commands/import_skill_job_function.py

import os
from django.core.management.base import BaseCommand
from django.conf import settings
from extractor.models import SkillJobFunction
import pandas as pd
from django.db import transaction


class Command(BaseCommand):
    help = 'Import skill job function data from Excel file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Path to the Excel file',
            default='skill_jobfunction_relatedskill53k.xlsx'
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
        parser.add_argument(
            '--sheet',
            type=str,
            default='skills_output_adv',
            help='Name of the Excel sheet to read'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        batch_size = options['batch_size']
        clear_data = options['clear']
        sheet_name = options['sheet']

        if not os.path.exists(file_path):
            self.stdout.write(
                self.style.ERROR(f'File not found: {file_path}')
            )
            return

        try:
            # Clear existing data if requested
            if clear_data:
                self.stdout.write('Clearing existing data...')
                SkillJobFunction.objects.all().delete()
                self.stdout.write(
                    self.style.SUCCESS('Existing data cleared.')
                )

            # Read Excel file
            self.stdout.write(f'Reading Excel file: {file_path}')
            self.stdout.write(f'Sheet: {sheet_name}')
            
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Verify expected columns
            expected_columns = [
                'Skill', 'Skill Type', 'Primary Industry', 'Secondary Industry',
                'Primary Job Function', 'Secondary Job Function', 'Related Skill 1',
                'Related Skill 2', 'Related Skill 3', 'Skill Definition', 'Number'
            ]
            
            if not all(col in df.columns for col in expected_columns):
                self.stdout.write(
                    self.style.ERROR(f'Missing columns. Expected: {expected_columns}')
                )
                self.stdout.write(f'Found: {list(df.columns)}')
                return

            # Clean data
            df = df.fillna('')  # Replace NaN with empty strings
            
            # Convert Number column to integer, handling any non-numeric values
            df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
            
            # Convert other columns to strings
            string_columns = [col for col in expected_columns if col != 'Number']
            for col in string_columns:
                df[col] = df[col].astype(str)
            
            total_rows = len(df)
            self.stdout.write(f'Total rows to import: {total_rows}')

            # Process in batches
            imported_count = 0
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Prepare batch data
                skill_job_function_batch = []
                
                for _, row in batch_df.iterrows():
                    skill_job_function = SkillJobFunction(
                        skill=row['Skill'].strip(),
                        skill_type=row['Skill Type'].strip(),
                        primary_industry=row['Primary Industry'].strip(),
                        secondary_industry=row['Secondary Industry'].strip(),
                        primary_job_function=row['Primary Job Function'].strip(),
                        secondary_job_function=row['Secondary Job Function'].strip(),
                        related_skill_1=row['Related Skill 1'].strip(),
                        related_skill_2=row['Related Skill 2'].strip(),
                        related_skill_3=row['Related Skill 3'].strip(),
                        skill_definition=row['Skill Definition'].strip(),
                        number=row['Number'] if pd.notna(row['Number']) else None,
                    )
                    skill_job_function_batch.append(skill_job_function)
                
                # Bulk create with transaction
                with transaction.atomic():
                    SkillJobFunction.objects.bulk_create(
                        skill_job_function_batch,
                        batch_size=batch_size
                    )
                
                imported_count += len(skill_job_function_batch)
                self.stdout.write(
                    f'Imported {imported_count}/{total_rows} records...'
                )

            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully imported {imported_count} skill job function records!'
                )
            )

            # Display some statistics
            total_records = SkillJobFunction.objects.count()
            unique_skills = SkillJobFunction.objects.values('skill').distinct().count()
            unique_skill_types = SkillJobFunction.objects.values('skill_type').distinct().count()
            unique_industries = SkillJobFunction.objects.values('primary_industry').distinct().count()
            unique_job_functions = SkillJobFunction.objects.values('primary_job_function').distinct().count()
            
            self.stdout.write('\nImport Statistics:')
            self.stdout.write(f'- Total records: {total_records}')
            self.stdout.write(f'- Unique skills: {unique_skills}')
            self.stdout.write(f'- Unique skill types: {unique_skill_types}')
            self.stdout.write(f'- Unique primary industries: {unique_industries}')
            self.stdout.write(f'- Unique primary job functions: {unique_job_functions}')

            # Sample skill types
            skill_types = list(SkillJobFunction.objects.values_list('skill_type', flat=True).distinct()[:10])
            self.stdout.write(f'- Sample skill types: {skill_types}')

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error importing data: {str(e)}')
            )
            raise