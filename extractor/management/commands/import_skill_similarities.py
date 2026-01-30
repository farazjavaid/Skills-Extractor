import json
import os
from django.core.management.base import BaseCommand
from django.db import transaction, IntegrityError
from extractor.models import SkillSimilarity
import time
import traceback


class Command(BaseCommand):
    help = 'Import skill similarity data from JSON file'

    def add_arguments(self, parser):
        parser.add_argument(
            'json_file',
            type=str,
            help='Path to the JSON file containing skill similarity data'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing similarity data before importing'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=500,
            help='Number of records to process in each batch'
        )
        parser.add_argument(
            '--skip-errors',
            action='store_true',
            help='Skip records that cause errors and continue importing'
        )

    def handle(self, *args, **options):
        json_file = options['json_file']
        clear_existing = options['clear']
        batch_size = options['batch_size']
        skip_errors = options['skip_errors']
        
        if not os.path.exists(json_file):
            self.stdout.write(
                self.style.ERROR(f'File not found: {json_file}')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting import from {json_file}...')
        )
        
        start_time = time.time()
        
        try:
            # Clear existing data if requested
            if clear_existing:
                self.stdout.write('Clearing existing similarity data...')
                SkillSimilarity.objects.all().delete()
                self.stdout.write(
                    self.style.SUCCESS('Existing data cleared.')
                )
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.stdout.write(f'Loaded {len(data)} skill records from JSON')
            
            # Process in batches WITHOUT atomic transaction for the entire import
            created_count = 0
            updated_count = 0
            error_count = 0
            error_details = []
            
            # Process records in batches
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch = data[batch_start:batch_end]
                
                self.stdout.write(
                    f'Processing batch {batch_start}-{batch_end} of {len(data)}...'
                )
                
                # Use atomic transaction per batch, not for entire import
                try:
                    with transaction.atomic():
                        skills_to_create = []
                        skills_to_update = []
                        
                        for i, skill_data in enumerate(batch):
                            record_index = batch_start + i
                            
                            try:
                                skill_id = skill_data.get('skill_id')
                                thresholds = skill_data.get('thresholds', {})
                                
                                if not skill_id:
                                    error_msg = f'Record {record_index}: No skill_id found'
                                    self.stdout.write(self.style.WARNING(error_msg))
                                    error_count += 1
                                    error_details.append(error_msg)
                                    continue
                                
                                # Convert skill_id to integer if it's not already
                                try:
                                    skill_id = int(skill_id)
                                except (ValueError, TypeError):
                                    error_msg = f'Record {record_index}: Invalid skill_id: {skill_id}'
                                    self.stdout.write(self.style.WARNING(error_msg))
                                    error_count += 1
                                    error_details.append(error_msg)
                                    continue
                                
                                # Prepare the thresholds data
                                # Ensure all threshold keys are strings and values are lists
                                processed_thresholds = {}
                                for threshold_key, threshold_values in thresholds.items():
                                    # Convert threshold key to string
                                    threshold_str = str(threshold_key)
                                    
                                    # Ensure values is a list
                                    if isinstance(threshold_values, list):
                                        processed_thresholds[threshold_str] = threshold_values
                                    else:
                                        self.stdout.write(
                                            self.style.WARNING(
                                                f'Record {record_index}: Invalid threshold values for {threshold_str}'
                                            )
                                        )
                                
                                # Check if record exists
                                existing = SkillSimilarity.objects.filter(
                                    skill_id=skill_id
                                ).first()
                                
                                if existing:
                                    # Update existing record
                                    existing.similarity_thresholds = processed_thresholds
                                    skills_to_update.append(existing)
                                else:
                                    # Create new record
                                    skills_to_create.append(
                                        SkillSimilarity(
                                            skill_id=skill_id,
                                            similarity_thresholds=processed_thresholds
                                        )
                                    )
                                
                            except Exception as e:
                                error_msg = f'Record {record_index}: {str(e)}'
                                self.stdout.write(self.style.ERROR(error_msg))
                                error_count += 1
                                error_details.append(error_msg)
                                
                                if not skip_errors:
                                    raise
                        
                        # Bulk create new records
                        if skills_to_create:
                            created_records = SkillSimilarity.objects.bulk_create(
                                skills_to_create,
                                ignore_conflicts=True  # Skip duplicates
                            )
                            created_count += len(created_records)
                        
                        # Bulk update existing records
                        if skills_to_update:
                            SkillSimilarity.objects.bulk_update(
                                skills_to_update, 
                                ['similarity_thresholds']
                            )
                            updated_count += len(skills_to_update)
                
                except IntegrityError as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f'Integrity error in batch {batch_start}-{batch_end}: {str(e)}'
                        )
                    )
                    if not skip_errors:
                        raise
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f'Error in batch {batch_start}-{batch_end}: {str(e)}'
                        )
                    )
                    if not skip_errors:
                        raise
                    else:
                        # Log the error and continue
                        error_count += len(batch)
                        traceback.print_exc()
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nImport completed in {duration:.2f} seconds!'
                )
            )
            self.stdout.write(f'Created: {created_count} records')
            self.stdout.write(f'Updated: {updated_count} records')
            self.stdout.write(f'Errors: {error_count} records')
            
            # Show first few error details
            if error_details:
                self.stdout.write('\nFirst few errors:')
                for error in error_details[:10]:
                    self.stdout.write(f'  - {error}')
                if len(error_details) > 10:
                    self.stdout.write(f'  ... and {len(error_details) - 10} more errors')
            
        except json.JSONDecodeError as e:
            self.stdout.write(
                self.style.ERROR(f'Invalid JSON file: {str(e)}')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Import failed: {str(e)}')
            )
            traceback.print_exc()