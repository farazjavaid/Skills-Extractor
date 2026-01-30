import pandas as pd
from django.core.management.base import BaseCommand
from extractor.models import SkillsDemand

class Command(BaseCommand):
    help = 'Import skills demand data from Excel file'

    def add_arguments(self, parser):
        parser.add_argument('excel_file', type=str, help='Path to Excel file')
        parser.add_argument('--clear', action='store_true', help='Clear existing data first')

    def handle(self, *args, **options):
        excel_file = options['excel_file']
        
        try:
            # Read Excel file
            self.stdout.write(f'Reading: {excel_file}')
            df = pd.read_excel(excel_file)
            
            self.stdout.write(f'Columns: {list(df.columns)}')
            self.stdout.write(f'Rows: {len(df)}')
            
            # Clear existing data if requested
            if options['clear']:
                SkillsDemand.objects.all().delete()
                self.stdout.write('Cleared existing data')
            
            # Import data
            created = 0
            for index, row in df.iterrows():
                try:
                    skill_name = str(row['Skill']).strip() if pd.notna(row['Skill']) else ''
                    
                    if skill_name:
                        SkillsDemand.objects.update_or_create(
                            skill=skill_name,
                            defaults={
                                'demand': str(row['Demand']).strip() if pd.notna(row['Demand']) else '',
                                'demand_rationale': str(row['Demand Rationale']).strip() if pd.notna(row['Demand Rationale']) else '',
                                'risk': str(row['Risk']).strip() if pd.notna(row['Risk']) else '',
                                'risk_rationale': str(row['Risk Rationale']).strip() if pd.notna(row['Risk Rationale']) else '',
                            }
                        )
                        created += 1
                        
                        if created % 100 == 0:
                            self.stdout.write(f'Processed {created} records...')
                
                except Exception as e:
                    self.stdout.write(f'Error on row {index + 1}: {e}')
            
            self.stdout.write(self.style.SUCCESS(f'Successfully imported {created} skills'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))