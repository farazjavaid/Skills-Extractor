# Create this migration manually:
# python manage.py makemigrations --empty extractor --name add_search_indexes

# Then replace the content with:

from django.db import migrations
from django.contrib.postgres.operations import TrigramExtension
from django.contrib.postgres.indexes import GinIndex, BTreeIndex

class Migration(migrations.Migration):

    dependencies = [
        ('extractor', '0002_remove_is_active'),  # Adjust based on your latest migration
    ]

    operations = [
        # Enable trigram extension for better text search
        TrigramExtension(),
        
        # Add GIN indexes for array fields
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS extractor_skill_lemmatized_gin ON extractor_skill USING gin (lemmatized_skills);",
            reverse_sql="DROP INDEX IF EXISTS extractor_skill_lemmatized_gin;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS extractor_skill_variations_gin ON extractor_skill USING gin (suggested_variations);",
            reverse_sql="DROP INDEX IF EXISTS extractor_skill_variations_gin;"
        ),
        
        # Add trigram indexes for text search
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS extractor_skill_name_trgm ON extractor_skill USING gin (skill_name gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS extractor_skill_name_trgm;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS extractor_skill_definition_trgm ON extractor_skill USING gin (skill_definition gin_trgm_ops);",
            reverse_sql="DROP INDEX IF EXISTS extractor_skill_definition_trgm;"
        ),
    ]