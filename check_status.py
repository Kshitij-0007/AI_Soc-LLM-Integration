import psycopg

conn = psycopg.connect('postgresql://postgres:postgres@localhost:5432/aisoc')
cur = conn.cursor()
# Check enum types
cur.execute("SELECT typname FROM pg_type WHERE typname IN ('lane_id_enum', 'run_status_enum', 'severity_enum', 'unified_class_enum', 'tactic_enum')")
types = cur.fetchall()
print('Types:', [t[0] for t in types])
# Check tables
cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
tables = cur.fetchall()
print('Tables in public:', [t[0] for t in tables])
cur.close()
conn.close()