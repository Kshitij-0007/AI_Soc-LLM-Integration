import psycopg

conn = psycopg.connect('postgresql://postgres:postgres@localhost:5432/aisoc')
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name")
tables = cur.fetchall()
print('Tables:', [t[0] for t in tables])
cur.close()
conn.close()