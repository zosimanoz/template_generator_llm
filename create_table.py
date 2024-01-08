import sqlite3

conn = sqlite3.connect('database.db')
print("Connected to database successfully")

conn.execute('CREATE TABLE HedisMeasure (name TEXT)')
conn.execute('CREATE TABLE Bucket (name TEXT)')

print("Created table successfully!")

conn.close()