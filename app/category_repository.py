import mysql.connector
from env import Env

db = mysql.connector.connect(
    host=Env.DATABASE_HOST,
    database=Env.DATABASE_NAME,
    user=Env.DATABASE_USER,
    password=Env.DATABASE_PASSWORD,
)

if not db.is_connected():
    raise Exception("MySQLサーバーへの接続に失敗しました")

# 結果を辞書形式にする
cursor = db.cursor(dictionary=True)

cursor.execute("SELECT * FROM category_table;")

rows = cursor.fetchall()
for row in rows:
    print(row)

cursor.close()
db.close()