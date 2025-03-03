from sqlalchemy.orm import Session
from data_models import Category, CategoryORM
from typing import List

# db = mysql.connector.connect(
#     host=Env.DATABASE_HOST,
#     database=Env.DATABASE_NAME,
#     user=Env.DATABASE_USER,
#     password=Env.DATABASE_PASSWORD,
# )

# if not db.is_connected():
#     raise Exception("MySQLサーバーへの接続に失敗しました")

# # 結果を辞書形式にする
# cursor = db.cursor(dictionary=True)

# cursor.execute("SELECT * FROM category_table;")

# rows = cursor.fetchall()

# cursor.close()
# db.close()

class CategoryRepository:
    def get_all_categories(
        self,
        db: Session,
    ) -> List[Category]:
        category_orms = db.query(CategoryORM).all()
        categories = [Category.from_orm(orm) for orm in category_orms]
        print(f'CategoryRepository get_all_categories:\n{categories}')
        return categories