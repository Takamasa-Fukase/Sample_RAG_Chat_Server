from sqlalchemy.orm import Session
from data_models import Category, CategoryORM
from typing import List

class CategoryRepository:
    def get_all_categories(
        self,
        db: Session,
    ) -> List[Category]:
        category_orms = db.query(CategoryORM).all()
        categories = [Category.from_orm(orm) for orm in category_orms]
        # TODO: 後で色々考え直したい。暫定処理。改行コードのエスケープ関連。
        for category in categories:
            category.introduction_text = category.introduction_text.replace("\\n", "\n")
        return categories