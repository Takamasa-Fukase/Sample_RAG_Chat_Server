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
        return categories