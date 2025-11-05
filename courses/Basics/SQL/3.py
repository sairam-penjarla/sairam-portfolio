import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "3 advanced.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "SQL",
    "module_title": "Advanced SQL",
    "category": "Basics",
    "url": "/learn/courses/basics/sql",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_10.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "SQL For Big Data Engineering [Full Course 2025]",
    "url": "https://www.youtube.com/watch?v=b0TCqhP2P7I",
    "type": "video"
  },
  {
    "name": "Advanced SQL Techniques for Data Engineering Success | Instalation of Sql | AWS | Azure",
    "url": "https://www.youtube.com/watch?v=twySXl1RLHU",
    "type": "video"
  },
  {
    "name": "Advanced SQL Tutorial for Beginners (Simplilearn)",
    "url": "https://www.youtube.com/watch?v=Cl_MSch3Ss0",
    "type": "video"
  },
  {
    "name": "Advanced SQL Course (Interactive Tutorial)",
    "url": "https://www.sqlcourse.com/",
    "type": "web"
  },
  {
    "name": "Advanced SQL for Query Tuning and Performance Optimization (LinkedIn Learning)",
    "url": "https://www.learndatasci.com/best-sql-courses/",
    "type": "web"
  },
  {
    "name": "Advanced SQL Concepts: Techniques For Data Engineers",
    "url": "https://www.dataforgelabs.com/advanced-sql-concepts",
    "type": "web"
  }
]
}