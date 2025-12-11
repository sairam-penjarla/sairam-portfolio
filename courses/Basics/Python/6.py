import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "6 Pandas.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Pandas",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Python Pandas Tutorial (Data Analysis) - freeCodeCamp",
    "url": "https://www.youtube.com/watch?v=vmEHCJofslg",
    "type": "video"
  },
  {
    "name": "Pandas Tutorial for Beginners - Programming with Mosh",
    "url": "https://www.youtube.com/watch?v=5JnMutdy6Fw",
    "type": "video"
  },
  {
    "name": "Python Pandas Tutorial (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=9b4Zm3hxwM4",
    "type": "video"
  },
  {
    "name": "Python Pandas Official Documentation",
    "url": "https://pandas.pydata.org/docs/",
    "type": "web"
  },
  {
    "name": "Pandas Tutorial - W3Schools",
    "url": "https://www.w3schools.com/python/pandas/default.asp",
    "type": "web"
  },
  {
    "name": "Python Pandas Tutorial - GeeksforGeeks",
    "url": "https://www.geeksforgeeks.org/python-pandas-tutorial/",
    "type": "web"
  },
  {
    "name": "10 Minutes to Pandas (Official Guide)",
    "url": "https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html",
    "type": "web"
  },
  {
    "name": "Pandas for Data Science (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=4H-KBbNQh8g",
    "type": "video"
  },
  {
    "name": "Pandas Tutorial - DataCamp",
    "url": "https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python",
    "type": "web"
  },
  {
    "name": "Python Pandas Tutorial: Beginner to Pro Guide (Analytics Vidhya)",
    "url": "https://www.analyticsvidhya.com/blog/2020/10/python-pandas-tutorial-for-beginners/",
    "type": "web"
  }
]

}