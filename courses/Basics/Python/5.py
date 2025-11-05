import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "5 Numpy.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Numpy",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "NumPy Tutorial for Beginners (freeCodeCamp, 1 Hour)",
    "url": "https://www.youtube.com/watch?v=QUT1VHiLmmI",
    "type": "video"
  },
  {
    "name": "NumPy Tutorial for Beginners (ProgrammingKnowledge)",
    "url": "https://www.youtube.com/watch?v=8Mpc9ukltVA",
    "type": "video"
  },
  {
    "name": "NumPy Official Documentation",
    "url": "https://numpy.org/doc/stable/",
    "type": "web"
  },
  {
    "name": "Python NumPy Tutorial (W3Schools)",
    "url": "https://www.w3schools.com/python/numpy_intro.asp",
    "type": "web"
  },
  {
    "name": "Python NumPy Tutorial (GeeksforGeeks)",
    "url": "https://www.geeksforgeeks.org/numpy-tutorial/",
    "type": "web"
  },
  {
    "name": "NumPy Tutorial: Data Analysis with Python (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=GB9ByFAIAH4",
    "type": "video"
  },
  {
    "name": "NumPy Crash Course for Data Science (Corey Schafer)",
    "url": "https://www.youtube.com/watch?v=8Q4v_8L2BzA",
    "type": "video"
  },
  {
    "name": "Python NumPy Tutorial: Beginner to Pro Guide (DataCamp)",
    "url": "https://www.datacamp.com/community/tutorials/python-numpy-tutorial",
    "type": "web"
  },
  {
    "name": "NumPy Beginner's Guide (TutorialsPoint)",
    "url": "https://www.tutorialspoint.com/numpy/index.htm",
    "type": "web"
  },
  {
    "name": "NumPy for Data Science (Analytics Vidhya)",
    "url": "https://www.analyticsvidhya.com/blog/2020/06/numpy-tutorial-for-beginners/",
    "type": "web"
  }
]

}