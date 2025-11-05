import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "data_preprocessing.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Data Preprocessing",
    "module_title": "Basics of Data Preprocessing",
    "category": "Basics",
    "url": "/learn/courses/basics/data_preprocessing",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_7.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources":[
  {
    "name": "A Complete Guide to Data Preprocessing Essential Tools in Python (YouTube)",
    "url": "https://www.youtube.com/watch?v=_OvLOS6sBH4",
    "type": "video"
  },
  {
    "name": "Data Preprocessing in Python for Machine Learning (YouTube)",
    "url": "https://www.youtube.com/watch?v=0JLWrqTVzl4",
    "type": "video"
  },
  {
    "name": "Data Cleaning/Data Preprocessing Before Building a Model - A Comprehensive Guide (YouTube)",
    "url": "https://www.youtube.com/watch?v=GP-2634exqA",
    "type": "video"
  },
  {
    "name": "Preprocessing Data in Python (YouTube)",
    "url": "https://www.youtube.com/watch?v=bFCe_e4raaY",
    "type": "video"
  },
  {
    "name": "Data Preprocessing: A Complete Guide with Python Examples (DataCamp blog)",
    "url": "https://www.datacamp.com/blog/data-preprocessing",
    "type": "web"
  },
  {
    "name": "Data Preprocessing in Machine Learning: Steps & Best Practices (LakeFS blog)",
    "url": "https://lakefs.io/blog/data-preprocessing-in-machine-learning/",
    "type": "web"
  },
  {
    "name": "Practical Guide on Data Preprocessing in Python using Scikit-Learn (Analytics Vidhya)",
    "url": "https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/",
    "type": "web"
  },
  {
    "name": "Data Preprocessing in Python (GeeksforGeeks)",
    "url": "https://www.geeksforgeeks.org/machine-learning/data-preprocessing-machine-learning-python/",
    "type": "web"
  },
  {
    "name": "Data Prep for Machine Learning Checklist (Medium)",
    "url": "https://medium.com/learning-data/data-prep-for-machine-learning-checklist-129b46b73782",
    "type": "web"
  },
  {
    "name": "Data Preprocessing: Step-by-Step Guide & Top Tools (Kanerika blog)",
    "url": "https://kanerika.com/blogs/data-preprocessing/",
    "type": "web"
  }
]


}
