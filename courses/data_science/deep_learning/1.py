import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Huggingface.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Deep Learning",
    "module_title": "Basics of HuggingFace",
    "url": "/learn/courses/data_science/deep_learning",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_14.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Hugging Face Course (Hugging Face YouTube Channel)",
    "url": "https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o",
    "type": "video"
  },
  {
    "name": "HuggingFace - YouTube Channel (All Playlists)",
    "url": "https://www.youtube.com/HuggingFace",
    "type": "video"
  },
  {
    "name": "Hugging Face Course - Transformer's GitHub Repository",
    "url": "https://github.com/huggingface/course",
    "type": "web"
  },
  {
    "name": "Getting Started With Hugging Face in 15 Minutes",
    "url": "https://www.youtube.com/watch?v=QEaBAZQCtwE",
    "type": "video"
  },
  {
    "name": "Basics of Hugging Face | Hugging Face Tutorial for Beginners [2024]",
    "url": "https://www.youtube.com/watch?v=J3tMzGigqww",
    "type": "video"
  },
  {
    "name": "Getting Started With Hugging Face in 10 Minutes (Article)",
    "url": "https://huggingface.co/blog/proflead/hugging-face-tutorial",
    "type": "web"
  }
]

}