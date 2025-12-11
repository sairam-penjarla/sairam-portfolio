import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "CNN.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Deep Learning",
    "module_title": "Basics of Convolution Neural Network",
    "url": "/learn/courses/data_science/deep_learning",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_14.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
    {
      "name": "MIT 6.S191: Introduction to Deep Learning",
      "url": "https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI",
      "type": "video"
    },
    {
      "name": "Complete Deep Learning (Krish Naik)",
      "url": "https://www.youtube.com/playlist?list=PLZoTAELRMXVPGU70ZGsckrMdr0FteeRUi",
      "type": "video"
    },
    {
      "name": "Deep Learning With Tensorflow 2.0, Keras and Python (codebasics)",
      "url": "https://youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&si=ogXixV-g8cdaOhCx",
      "type": "video"
    },
    {
      "name": "Deep Learning Crash Course for Beginners (freeCodeCamp.org)",
      "url": "https://www.youtube.com/watch?v=VyWAvY2CF9c",
      "type": "video"
    },
    {
      "name": "DeepLearning.AI YouTube Channel",
      "url": "https://www.youtube.com/@Deeplearningai",
      "type": "video"
    },
    {
      "name": "Machine Learning vs Deep Learning (IBM Technology)",
      "url": "https://www.youtube.com/watch?v=q6kJ71tEYqM",
      "type": "video"
    },
    {
      "name": "But what is a neural network? | Deep learning chapter 1 (3Blue1Brown)",
      "url": "https://www.youtube.com/watch?v=aircAruvnKk",
      "type": "video"
    },
    {
    "name": "What are Convolutional Neural Networks (CNNs)? (IBM Technology)",
    "url": "https://www.youtube.com/watch?v=QzY57FaENXg",
    "type": "video"
  },
  {
    "name": "Simple explanation of Convolutional Neural Network | Deep Learning Tutorial 23 (Tensorflow & Python) - codebasics",
    "url": "https://www.youtube.com/watch?v=zfiSAzpy9NM",
    "type": "video"
  },
  {
    "name": "Convolutional Neural Networks (CNNs) Explained - deeplizard",
    "url": "https://www.youtube.com/watch?v=YRhxdVk_sIs",
    "type": "video"
  },
  {
    "name": "MIT 6.S191: Convolutional Neural Networks (Alexander Amini)",
    "url": "https://www.youtube.com/watch?v=oGpzWAlP5p0",
    "type": "video"
  },
  {
    "name": "Convolutional Neural Networks (Course 4 of Deep Learning Specialization) - DeepLearning.AI",
    "url": "https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF",
    "type": "video"
  },
  {
    "name": "Neural Networks Part 8: Image Classification with CNNs - StatQuest with Josh Starmer",
    "url": "https://www.youtube.com/watch?v=HGwBXDKFk9I",
    "type": "video"
  }
  ]
}