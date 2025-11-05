import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "TensorFlow.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Deep Learning",
    "module_title": "Basics of Tensoflow",
    "url": "/learn/courses/data_science/deep_learning",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_14.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
{
"name": "Deep Learning With Tensorflow 2.0, Keras and Python (codebasics playlist)",
"url": "[https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO](https://www.youtube.com/playlist?list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO)",
"type": "video"
},
{
"name": "Tensorflow Tutorial for Python in 10 Minutes (Nicholas Renotte)",
"url": "[https://www.youtube.com/watch?v=6_2hzRopPbQ](https://www.youtube.com/watch?v=6_2hzRopPbQ)",
"type": "video"
},
{
"name": "TensorFlow 2.0 Tutorial For Beginners | TensorFlow Demo | Deep Learning & TensorFlow (Simplilearn)",
"url": "[https://www.youtube.com/watch?v=QPDsEtUK_D4](https://www.youtube.com/watch?v=QPDsEtUK_D4)",
"type": "video"
}
]

}