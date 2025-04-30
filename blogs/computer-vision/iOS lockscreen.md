# **Craft Your Photos with Style: Image Segmentation with Templates using Python!**  
📅 *June 25, 2024*  
**Unleash your creativity with image segmentation and templates!**

Ever wished you could instantly transform your photos with a touch of artistic flair? This blog post introduces you to a captivating **Python project** that leverages **image segmentation** and **templates** to give your photos a stylish, professional touch! (Unleash your creativity with the project video [here](https://www.youtube.com/watch?v=XcdtFqYt5RA&t=123s)).

---

## 🎨 **The Power of Image Segmentation with MediaPipe**

This project uses **MediaPipe**, an open-source framework developed by Google that includes pre-built models for computer vision tasks. In this project, **MediaPipe** is employed for **image segmentation**, allowing you to separate the foreground (like a person) from the background in your photos.

But it doesn’t stop there! This project goes beyond basic segmentation. After isolating the foreground object, you can seamlessly apply different **templates** to the background, transforming your photo into something extraordinary. Imagine placing your portrait on a stylish black-and-white background, or an artistic template – the creative possibilities are endless!

---

## 🖼️ **Template Magic: Transforming Your Photos**

Here’s how it works:

1. **Segmentation**: MediaPipe segments the image, detecting and isolating the foreground from the background.
2. **Template Application**: After segmentation, you can apply various templates, giving your photos an artistic makeover, whether it's a classic black-and-white look or a vibrant, modern design.

---

## 🚀 **Getting Started: Shape Your Photos**

Follow these steps to get your hands on this powerful photo manipulation project:

### **Prerequisites**  
Before you dive into the code, ensure you have the following installed:
- **Python 3.x**
- **OpenCV** (for image processing)
- **NumPy** (for numerical operations)
- **Pillow** (for image manipulation)
- **MediaPipe** (for image segmentation)
- **PyYAML** (for reading configuration files)

### **Installation**

1. **Clone the Codebase (Optional)**  
If you prefer using Git version control, you can clone the project's repository:

```bash
git clone https://github.com/sairam-penjarla/ioslockscreen.git
cd ioslockscreen
```

2. **Manual Download (Optional)**  
Alternatively, you can download the project’s source code directly from the GitHub repository.

3. **Install Dependencies**  
Once you have the code, you can install the necessary libraries with pip:

```bash
pip install -r requirements.txt
```

### **Usage**

1. **Place Your Image**  
Ensure your photo is placed in the directory specified within the project (usually called the `input` directory).

2. **Run the Script**  
With everything in place, launch the project using:

```bash
python image_segmentation_app.py
```

This script will process your image, perform segmentation, and apply the default templates.

### **Customization (Optional)**  
Feeling adventurous? You can customize various aspects of the project using the `config.yaml` file. Here, you can specify:
- Input and output paths for your images.
- Paths to different templates for creating unique styles.
- The path to the MediaPipe segmentation model.

Here’s an example `config.yaml` file:

```yaml
input_path: "../input/b.jpg"  # Path to your image
output_path: "../output"  # Where the output image will be saved
white_template_path: "../assets/white.png"  # Path to white template
black_template_path: "../assets/black.png"  # Path to black template
model_path: "../model/deeplab_v3.tflite"  # Path to MediaPipe segmentation model
```

---

## 🎉 **Unleash Your Creativity!**

This project empowers you to unleash your creativity and transform your photos with the power of **image segmentation** and **templates**. So, grab your photos and get ready to create unique and artistic masterpieces!

---

**Happy Coding and Happy Segmenting!** ✨