# **Tesseract OCR with Python**  
📅 *June 27, 2024*  
**Ever come across a scanned document or an image with text and wished you could easily extract it?** Look no further than **Optical Character Recognition (OCR)**! In this blog, we'll explore the magic of Tesseract OCR and guide you through the process of extracting text from images using Python.

[GitHub Repository](https://github.com/sairam-penjarla/Tesseract-OCR)

---

## 📜 **The Enthralling History of Tesseract OCR**

Tesseract OCR has an impressive history. Originally developed by **Hewlett-Packard (HP)** in the early 1990s, Tesseract was open-sourced in 2005, leading to contributions from a vibrant community of developers. In 2006, **Google** recognized its potential and adopted the project, continuing to improve it. Today, Tesseract is one of the leading open-source OCR engines, supporting over 100 languages and powering applications like **Google Drive’s image-to-text functionality**.

---

## 🧠 **The Inner Workings of Tesseract OCR**

So, how does Tesseract work its magic? Here's a breakdown of the key stages it follows:

1. **Preprocessing**: The image is adjusted to enhance the clarity of text. This involves techniques like noise reduction and thresholding (converting the image to black and white).
2. **Segmentation**: Individual characters are isolated from the background.
3. **Feature Extraction**: Important features of each character are identified.
4. **Pattern Recognition**: These features are compared with a built-in database of character patterns for recognition.
5. **Text Reconstruction**: The recognized characters are combined to form words and sentences.

---

## 🧑‍💻 **Unveiling the Python Code (Part by Part)**

Let's dive into the provided Python code to see how it works.

### **1. Dependencies (requirements.txt)**

```txt
pytesseract
argparse
```

These are the required libraries:
- **pytesseract**: A Python wrapper for Tesseract OCR.
- **argparse**: A library for parsing command-line arguments, making the script user-friendly.

### **2. Extracting Text Function (main.py)**

```python
def extract_text(image_path):
    # ... code for processing and saving extracted text ...
```

This function is the heart of the script. It works as follows:
- **Folder Management (Optional)**: Checks for the existence of input and output folders to organize images (you can modify this behavior).
- **Image Handling**: Reads the image using **Pillow's** Image library.
- **Error Handling**: Catches potential `FileNotFoundError` for missing images.
- **Text Extraction**: Uses `pytesseract.image_to_string` to extract text from the image.
- **Output Generation**: Creates a unique filename to save the extracted text in a `.txt` file within the output folder.

### **3. Command-Line Arguments (main.py)**

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Text Extraction Tool")
    parser.add_argument("--image_path", required=True, help="Path to the image file")
    args = parser.parse_args()
    extract_text(args.image_path)
```

This section defines how the script interacts with the user via the command line:
- **Argument Parser**: Creates an `argparse` object to handle command-line arguments.
- **Image Path Argument**: The `--image_path` argument specifies the path to the image file.
- **Parsing Arguments**: The script retrieves the provided image path via `parser.parse_args()`.
- **Function Call**: Calls the `extract_text` function with the image path, starting the text extraction process.

---

## ⚙️ **Getting Started (It's Time to Extract Text!)**

### **Step-by-Step Guide**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sairam-penjarla/Tesseract-OCR.git
   cd Tesseract-OCR
   ```

2. **Install Dependencies**:

   Install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Tesseract OCR is Installed**:

   You may need to install **Tesseract OCR** on your machine. This can involve system-specific steps, so refer to the [Tesseract documentation](https://github.com/tesseract-ocr/tesseract) for details.

4. **Run the Script**:

   Use the following command to run the script and extract text from your image:

   ```bash
   python main.py --image_path <path_to_your_image.jpg>
   ```

   This command will:
   - Process the specified image.
   - Extract the text.
   - Save the extracted text to a `.txt` file in the output folder.

---

## 🚀 **Conclusion**

With Tesseract OCR and Python, you can easily extract text from images and scanned documents. This powerful combination opens up endless possibilities, from automating document processing to analyzing handwritten notes. With just a few lines of code, you’re equipped to take on OCR tasks like a pro!

---

**Happy Coding and Happy OCR-ing!**