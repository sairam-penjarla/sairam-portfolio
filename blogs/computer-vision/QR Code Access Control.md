# **QR Code Access Control with Pyzbar and OpenCV**  
📅 *June 25, 2024*  
**Secure Your Space with a Python QR Code Access Control System!**

In this blog post, we explore a Python project that implements a **QR code-based access control system**—a modern, secure way to control access to physical spaces. By combining **OpenCV** for video processing and **pyzbar** for QR code decoding, we can build a reliable and effective security solution.

---

## 🔐 **Python Meets Security**

This project leverages the power of **OpenCV** and the **pyzbar** library to create a QR code access control system. Say goodbye to traditional keys or PIN codes, and grant access with just a **QR code scan**.

---

## 🛠️ **Project Breakdown**

The project consists of two key parts:

1. **qr_code_access.py**:  
   This script is the core of the system. It captures webcam input, detects QR codes, decodes the data, and shows access control messages based on the decoded value.

2. **requirements.txt**:  
   This file lists all external libraries needed to run the project.

---

## 🚀 **Setting Up Your Security System**

Follow these steps to get your QR code access control system up and running:

### 1. **Grab the Code**  
Clone the repository by running the following command:

```bash
git clone https://github.com/sairam-penjarla/QR-Code.git
```

### 2. **Install Dependencies**  
The project uses external libraries like OpenCV and pyzbar. Install them with:

```bash
pip install -r requirements.txt
```

### 3. **Run the Script**  
Once everything is set up, run the script to start the access control system:

```bash
python qr_code_access.py
```

---

## 🎯 **Granting or Denying Access**

### 1. **Launch and Position**  
Running the script will activate your webcam and display the video feed. A blue square will appear in the center of the feed. This is your **target zone**!

### 2. **Scan the Code**  
Hold your QR code within the blue square for optimal detection. The script will decode the data from the QR code.

### 3. **Access Granted or Denied**  
- If the decoded data matches a predefined access code, the script will display **“Access Granted”** along with the decoded value.  
- A **green bounding box** will surround the QR code as confirmation.

If there's no match, the message **“Access Denied”** will appear, and a **red bounding box** will surround the QR code.

### 4. **Exit the System**  
To stop the system, simply press the **"q"** key to close the webcam feed and exit the program.

---

## 🔑 **Customizing Your Access Codes**

The script comes with a predefined list of authorized access codes. You can modify this list in the script to suit your specific needs. By default, the script includes these example codes:

```python
access_codes = ["12345", "67890", "ABCDE"]
```

---

## 🛠️ **Technical Requirements**

- **OpenCV** and **pyzbar**: These libraries form the backbone of the project and are listed in the `requirements.txt` file.
- **Webcam**: Ensure that you have a working webcam connected to your computer for the script to function properly.

---

## 🌐 **Exploring Further**

This blog post equips you with the knowledge to build and run your own **QR code access control system** in Python. The project offers numerous possibilities for further development, including:

- **Logging access attempts** for record-keeping.
- **Database integration** for more sophisticated access management.
- Implementing **advanced QR code detection** techniques for better accuracy.

---

## 📚 **References**

For deeper exploration of the libraries used:

- [OpenCV](https://opencv.org/)
- [Pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar)

---

**YouTube Video:**  
Watch a demo of the system in action on [YouTube](https://www.youtube.com/watch?v=iDYGHETBhpQ).

---

With some customization, this Python project can be a valuable addition to your security system. Happy coding and happy securing! 🔒
