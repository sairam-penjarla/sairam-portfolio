# [![Website](https://img.shields.io/badge/Website-Visit-brightgreen)](https://psairam9301.wixsite.com/website) [![YouTube](https://img.shields.io/badge/YouTube-Subscribe-red)](https://www.youtube.com/@sairampenjarla) [![GitHub](https://img.shields.io/badge/GitHub-Explore-black)](https://github.com/sairam-penjarla) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/sairam-penjarla-b5041b121/) [![Instagram](https://img.shields.io/badge/Instagram-Follow-ff69b4)](https://www.instagram.com/sairam.ipynb/)

<div align="center">
  <img src="https://github.com/sairam-penjarla.png" alt="Sai Ram Penjarla" width="200" style="border-radius: 50%;" />
  <h1>Sai Ram Penjarla - Personal Website</h1>
  <p>Data Scientist | Content Creator | Educator</p>
</div>

## 🚀 About This Project

This repository contains the source code for my personal website, built with Flask and modern HTML/CSS. The website serves as a hub for my projects, educational resources, and technical blogs, focusing on data science, machine learning, AI, and programming.

## ✨ Features

- **Projects Showcase**: Horizontal card layout with detailed information about each project, including descriptions, technologies used, and links to GitHub, demos, and YouTube videos.
- **Educational Resources**: A curated collection of resources organized by topic to help learners navigate the world of data science.
- **Free Courses Section**: A hierarchical, searchable collection of educational content with nested folder navigation.
- **Responsive Design**: Fully responsive layout that works across all devices.
- **Modern UI Components**: Interactive elements like expanding action buttons and searchable sidebar navigation.
- **Syntax Highlighting**: GitHub Dark theme for code blocks using highlight.js.
- **Blog System**: Markdown-based blog with category organization.

## 🛠️ Technologies Used

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **Content**: Markdown with YAML frontmatter
- **Code Highlighting**: highlight.js with GitHub Dark theme
- **Responsive Design**: Custom CSS with mobile-first approach
- **Icons**: Custom SVG icons and FontAwesome

## 📂 Project Structure

```
├── app.py                 # Main Flask application
├── static/                # Static files
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── images/            # Images
├── templates/             # HTML templates
│   ├── blog.html          # Blog template
│   ├── free_courses.html  # Courses template
│   └── ...                # Other templates
├── blogs/                 # Blog content in Markdown
└── courses/               # Course content in Markdown
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/sairam-penjarla/personal-website.git
   cd personal-website
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## 📚 Adding Content

### Adding Projects

Edit the HTML in `templates/home_section_5_projects.html` to add a new project card.

### Adding Blog Posts

1. Create a new Markdown file in the appropriate category folder under `blogs/`
2. Add YAML frontmatter with title, date, and any other metadata
3. Write your content in Markdown

### Adding Courses

1. Create a new folder or use an existing one in the `courses/` directory
2. Add Markdown files with course content
3. Use frontmatter for metadata

## 🎨 Key Features

### Projects Section
- Horizontal card layout with content on the left and media on the right
- Expandable action buttons that show text on hover
- Technology tags displayed in a responsive grid

### Free Courses
- Nested folder navigation for organizing educational content
- Local search function to quickly find specific courses
- Syntax highlighting for code examples

### Blog System
- Category-based navigation
- Markdown rendering with code highlighting
- Mobile-friendly reading experience

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions or feedback, please reach out through any of the channels in the badges at the top of this README. 