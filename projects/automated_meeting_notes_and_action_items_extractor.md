# Automated Meeting Notes & Action Items Extractor

## Overview

This AI-powered tool automatically generates **concise meeting summaries** and **extracts actionable tasks** from conversations. It is designed to help teams save time, improve clarity, and ensure follow-ups are not missed.

By combining advanced language models with natural language processing techniques, the project turns raw meeting transcripts into structured, digestible outputs.

---

## Key Features

- **Automatic Summaries:** Converts meeting conversations into clear, concise summaries.
- **Action Item Extraction:** Identifies tasks, assigns priorities, and suggests responsible team members if mentioned.
- **Multi-Platform Input:** Works with transcript text from video calls, audio recordings (transcribed), or chat logs.
- **Real-Time Processing:** Optional streaming mode for ongoing meetings.
- **Export Options:** Download summaries and action items as PDF or CSV for easy distribution.

---

## Technology Stack

- **Core AI Models:** OpenAI GPT-4, Anthropic Claude
- **NLP Libraries:** spaCy, NLTK
- **Workflow Orchestration:** LangChain
- **Front-end / Demo:** Streamlit for an interactive web interface
- **Programming Language:** Python

---

## Workflow

1. **Input:** Users provide a meeting transcript or live audio (transcribed).
2. **Preprocessing:** Text is cleaned and tokenized using spaCy and NLTK.
3. **Summary Generation:** GPT-4 or Claude produces a concise meeting summary.
4. **Action Item Extraction:** NLP models identify tasks, deadlines, and responsibilities.
5. **Output:** Structured summary and task list displayed in the web interface with download options.

---

## User Interaction

- Users can **upload transcripts** or **paste text directly**.
- Summaries and action items are displayed in a **clean, interactive interface**.
- Action items are presented in **tables with columns** like Task, Owner, and Priority.
- Users can **edit, export, or share** results directly from the platform.

---

## Impact

- Reduces manual effort in documenting meetings.
- Helps teams stay organized and aligned on action items.
- Improves productivity by summarizing lengthy discussions quickly.
- Provides a foundation for integrating AI into everyday workplace workflows.
