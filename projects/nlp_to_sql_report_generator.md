# Natural Language to SQL Report Generator

## Overview
The **Natural Language to SQL Report Generator** is a tool designed to simplify business intelligence workflows. It allows users to input queries in plain English and automatically generates SQL queries to extract the relevant data from databases. The results are then converted into interactive reports for data-driven decision-making.

This project bridges the gap between non-technical users and data analytics, reducing the need to write complex SQL manually while maintaining accuracy and flexibility.

---

## Workflow

1. **User Input**  
   Users submit queries in natural language (e.g., "Show me monthly sales by region").

2. **NLP Processing**  
   The input is processed using NLP libraries like **spaCy** and **NLTK** to identify entities, intents, and key data requirements.

3. **SQL Generation**  
   A **GPT-4 powered model** converts the processed natural language into a syntactically correct SQL query.

4. **Data Retrieval**  
   The SQL query is executed against **SQL Server** or other structured databases via **Databricks Notebooks**.

5. **Report Creation**  
   Retrieved data is transformed using **pandas** and visualized with **Plotly/Dash** to produce interactive charts and dashboards.

6. **Versioning and Model Management**  
   Models and queries are tracked and managed with **Databricks MLflow** and **Model Registry**, ensuring reproducibility and easy updates.

---

## Technology Stack
- **Programming & Scripting:** Python  
- **Natural Language Processing:** spaCy, NLTK  
- **Database & Querying:** SQL Server  
- **Data Platform:** Databricks Notebooks, Unity Catalog  
- **Model Management:** MLflow, Model Registry  
- **AI Model:** GPT-4 API  
- **Data Handling & Visualization:** pandas, Plotly/Dash  

---

## User Interaction
- Enter plain English queries via a web interface.
- View generated SQL queries for transparency and validation.
- Access automatically generated interactive reports.
- Download or share reports for collaborative decision-making.

---

## Impact
- Reduces dependency on SQL expertise for business users.
- Speeds up report generation for data-driven insights.
- Provides an auditable and reproducible workflow for data analytics teams.
- Serves as a practical demonstration of AI-powered natural language understanding in enterprise reporting.