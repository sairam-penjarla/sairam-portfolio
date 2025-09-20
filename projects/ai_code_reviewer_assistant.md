# AI Code Reviewer Assistant

**Tech Stack:** Python, Pandas, OpenAI GPT-4, Claude, Huggingface  
**Project Type:** Web-based prototype for developers

## Project Overview
The AI Code Reviewer Assistant is a **web-accessible tool** that helps developers perform **context-aware code reviews** faster and more efficiently. It augments the review process by providing automated feedback on code quality, style, and maintainability.

## How it Works
1. **Integration with Repositories:** Users connect the tool to their **Azure DevOps** repositories. It monitors pull requests and extracts the relevant **code diffs**.
2. **Automated Analysis:**
   - **GPT-4 and Claude** generate human-readable comments suggesting improvements for readability, naming conventions, and adherence to best practices.
   - **Huggingface code models** analyze patterns in the code to detect potential **code smells**, anti-patterns, or repetitive logic that could be simplified.
3. **Data Aggregation:**
   - **Python and Pandas** collect metrics such as number of lines changed, types of issues commonly flagged, and PR review trends.
   - Users can view dashboards showing the **most frequent code issues** or areas needing attention across the codebase.
4. **Feedback Delivery:**
   - Suggestions are presented either **inline in the PR via comments** or through a **dashboard summary**, giving developers actionable guidance while maintaining **human oversight**.

## Impact / Takeaways
- Helps developers **spot potential readability and maintainability issues** quickly.
- Reduces repetitive manual review work, allowing human reviewers to focus on **logic correctness and complex issues**.
- Provides a foundation for integrating **security scanners, static analyzers, or other AI models** in future versions.
