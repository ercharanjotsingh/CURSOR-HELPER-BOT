A. Getting started
Q1. What is Cursor, and what are its main ways to use AI?
Cursor is an AI code editor built on the VS Code experience. It has three main interaction modes. Tab gives inline autocomplete while you type. Cmd/Ctrl+K edits selected code in place from a short instruction. Chat/Agent (Cmd/Ctrl+L) opens a conversation panel, and in Agent mode Cursor can edit several files and run terminal commands.
Q2. What should I set up before building a chatbot in Cursor?
Write a one-paragraph spec first: what the bot does, who uses it, the tech stack, and what "done" looks like. Then install Cursor, open an empty folder as your workspace, run git init, create a GitHub repository, and add a .gitignore. Keep API keys in a .env file and list it in .gitignore before your first commit. Choose a stack you can at least read, such as Python.
Q3. How do I start a new project from scratch in Cursor?
Open an empty folder with File > Open Folder, so the Agent sees the whole project and not a single file. Create the repository with git init, then ask the Agent in a Plan-first prompt to propose a folder structure, requirements.txt, .gitignore, and a README. Review the proposal, approve it, and commit the scaffold as your first commit.
B. Prompting and context
Q4. How do I give Cursor context about my code and docs?
Type @ references in your prompt. Use @file or @folder to point at specific code, @docs to reference documentation (you can add custom documentation URLs), @web to search the web, and @git to reference commits and diffs. When something fails, paste the exact error message or terminal output instead of describing it.
Q5. What are Cursor Rules, and where do they live?
Rules are standing instructions that Cursor applies to your project, such as the language version, test framework, and coding style. They live in the .cursor/team-rules folder; older versions used a single .cursorrules file. A useful rule for the chatbot project: use Python 3.11, add type hints, write pytest tests, keep functions small, and never commit secrets. Rules save you from repeating these instructions in every prompt.
Q6. What makes a good prompt in Cursor?
A good prompt states four things: the goal, the relevant context, the constraints (libraries to use or avoid), and the expected output. A weak prompt is "make a chatbot." A strong prompt is "Create ingest.py that loads PDFs from /data, splits them into 500-token chunks, embeds them, and stores them in Chroma. Do not touch other files."
Q7. How do I avoid large, messy changes from the Agent?
Work in small steps: one feature per prompt, then test, then commit. Ask for a plan before code, for example "outline the files you will change and wait for my approval." Name the files the Agent may touch. Large vague prompts produce large vague diffs that are hard to review.
C. Agent mode
Q8. What is Agent mode, and how is it different from Chat and Cmd/Ctrl+K?
Cmd/Ctrl+K edits a selected piece of code in place. Chat answers questions and suggests code. Agent mode goes further: it can create and edit multiple files, run terminal commands, read the results, and fix its own errors. Use Cmd/Ctrl+K for small local edits and Agent for features that span several files.
Q9. How should I communicate with the Agent?
Treat it like a capable junior developer who cannot read your mind. Ask for a plan first and approve it. Review every diff and read each terminal command before allowing it. When something breaks, paste the error and ask for the root cause before any change. Correct it specifically, for example "dedupe retrieved chunks by chunk ID" rather than "it's broken."
Q10. How do I undo something the Agent got wrong?
Reject the diff for the files you do not want. Use Cursor's checkpoints to restore an earlier state of the conversation. For a stronger safety net, commit to Git before every big task, so git restore or git reset can bring back a working version.
Q11. When should I start a new chat?
Start a fresh chat for each new feature or task. Long conversations drift and fill the context window, which lowers answer quality. Continuity should come from your rules file, your README, and your Git history, not from one endless thread.
D. Building the RAG chatbot
Q12. What are the steps to build a RAG chatbot with Cursor?
RAG means the bot retrieves relevant document chunks and answers only from them. Build it in this order, with one prompt and one commit per step: (1) scaffold the project, (2) write the rules file, (3) ingestion: load, chunk, embed, store in a vector database such as Chroma, (4) retrieval: return the top-k chunks for a question, (5) generation: a prompt that answers only from the chunks and says "I don't know" otherwise, (6) a simple UI such as Streamlit with source citations, (7) tests and a README, (8) push to GitHub.
Q13. What prompt can I use for the ingestion step?
"Create ingest.py that loads Markdown files from /data, splits them into 500-token chunks with 50-token overlap, embeds them with my chosen embedding API, and stores them in a local Chroma database. Do not touch other files. Add a short README section on how to run it. Explain your plan first and wait for my approval."
Q14. How do I use GitHub with Cursor while building?
Create a GitHub repository at the start and commit after each working feature. Use a feature branch for each step, push it, and open a pull request. Review the diff yourself before merging, and let Cursor help draft commit messages and the pull request description. Never commit .env or API keys.
E. Team conventions
Q15. What is our team's code review rule?
Every pull request needs two approvals, and one must come from a senior engineer. Pull requests larger than 400 changed lines must be split before review.
Q16. Who owns the CURSOR-HELPER-BOT project?
The project is owned by the Platform Enablement team. Questions go to the #cursor-helper channel on Slack.
Q17. What is the name of our internal staging environment?
The staging environment is called Falcon. Deploy to Falcon before every production release.
