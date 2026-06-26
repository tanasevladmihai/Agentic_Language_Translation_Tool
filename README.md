# Agentic Language Translation Tool
A tool that can be used by AI agents in popular agentic coding environments such as Claude Code, Antigravity, Codex and others to translate large documents into the target language while matching the exact formatting and style of the original document. \
This method is much more precise, efficient and cost-effective than typical approaches:
- Manually translating the document line by line (if you have the language skills)
- Using a simple translation API
- Dropping text into a translation tool (like Google Translate) and then manually formatting the output into the new document
- Simply dropping the document into the desired LLM's chat interface and asking it to translate the document.

LLMs are **lazy** and **hallucinate often**, this tool is designed to avoid this critical issue using a more structured, modern approach, where sub-agent are deployed to handle the translation, hallucination detection, formatting and style matching and other tasks. 

The tool supports a wide range of document formats, including but not limited to:
- Markdown
- HTML
- LaTeX
- PDF
- Microsoft Word (DOCX)
- Plain text (TXT)
- Rich Text Format (RTF)
- OpenDocument Text (ODT)
- EPUB
- JSON
- XML

> More formats can be added in the future as needed. \
> The tool is designed to be modular and extensible, allowing for **you** to also add support for additional formats if needed. 
