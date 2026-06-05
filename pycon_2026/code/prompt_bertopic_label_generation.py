labels_generation_prompt = """
You are an expert at creating short, clear topic labels.

I will give you:
- A set of documents (captions / short texts) that belong to one topic.
- A list of keywords that describe that topic.

[DOCUMENTS]

The topic is described by the following keywords:
[KEYWORDS]

Your task:
1. Understand what these documents and keywords have in common.
2. Create a single, concise topic label (max 8-10 words).
3. The label must be:
   - Specific, not generic (avoid "general topics", "other").
   - Descriptive of the shared theme, not just reusing one keyword verbatim.
   - In English.
4. Return ONLY the label text, with no explanations, quotes, or punctuation around it.

Topic label:
"""