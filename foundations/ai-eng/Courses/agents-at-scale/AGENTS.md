# brief: agent conventions

- Quote verbatim; never paraphrase inside a quote. (chapter 4, task t14: a paraphrased quote shipped as verbatim)
- Never cite a URL you did not fetch in this run. (chapter 4, task t15: a URL lifted from a search snippet)
- If a search returns nothing useful, change the query. Never repeat a search with identical arguments. (chapter 6, failure 2: eight identical searches)
- Large tool outputs are saved to disk; read them in ranges with read_file. (chapter 6, failure 1: a 26k-token context)
- Code runs in a sandbox with no network. Do not try to install packages or reach the internet from code. (chapter 6, failure 5)
- Three sources minimum, from at least two different sites. (chapter 3 spec)
- When the brief is complete, call submit_brief exactly once. (chapter 3: prose is not done)
