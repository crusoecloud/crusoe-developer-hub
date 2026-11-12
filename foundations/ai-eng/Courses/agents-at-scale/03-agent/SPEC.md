# brief

Input:  a topic (string)
Tools:  web_search, fetch_url, read_file, write_file
Output: workspace/briefs/<slug>.md with title, summary, sections, sources

Done when:
  1. the file exists
  2. it has at least three sources
  3. every source URL was fetched during this run
  4. every source quote appears verbatim on that fetched page
