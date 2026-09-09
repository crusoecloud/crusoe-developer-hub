mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
base = \"dark\"\n\
primaryColor = \"#0066FF\"\n\
backgroundColor = \"#0A0A0F\"\n\
secondaryBackgroundColor = \"#12121A\"\n\
textColor = \"#FFFFFF\"\n\
" > ~/.streamlit/config.toml
