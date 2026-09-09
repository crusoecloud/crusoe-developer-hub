# Tic-Tac-Toe 5x5 AI - Quick Start Guide

## Prerequisites

- Python 3.8+
- pip package manager
- Valid Crusoe Foundry API credentials

## 1. Get Crusoe Foundry API Credentials

1. Visit [Crusoe Foundry](https://crusoecloud.com)
2. Create an account or log in
3. Generate an API key
4. Note the API endpoint (typically `https://managed-inference-api-proxy.crusoecloud.com/v1`)

## 2. Install Dependencies

```bash
cd tictactoe-5x5-ai
pip install -r requirements.txt
```

This installs:
- `gradio` - Web UI framework
- `openai` - Crusoe API client (compatible with OpenAI API)

## 3. Run Locally

```bash
python app.py
```

Output:
```
Running on http://0.0.0.0:7860
```

Open your browser to `http://localhost:7860`

## 4. Configure and Play

1. **API URL**: Enter your Crusoe endpoint (pre-filled with default)
2. **API Key**: Enter your Crusoe API key securely
3. **Player 1**: Select a model for X (Cyan)
4. **Player 2**: Select a model for O (Pink)
5. **Start Game**: Click to begin autonomous play

Example configuration:
```
API URL: https://managed-inference-api-proxy.crusoecloud.com/v1
API Key: [your-secret-key-here]
Player 1: deepseek-ai/DeepSeek-V3-0324
Player 2: meta-llama/Llama-3.3-70B-Instruct
```

## 5. Watch the Game

- **Canvas shows**: Live board state with your chosen models playing
- **Cyan (X)**: Player 1 marks
- **Pink (O)**: Player 2 marks
- **Glowing**: Last move gets highlighted
- **Gold dashed line**: Winning 4-in-a-row (if any)
- **Log**: Real-time move history and game state

## Example Game Flow

```
Turn 1 (X): A1 → B3 (Player 1 plays)
Turn 2 (O): C3 → D4 (Player 2 plays)
Turn 3 (X): B4 → E1 (Player 1 plays)
...
Final: deepseek-ai/DeepSeek-V3-0324 (X) WINS with A1-A2-A3-A4!
```

## Available Models

- **DeepSeek V3**: `deepseek-ai/DeepSeek-V3-0324`
- **DeepSeek R1**: `deepseek-ai/DeepSeek-R1-0528`
- **Llama 3.3**: `meta-llama/Llama-3.3-70B-Instruct`
- **Qwen 3**: `Qwen/Qwen3-235B-A22B-Instruct-2507`
- **Gemma 3**: `google/gemma-3-12b-it`

## Game Rules Reminder

- **Board**: 5x5 grid (25 cells)
- **Positions**: A1 through E5 (row letter + column number)
- **Win**: First to get 4 in a row (horizontal, vertical, or diagonal)
- **Draw**: Full board with no winner
- **Max moves**: 25

## Troubleshooting

### "Invalid API Key"
- Double-check your Crusoe API key
- Verify endpoint URL is correct
- Ensure API key has model access permissions

### "Model Not Found"
- Verify model name matches exactly (case-sensitive)
- Confirm your API account has access to selected models

### "Game Appears Stuck"
- LLM responses can be slow - wait 10-30 seconds
- Check console for error messages
- Try simpler models first

### "Invalid Move"
- This is normal - the app handles invalid moves gracefully
- Falls back to random valid move automatically
- Game continues without interruption

## Deployment to Heroku

```bash
# Initialize git repo (if not already)
git init

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set PORT=7860

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

Your app will be live at: `https://your-app-name.herokuapp.com`

Users need to provide API credentials in the UI when visiting your app.

## API Rate Limits

- Crusoe Foundry has rate limits per API key
- Each game uses 5-25 API calls depending on game length
- Spread multiple games to avoid hitting limits
- Check your Crusoe dashboard for usage

## Advanced: Custom Models

To add a custom model:
1. Edit `app.py` line with `model1_choices`
2. Add your model ID to the list
3. Restart the app

```python
model1_choices = [
    "your-org/your-custom-model",
    # ... other models ...
]
```

## Performance Tips

- **Deterministic play**: Temperature is set to 0.2 (consistent moves)
- **Fast responses**: Max tokens limited to 10 (brief replies)
- **Small models**: Faster than large models but less strategic
- **Large models**: Slower but potentially better gameplay

## Next Steps

1. Run a few games locally to understand gameplay
2. Try different model pairings for different strategies
3. Deploy to Heroku for online sharing
4. Monitor Crusoe API usage in your dashboard

## Support

- Crusoe Foundry docs: https://crusoecloud.com/docs
- Gradio docs: https://www.gradio.app/
- OpenAI Python client: https://github.com/openai/openai-python

Enjoy your AI Tic-Tac-Toe battles!
