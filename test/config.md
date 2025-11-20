# Model Configuration Guide

## Overview
This project now uses a **simplified model configuration** system where you can easily customize which AI model to use by editing your `.env` file.

## Quick Start

### 1. Set Your API Key
```env
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
```
Get your key at: https://openrouter.ai/keys

### 2. Choose Your Model
```env
MODEL_NAME=deepseek/deepseek-chat
```

That's it! The app will automatically use your selected model.

## Available Models

### Recommended for SQL Tasks

| Model | Price | Best For |
|-------|-------|----------|
| `deepseek/deepseek-chat` | ~$0.14/M tokens | SQL generation, code |
| `google/gemini-2.0-flash-exp:free` | FREE | Fast queries, vision support |
| `qwen/qwen-2.5-coder-32b-instruct` | ~$0.18/M tokens | Code and SQL |

### Premium Models

| Model | Price | Best For |
|-------|-------|----------|
| `anthropic/claude-3.5-sonnet` | ~$3/M tokens | Complex reasoning, analysis |
| `openai/gpt-4o` | ~$2.50/M tokens | General purpose, vision |
| `google/gemini-pro-1.5` | ~$1.25/M tokens | Long context, analysis |

### Budget Options

| Model | Price | Best For |
|-------|-------|----------|
| `mistralai/mistral-7b-instruct` | ~$0.07/M tokens | Basic queries |
| `meta-llama/llama-3.1-8b-instruct` | ~$0.06/M tokens | Simple tasks |

*Prices are approximate and may change. Check https://openrouter.ai/models for current pricing.*

## Configuration Examples

### Example 1: Free Tier (Google Gemini)
```env
OPENROUTER_API_KEY=sk-or-v1-abc123...
MODEL_NAME=google/gemini-2.0-flash-exp:free
OPENROUTER_API_URL=https://openrouter.ai/api/v1
```

### Example 2: Best SQL Performance (DeepSeek)
```env
OPENROUTER_API_KEY=sk-or-v1-abc123...
MODEL_NAME=deepseek/deepseek-chat
OPENROUTER_API_URL=https://openrouter.ai/api/v1
```

### Example 3: Premium Quality (Claude)
```env
OPENROUTER_API_KEY=sk-or-v1-abc123...
MODEL_NAME=anthropic/claude-3.5-sonnet
OPENROUTER_API_URL=https://openrouter.ai/api/v1
```

## How to Change Models

1. **Edit `.env` file**
   ```bash
   nano .env
   # or
   code .env
   ```

2. **Update MODEL_NAME**
   ```env
   MODEL_NAME=your-chosen-model
   ```

3. **Restart Streamlit**
   ```bash
   streamlit run app.py
   ```

4. **Click "Clear Cache"** in the sidebar if needed

## Finding More Models

Browse all available models at: https://openrouter.ai/models

Filter by:
- **Pricing** (free, budget, premium)
- **Capabilities** (code, reasoning, vision)
- **Context Length** (how much text they can process)
- **Provider** (OpenAI, Anthropic, Google, etc.)

## Model Selection Tips

### For SQL Generation:
✅ **Best**: `deepseek/deepseek-chat`
- Excellent at understanding SQL syntax
- Cost-effective
- Fast response times

✅ **Alternative**: `qwen/qwen-2.5-coder-32b-instruct`
- Great code understanding
- Slightly more expensive

### For Complex Analysis:
✅ **Best**: `anthropic/claude-3.5-sonnet`
- Superior reasoning
- Handles complex queries
- More expensive

✅ **Alternative**: `openai/gpt-4o`
- Strong general capabilities
- Vision support included

### For Budget Projects:
✅ **Best**: `google/gemini-2.0-flash-exp:free`
- Completely free
- Good performance
- Vision capabilities

✅ **Alternative**: `mistralai/mistral-7b-instruct`
- Very cheap (~$0.07/M tokens)
- Decent for simple queries

## Troubleshooting

### Model Not Working?

1. **Check model name format**
   ```env
   # Correct ✅
   MODEL_NAME=deepseek/deepseek-chat
   
   # Wrong ❌
   MODEL_NAME=deepseek-chat
   MODEL_NAME=openrouter/deepseek/deepseek-chat
   ```

2. **Verify API key**
   - Must start with `sk-or-v1-`
   - No spaces or quotes
   - Check at https://openrouter.ai/keys

3. **Check model availability**
   - Visit https://openrouter.ai/models
   - Some models may be temporarily unavailable
   - Free tier has rate limits

4. **Clear cache**
   - Click "Clear Cache" button in sidebar
   - Restart Streamlit app

### Rate Limits

Free tier limitations:
- **Google Gemini Free**: ~10 requests/minute
- **Paid models**: Depends on your OpenRouter credits

If you hit rate limits:
1. Wait 60 seconds
2. Upgrade to paid tier
3. Switch to a different model

## Environment Variable Reference

```env
# Required
OPENROUTER_API_KEY=sk-or-v1-your-key    # Your OpenRouter API key
MODEL_NAME=deepseek/deepseek-chat        # AI model to use

# Optional
OPENROUTER_API_URL=https://openrouter.ai/api/v1  # API endpoint
OPENROUTER_REFERER=https://github.com/you/repo   # Your app URL
OPENROUTER_TITLE=Text-to-SQL Agent               # App name

# Database
APP_DB_URL=postgresql://user:pass@host:port/db   # Chat history DB
DATA_DB_URL=postgresql://user:pass@host:port/db  # Data to query

# Optional Features
DEBUG=True                               # Enable debug mode
SCHEMA_TABLE_WHITELIST=table1,table2    # Limit visible tables
MAX_SCHEMA_TABLES=50                    # Max tables in schema
```

## Cost Optimization

### Strategy 1: Use Free Tier
```env
MODEL_NAME=google/gemini-2.0-flash-exp:free
```
- $0 cost
- Good for development/testing
- Rate limited

### Strategy 2: Budget Model
```env
MODEL_NAME=deepseek/deepseek-chat
```
- ~$0.14 per million tokens
- Excellent for SQL
- Fast responses

### Strategy 3: Model Switching
Use different models for different tasks:
- **Simple queries**: Free Gemini
- **Complex SQL**: DeepSeek
- **Data analysis**: Claude (when needed)

Just change `MODEL_NAME` in `.env` and restart!

## Support

**Issues?**
- Check https://openrouter.ai/docs
- Review model documentation
- Open GitHub issue with your config (hide API key!)

**Want to contribute?**
- Add support for more providers
- Optimize prompts for specific models
- Share your model recommendations