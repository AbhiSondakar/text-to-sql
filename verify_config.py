#!/usr/bin/env python3
"""
Configuration Verification Script
Run this to see exactly what configuration is being loaded
"""

import os
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

print("\n" + "=" * 70)
print("CONFIGURATION VERIFICATION REPORT".center(70))
print("=" * 70)

# Get all relevant environment variables
sql_model = os.getenv('SQL_AGENT_MODEL', 'NOT SET')
validator_model = os.getenv('VALIDATOR_AGENT_MODEL', 'NOT SET')
google_key = os.getenv('GOOGLE_API_KEY', 'NOT SET')
deepseek_key = os.getenv('DEEPSEEK_API_KEY', 'NOT SET')
openrouter_key = os.getenv('OPENROUTER_API_KEY', 'NOT SET')
anthropic_key = os.getenv('ANTHROPIC_API_KEY', 'NOT SET')
openai_key = os.getenv('OPENAI_API_KEY', 'NOT SET')

print("\n📋 MODEL CONFIGURATION")
print("-" * 70)
print(f"SQL_AGENT_MODEL:       {sql_model}")
print(f"VALIDATOR_AGENT_MODEL: {validator_model}")

print("\n🔑 API KEYS STATUS")
print("-" * 70)


def show_key_status(name, key):
    if key == 'NOT SET':
        print(f"❌ {name}: NOT SET")
    elif 'your' in key.lower() or 'here' in key.lower():
        print(f"⚠️  {name}: PLACEHOLDER (not real key)")
    else:
        print(f"✅ {name}: Set ({key[:15]}...{key[-5:]})")


show_key_status("GOOGLE_API_KEY     ", google_key)
show_key_status("DEEPSEEK_API_KEY   ", deepseek_key)
show_key_status("OPENROUTER_API_KEY ", openrouter_key)
show_key_status("ANTHROPIC_API_KEY  ", anthropic_key)
show_key_status("OPENAI_API_KEY     ", openai_key)


# Detect providers from model names
def detect_provider(model):
    if model == 'NOT SET':
        return 'NOT SET'
    model_lower = model.lower()
    if '/' in model:
        return 'openrouter'
    elif 'gemini' in model_lower:
        return 'google'
    elif 'deepseek' in model_lower:
        return 'deepseek'
    elif 'claude' in model_lower:
        return 'anthropic'
    elif 'gpt' in model_lower:
        return 'openai'
    else:
        return 'unknown'


sql_provider = detect_provider(sql_model)
validator_provider = detect_provider(validator_model)

print("\n🤖 DETECTED PROVIDERS")
print("-" * 70)
print(f"SQL Agent:       {sql_provider}")
print(f"Validator Agent: {validator_provider}")

# Validation
print("\n🔍 VALIDATION RESULTS")
print("-" * 70)

issues = []
warnings = []

# Check for slash in model names
if '/' in sql_model:
    issues.append({
        'type': 'CRITICAL',
        'msg': f"SQL_AGENT_MODEL contains '/' which triggers OpenRouter",
        'detail': f"Current: {sql_model}",
        'fix': "Remove '/' prefix. Example: 'deepseek/deepseek-chat' → 'deepseek-chat'"
    })

if '/' in validator_model:
    issues.append({
        'type': 'CRITICAL',
        'msg': f"VALIDATOR_AGENT_MODEL contains '/' which triggers OpenRouter",
        'detail': f"Current: {validator_model}",
        'fix': "Remove '/' prefix. Example: 'anthropic/claude' → 'claude-3-5-sonnet-20241022'"
    })

# Check if OpenRouter is being used without key
if (sql_provider == 'openrouter' or validator_provider == 'openrouter'):
    if openrouter_key == 'NOT SET' or 'your' in openrouter_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "Using OpenRouter format but OPENROUTER_API_KEY not configured",
            'detail': "Model names with '/' require OPENROUTER_API_KEY",
            'fix': "Either: (1) Set OPENROUTER_API_KEY, OR (2) Remove '/' from model names"
        })

# Check API key matches for SQL agent
if sql_provider == 'google':
    if google_key == 'NOT SET' or 'your' in google_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "SQL Agent uses Gemini but GOOGLE_API_KEY not configured",
            'detail': f"Model: {sql_model}",
            'fix': "Get key from: https://aistudio.google.com/app/apikey"
        })
    elif not google_key.startswith('AIzaSy'):
        warnings.append({
            'msg': "GOOGLE_API_KEY doesn't start with 'AIzaSy'",
            'detail': "Google API keys typically start with 'AIzaSy'"
        })

elif sql_provider == 'deepseek':
    if deepseek_key == 'NOT SET' or 'your' in deepseek_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "SQL Agent uses DeepSeek but DEEPSEEK_API_KEY not configured",
            'detail': f"Model: {sql_model}",
            'fix': "Get key from: https://platform.deepseek.com/api_keys"
        })
    elif not deepseek_key.startswith('sk-'):
        warnings.append({
            'msg': "DEEPSEEK_API_KEY doesn't start with 'sk-'",
            'detail': "DeepSeek API keys typically start with 'sk-'"
        })

# Check API key matches for Validator agent
if validator_provider == 'deepseek':
    if deepseek_key == 'NOT SET' or 'your' in deepseek_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "Validator Agent uses DeepSeek but DEEPSEEK_API_KEY not configured",
            'detail': f"Model: {validator_model}",
            'fix': "Get key from: https://platform.deepseek.com/api_keys"
        })

elif validator_provider == 'anthropic':
    if anthropic_key == 'NOT SET' or 'your' in anthropic_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "Validator Agent uses Claude but ANTHROPIC_API_KEY not configured",
            'detail': f"Model: {validator_model}",
            'fix': "Get key from: https://console.anthropic.com/settings/keys"
        })

elif validator_provider == 'google':
    if google_key == 'NOT SET' or 'your' in google_key.lower():
        issues.append({
            'type': 'CRITICAL',
            'msg': "Validator Agent uses Gemini but GOOGLE_API_KEY not configured",
            'detail': f"Model: {validator_model}",
            'fix': "Get key from: https://aistudio.google.com/app/apikey"
        })

# Display issues
if issues:
    print("\n❌ CRITICAL ISSUES FOUND:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['msg']}")
        print(f"   Detail: {issue['detail']}")
        print(f"   Fix: {issue['fix']}")
        print()
else:
    print("\n✅ No critical issues found!")

# Display warnings
if warnings:
    print("\n⚠️  WARNINGS:\n")
    for warning in warnings:
        print(f"• {warning['msg']}")
        print(f"  {warning['detail']}")
        print()

# Recommendations
print("\n💡 RECOMMENDATIONS")
print("-" * 70)

if issues:
    print("\n🔧 IMMEDIATE ACTION REQUIRED:")
    print("\n1. Open your .env file")
    print("2. Make these changes:\n")

    if any('/' in model for model in [sql_model, validator_model]):
        print("   # Remove slashes from model names:")
        if '/' in sql_model:
            clean_name = sql_model.split('/')[-1] if '/' in sql_model else sql_model
            print(f"   SQL_AGENT_MODEL={clean_name}")
        if '/' in validator_model:
            clean_name = validator_model.split('/')[-1] if '/' in validator_model else validator_model
            print(f"   VALIDATOR_AGENT_MODEL={clean_name}")
        print()

    if sql_provider == 'google' and (google_key == 'NOT SET' or 'your' in google_key.lower()):
        print("   # Add Google Gemini API key:")
        print("   GOOGLE_API_KEY=AIzaSy-your-actual-key-here")
        print("   Get from: https://aistudio.google.com/app/apikey")
        print()

    if 'deepseek' in [sql_provider, validator_provider] and (
            deepseek_key == 'NOT SET' or 'your' in deepseek_key.lower()):
        print("   # Add DeepSeek API key:")
        print("   DEEPSEEK_API_KEY=sk-your-actual-key-here")
        print("   Get from: https://platform.deepseek.com/api_keys")
        print()

    print("3. Save the file")
    print("4. Run this script again: python verify_config.py")
    print("5. If all checks pass, restart Streamlit")

else:
    print("\n✅ Your configuration is correct!")
    print("\nYou can now:")
    print("1. Start Streamlit: streamlit run app.py")
    print("2. Click 'Clear Cache' in the sidebar")
    print("3. Test with a query like: 'Show me all tables'")

print("\n" + "=" * 70)
print("\n")

# Exit with error code if issues found
if issues:
    exit(1)
else:
    exit(0)