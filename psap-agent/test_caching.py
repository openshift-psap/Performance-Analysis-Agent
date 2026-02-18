#!/usr/bin/env python3
"""Test script to verify Gemini API context caching is working."""

import os
from google import genai

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY environment variable not set!")
    exit(1)

# Create client
client = genai.Client(api_key=api_key)

# Use the cached content (from agent startup logs)
cache_name = "cachedContents/8ts8l3qvz6rji6yykm9d0sm4qfi4jvrzeqhgh2pn"

print("🔍 Testing Gemini API Context Caching...")
print(f"📦 Using cache: {cache_name}")
print("-" * 70)

try:
    # Generate content using cached content
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is 2 + 2?",
        config={
            "cached_content": cache_name
        }
    )
    
    # Print usage metadata
    usage = response.usage_metadata
    
    print("\n✅ SUCCESS! Caching is working!\n")
    print("📊 Token Usage:")
    print(f"  • Prompt tokens (your query):     {usage.prompt_token_count:,}")
    print(f"  • Cached tokens (system prompt):  {usage.cached_content_token_count:,}")
    print(f"  • Completion tokens (response):   {usage.candidates_token_count:,}")
    print(f"  • Total tokens:                   {usage.total_token_count:,}")
    
    # Calculate cost savings
    if usage.cached_content_token_count > 0:
        full_price_cost = usage.cached_content_token_count * 0.30 / 1_000_000
        cached_price_cost = usage.cached_content_token_count * 0.03 / 1_000_000
        savings = full_price_cost - cached_price_cost
        
        print(f"\n💰 Cost Analysis (for cached tokens):")
        print(f"  • Without caching: ${full_price_cost:.6f}")
        print(f"  • With caching:    ${cached_price_cost:.6f}")
        print(f"  • Savings:         ${savings:.6f} ({(savings/full_price_cost)*100:.1f}% off!)")
    
    print(f"\n📝 Response: {response.text[:100]}...")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nThis might mean:")
    print("  1. Cache has expired (TTL = 4 hours)")
    print("  2. Agent needs to be restarted to create a new cache")
    print("  3. API key is invalid")
    
    exit(1)

