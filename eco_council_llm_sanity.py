# eco_council_llm_sanity.py

import math
import os
import tiktoken

# Part A — Build a realistic prompt string

safety_block = (
    "Stay factual and accurate in all statements. Do not invent sponsors, donors, "
    "or event outcomes. If required information is missing, clearly state that "
    "instead of guessing. Keep all claims professional, verifiable, and based "
    "only on provided details."
)

email_body = (
    "Dear GreenLeaf Motors Team,\n\n"
    "Thank you for generously supporting our college Eco-Council tree-planting "
    "drive on April 22, 2026, in Pune. Your funding helped us purchase saplings, "
    "arrange transportation, and equip student volunteers who dedicated their time "
    "to restoring community green spaces. Because of your partnership, our team "
    "successfully organized a meaningful environmental initiative that encouraged "
    "sustainability awareness among students and residents alike. Your support "
    "strengthened our mission and empowered volunteers to make a visible local "
    "impact. We deeply appreciate your commitment to environmental responsibility "
    "and community development. Thank you for helping us create a greener future.\n\n"
    "Sincerely,\n"
    "College Eco-Council"
)

follow_up_instruction = "Suggest one professional email subject line under 60 characters."

# Concatenate full prompt
full_prompt = f"{safety_block}\n\n{email_body}\n\n{follow_up_instruction}"

# Part B — Exact prompt token count with tiktoken

try:
    encoder = tiktoken.encoding_for_model("gpt-4")
except Exception:
    encoder = tiktoken.get_encoding("cl100k_base")
    print("Fallback encoder used: cl100k_base")

prompt_tokens = len(encoder.encode(full_prompt))

print(f"Prompt tokens (tiktoken): {prompt_tokens}")

# Part C — Completion budget estimate (220 words)
completion_words = 220
completion_tokens_est = completion_words / 0.75  # words ÷ 0.75

print(f"Estimated completion tokens (rule of thumb): {completion_tokens_est:.2f}")

# Part D — Context window check

CONTEXT_LIMIT = 4096

completion_tokens_fit = math.ceil(completion_tokens_est)

total_tokens_needed = prompt_tokens + completion_tokens_fit

print(f"Total tokens needed (rounded-up completion): {total_tokens_needed}")
print(f"Fits in 4096 window? {total_tokens_needed <= CONTEXT_LIMIT}")

# Part F — Optional live API temperature demo
# SDK method: client.chat.completions.create
# API key env variable: OPENAI_API_KEY

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    try:
        from openai import OpenAI

        client = OpenAI()

        messages = [
            {"role": "user", "content": "Describe how is the cloud today??"}
        ]

        response_1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )

        response_2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.9
        )

        print("\n--- temperature=0.2 ---")
        print(response_1.choices[0].message.content)

        print("\n--- temperature=0.9 ---")
        print(response_2.choices[0].message.content)

    except Exception as e:
        print(f"Live temperature demo failed: {e}")

else:
    print("Skipping live temperature demo — set OPENAI_API_KEY to enable.")

# Part E — Concept recall paragraph

concept_paragraph = (
    "The same prompt can produce different answers because language models generate "
    "responses by predicting one token at a time based on probabilities, not by "
    "retrieving a single fixed script. Multiple possible next tokens may all be "
    "reasonable, and sampling choices can vary depending on randomness or settings "
    "such as temperature. Once one token is selected, it influences every following "
    "token, creating a slightly different response path. This is why repeated calls "
    "can change wording, creativity, or detail while still answering the same core "
    "request. In short, outputs are shaped by probabilistic next-token generation."
)

print(concept_paragraph)