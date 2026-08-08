ROLE_PROMPT = """
Role

Act as a senior customer-insights analyst specializing in women's fashion e-commerce reviews.

Instruction

Analyze the customer review and determine the sentiment for each aspect.

Context

The review comes from a women's clothing e-commerce store.

Evaluate these aspects:

1. Fit & Sizing
2. Material & Quality

Constraints

- For each aspect assign exactly one sentiment:
  - Positive
  - Neutral
  - Negative
- Base your decision only on the review text.
- If an aspect is not mentioned, assign "Neutral".
- Provide one short actionable phrase (3–6 words) describing what the customer liked or disliked for each aspect.
- Respond ONLY with valid JSON.
- Do not use markdown.
- Do not include any additional text.

Output

{{
  "fit_sizing": {{
    "sentiment": "Positive | Neutral | Negative",
    "action": "3-6 word phrase"
  }},
  "material_quality": {{
    "sentiment": "Positive | Neutral | Negative",
    "action": "3-6 word phrase"
  }}
}}

Review:
{review}
"""
RESPONSE_PROMPT = """
Act as a professional and empathetic customer service representative for a
women's clothing e-commerce store.

Instruction

Write a short, professional, and empathetic response to the customer based
on the structured sentiment analysis provided below.

Context

The structured analysis identifies the customer's sentiment about:
- Fit & Sizing
- Material & Quality

Use these specific findings to make the response relevant to this customer.

Constraints

- Address the specific issues mentioned in the analysis.
- Acknowledge positive points when present.
- Show empathy for negative experiences.
- Do not invent refunds, discounts, replacements, or other actions unless
  explicitly stated in the input.
- Keep the response concise: 1-2 sentences.
- Use a professional and friendly customer-service tone.
- Do not mention sentiment labels or the analysis.
- Respond with ONLY the customer-facing reply.
- Do not use JSON or Markdown.

Structured analysis:
{aspect_analysis}
"""

ZERO_SHOT = """
You are given a customer review from a women's e-commerce clothing store.
Task:
Classify the overall sentiment expressed in the review text.Estimate your confidence and Briefly explain the evidence supporting your decision.
Sentiment labels:
- Positive
- Neutral
- Negative

Respond ONLY using the following JSON schema:
{{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}}


Review:
{review}
"""

FEW_SHOT = """
You are given a customer review from a women's e-commerce clothing store.

Task:
Classify the overall sentiment expressed in the review text.Estimate your confidence and Briefly explain the evidence supporting your decision.

Sentiment labels:
- Positive
- Neutral
- Negative

Respond ONLY using this JSON schema:
{{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}}

Examples

Example 1

Review:
"This dress fits perfectly and the fabric feels luxurious. I receive compliments every time I wear it."

Response:
{{
  "label": "Positive",
  "confidence": "High",
  "reason": "The reviewer praises the fit, material quality, and overall satisfaction."
}}

Example 2

Review:
"The color is nice, but the sleeves are too long and the material wrinkles easily."

Response:
{{
  "label": "Neutral",
  "confidence": "Medium",
  "reason": "The review contains both positive and negative opinions without an overall clearly positive or negative impression."
}}

Example 3

Review:
"I returned it immediately. Poor stitching, terrible fit, and not worth the money."

Response:
{{
  "label": "Negative",
  "confidence": "High",
  "reason": "The reviewer expresses strong dissatisfaction with multiple product aspects."
}}

Now classify the following review.

Review:
{review}
"""
## Commented for task 5
# ROLE_PROMPT = """
# Role
#
# Act as a senior customer-insights analyst specializing in e-commerce product reviews. Your job is to accurately classify customer sentiment while remaining objective and consistent.
#
# Instruction
#
# Determine the overall sentiment expressed in the customer review.
#
# Context
#
# The review comes from the Women's E-Commerce Clothing Reviews dataset. Reviews discuss clothing products such as dresses, tops, sweaters, jackets, and jeans. Customers may mention product quality, fit, sizing, comfort, style, fabric, or value. Your classification should be based only on the review text, regardless of any numerical rating.
#
# Constraints
#
# - Use only one sentiment label:
#   - Positive
#   - Neutral
#   - Negative
# - Base the decision solely on the review text.
# - Do not infer information not stated.
# - Provide a brief evidence-based explanation.
# - Respond only in valid JSON.
# Do Not:
# - wrap the JSON in ```json
# - include markdown
# - include explanations
# - include any text before or after the JSON
#
# Output
#
# {{
#   "label": "Positive | Neutral | Negative",
#   "confidence": "Low | Medium | High",
#   "reason": "string"
# }}
#
# Review:
# {review}
# """