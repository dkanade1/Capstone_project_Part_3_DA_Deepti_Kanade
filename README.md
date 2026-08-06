# Part 3 — GenAI-Powered Text Analytics: Prompt Engineering & LLM API Integration 
### Database Schema :Women's E-Commerce Clothing Reviews
This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Each row corresponds to a customer review, and includes the variables:Clothing ID,Age,Title,Review Text,Rating,Recommended IND,Positive Feedback Count,Division Name,Department Name, Class Name.
## Prerequisites
Before getting started, make sure your development environment meets the following requirements:
Python: Version 3.11 


## Task 1.	Design three prompt templates 
# Zero-Shot Prompt Template
```
You are given a customer review from a women's e-commerce clothing store.

Task:
Classify the overall sentiment expressed in the review text.

Sentiment labels:
- Positive
- Neutral
- Negative

Instructions:
- Base your decision only on the review text.
- Ignore any numerical rating or metadata.
- Choose the single best sentiment label.
- Estimate your confidence.
- Briefly explain the evidence supporting your decision.

Respond ONLY using the following JSON schema:

{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}

Review:
{{review_text}}

```
# Few-Shot Prompt Template
```
You are given a customer review from a women's e-commerce clothing store.

Task:
Classify the overall sentiment expressed in the review text.

Sentiment labels:
- Positive
- Neutral
- Negative

Respond ONLY using this JSON schema:

{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}

Examples

Example 1

Review:
"This dress fits perfectly and the fabric feels luxurious. I receive compliments every time I wear it."

Response:
{
  "label": "Positive",
  "confidence": "High",
  "reason": "The reviewer praises the fit, material quality, and overall satisfaction."
}

Example 2

Review:
"The color is nice, but the sleeves are too long and the material wrinkles easily."

Response:
{
  "label": "Neutral",
  "confidence": "Medium",
  "reason": "The review contains both positive and negative opinions without an overall clearly positive or negative impression."
}

Example 3

Review:
"I returned it immediately. Poor stitching, terrible fit, and not worth the money."

Response:
{
  "label": "Negative",
  "confidence": "High",
  "reason": "The reviewer expresses strong dissatisfaction with multiple product aspects."
}

Now classify the following review.

Review:
{{review_text}}
```
# Role-Prompted Template (Using ECO Framework)
```
Role

Act as a senior customer-insights analyst specializing in e-commerce product reviews. Your job is to accurately classify customer sentiment while remaining objective and consistent.

Instruction

Determine the overall sentiment expressed in the customer review.

Context

The review comes from the Women's E-Commerce Clothing Reviews dataset. Reviews discuss clothing products such as dresses, tops, sweaters, jackets, and jeans. Customers may mention product quality, fit, sizing, comfort, style, fabric, or value. Your classification should be based only on the review text, regardless of any numerical rating.

Constraints

- Use only one sentiment label:
  - Positive
  - Neutral
  - Negative
- Base the decision solely on the review text.
- Do not infer information not stated.
- Provide a brief evidence-based explanation.
- Respond only in valid JSON.
- Do not include markdown, comments, or additional text.

Output

{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}

Review:
{{review_text}}
```
