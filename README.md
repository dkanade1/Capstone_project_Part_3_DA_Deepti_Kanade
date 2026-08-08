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

```
# Few-Shot Prompt Template
```
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
Do Not:
- wrap the JSON in ```json
- include markdown
- include explanations
- include any text before or after the JSON

Output

{{
  "label": "Positive | Neutral | Negative",
  "confidence": "Low | Medium | High",
  "reason": "string"
}}

Review:
{review}
```
**Here is the retry logic for call_llm() failures (network error, rate limit, or non-200/error response),It  retries up to 3 times before logging a descriptive error and moving on,**
```
def call_llm(prompt, temperature=0.2, max_tokens=300):
    """
    Send a prompt to OpenRouter and return the model's response.
    Retries up to 3 times if an error occurs.
    """

    retries = 3

    for attempt in range(retries):

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:

            logging.warning(
                f"Attempt {attempt+1}/{retries} failed: {e}"
            )

            if attempt < retries - 1:
                time.sleep(5)
            else:
                logging.error(
                    "LLM request failed after 3 attempts."
                )
                return None

```
**Output showing retries after api call fails due to too many requests**
```
"C:\AI Capstone project\Part 3\.venv\Scripts\python.exe" "C:/AI Capstone project/Part 3/main.py"
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The reviewer explicitly states the material feels cheap, expresses disappointment, and indicates they will return the item.', 'template': 'Zero-shot', 'record': 0}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:openai._base_client:Retrying request to /chat/completions in 24.000000 seconds
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
```

**Here is the error handling code for Json parsing.It removes the leading and  trailing quotes from the Json response and load it**
```
        try:
            response = re.sub(r"^```json\s*", "", response.strip())
            response = re.sub(r"^```\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            parsed = json.loads(response)

            parsed["template"] = template_name
            parsed["record"] = idx

            results.append(parsed)

            print(parsed)

        except json.JSONDecodeError:

            logging.error(
                f"JSON parsing failed "
                f"(Template={template_name}, Record={idx})"
            )
```
 **Here is the output of call_llm executed for each prompt template(zero shot, few shot and role based prompt) on 5 reviews and its comparison***

```
"C:\AI Capstone project\Part 3\.venv\Scripts\python.exe" "C:/AI Capstone project/Part 3/main.py"
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The reviewer explicitly states the material feels cheap, expresses disappointment, and indicates they will return the item.', 'template': 'Zero-shot', 'record': 0}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:openai._base_client:Retrying request to /chat/completions in 24.000000 seconds
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The review repeatedly criticizes the product for being itchy, uncomfortable, thin, flimsy, and lacking support, indicating a clear negative sentiment.', 'template': 'Zero-shot', 'record': 1}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Neutral', 'confidence': 'Medium', 'reason': 'The reviewer notes that the dress arrived on time and the color/style match the description, indicating some satisfaction. However, they also mention it is not exactly what they imagined, showing a mild disappointment. The overall tone balances positive and negative points, leading to a neutral classification.', 'template': 'Zero-shot', 'record': 2}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Positive', 'confidence': 'High', 'reason': 'The review highlights several positive aspects—nice color, good match with skirts and pants, and overall satisfaction with the look—while the only negative points are about sizing and fit. The tone is constructive rather than critical, and the reviewer ultimately kept the item, indicating a favorable overall impression.', 'template': 'Zero-shot', 'record': 3}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:openai._base_client:Retrying request to /chat/completions in 24.000000 seconds
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Positive', 'confidence': 'High', 'reason': 'The review contains enthusiastic praise such as "love this shirt," "very flattering," and "perfect length," indicating a clear positive sentiment.', 'template': 'Zero-shot', 'record': 4}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The reviewer expresses strong dissatisfaction, noting the material feels cheap, disappointment, and plans to return the item.', 'template': 'Few-shot', 'record': 0}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The reviewer expresses strong dissatisfaction with the product, citing itchy tags, lack of comfort, and insufficient support, indicating a negative overall sentiment.', 'template': 'Few-shot', 'record': 1}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Neutral', 'confidence': 'Medium', 'reason': 'The review notes timely delivery and that the color and style match the description, but also expresses mild disappointment that it did not meet the buyer’s exact expectations.', 'template': 'Few-shot', 'record': 2}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Neutral', 'confidence': 'Medium', 'reason': 'The review highlights positive aspects (good color, versatile pairing) while also noting sizing and fit issues, resulting in a balanced overall sentiment.', 'template': 'Few-shot', 'record': 3}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Positive', 'confidence': 'High', 'reason': 'The reviewer highlights many positive aspects—flattering fit, adjustable tie, perfect length, sleeveless design, versatility with cardigans—and ends with "love this shirt!!!" indicating strong satisfaction.', 'template': 'Few-shot', 'record': 4}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The review expresses disappointment, criticizes the material as cheap, and states the item will be returned.', 'template': 'Role', 'record': 0}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Negative', 'confidence': 'High', 'reason': 'The review expresses dissatisfaction with the product’s comfort, support, and quality, describing it as itchy, thin, flimsy, and lacking support.', 'template': 'Role', 'record': 1}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Neutral', 'confidence': 'Medium', 'reason': 'The review acknowledges timely delivery and similarity to description, but also notes a mismatch with expectations, indicating a balanced sentiment.', 'template': 'Role', 'record': 2}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Positive', 'confidence': 'Medium', 'reason': 'The review highlights several positive points about the top—its color, style compatibility, and overall look—while mentioning only a few minor sizing issues. The overall tone is favorable, indicating a positive sentiment.', 'template': 'Role', 'record': 3}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'label': 'Positive', 'confidence': 'High', 'reason': 'The review expresses strong positive feelings, praising the shirt’s flattering fit, adjustable front tie, perfect length, and versatility, ending with "love this shirt!!!"', 'template': 'Role', 'record': 4}
       label confidence                                                                                                                                                                                                                                                                                                                             reason   template  record
0   Negative       High                                                                                                                                                                                                        The reviewer explicitly states the material feels cheap, expresses disappointment, and indicates they will return the item.  Zero-shot       0
1   Negative       High                                                                                                                                                                             The review repeatedly criticizes the product for being itchy, uncomfortable, thin, flimsy, and lacking support, indicating a clear negative sentiment.  Zero-shot       1
2    Neutral     Medium            The reviewer notes that the dress arrived on time and the color/style match the description, indicating some satisfaction. However, they also mention it is not exactly what they imagined, showing a mild disappointment. The overall tone balances positive and negative points, leading to a neutral classification.  Zero-shot       2
3   Positive       High  The review highlights several positive aspects—nice color, good match with skirts and pants, and overall satisfaction with the look—while the only negative points are about sizing and fit. The tone is constructive rather than critical, and the reviewer ultimately kept the item, indicating a favorable overall impression.  Zero-shot       3
4   Positive       High                                                                                                                                                                                 The review contains enthusiastic praise such as "love this shirt," "very flattering," and "perfect length," indicating a clear positive sentiment.  Zero-shot       4
5   Negative       High                                                                                                                                                                                                      The reviewer expresses strong dissatisfaction, noting the material feels cheap, disappointment, and plans to return the item.   Few-shot       0
6   Negative       High                                                                                                                                                             The reviewer expresses strong dissatisfaction with the product, citing itchy tags, lack of comfort, and insufficient support, indicating a negative overall sentiment.   Few-shot       1
7    Neutral     Medium                                                                                                                                                   The review notes timely delivery and that the color and style match the description, but also expresses mild disappointment that it did not meet the buyer’s exact expectations.   Few-shot       2
8    Neutral     Medium                                                                                                                                                                         The review highlights positive aspects (good color, versatile pairing) while also noting sizing and fit issues, resulting in a balanced overall sentiment.   Few-shot       3
9   Positive       High                                                                                                                     The reviewer highlights many positive aspects—flattering fit, adjustable tie, perfect length, sleeveless design, versatility with cardigans—and ends with "love this shirt!!!" indicating strong satisfaction.   Few-shot       4
10  Negative       High                                                                                                                                                                                                                       The review expresses disappointment, criticizes the material as cheap, and states the item will be returned.       Role       0
11  Negative       High                                                                                                                                                                                  The review expresses dissatisfaction with the product’s comfort, support, and quality, describing it as itchy, thin, flimsy, and lacking support.       Role       1
12   Neutral     Medium                                                                                                                                                                               The review acknowledges timely delivery and similarity to description, but also notes a mismatch with expectations, indicating a balanced sentiment.       Role       2
13  Positive     Medium                                                                                                      The review highlights several positive points about the top—its color, style compatibility, and overall look—while mentioning only a few minor sizing issues. The overall tone is favorable, indicating a positive sentiment.       Role       3
14  Positive       High                                                                                                                                                        The review expresses strong positive feelings, praising the shirt’s flattering fit, adjustable front tie, perfect length, and versatility, ending with "love this shirt!!!"       Role       4

Process finished with exit code 0

```

**The few-shot,zero shot  and role-prompted templates performed well in terms of classification results and JSON/schema conformity.Only the 4th  review was classified as Positive by zero-shot and role-prompted ,whereas Neutral by few-shot template. All achieved valid, schema-conformant responses.But based on the Output quality, i.e the reason being concise and relevant ,the role based prompt can be considered superior templates in this particular case.**

## Task 5.	Build an aspect-based sentiment extension
**Modified prompt**
```
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
```
**Result of aspect-based sentiment analysis in a tabular format for fit_sizing and material_quality**
```
   record fit_sentiment                          fit_action material_sentiment                material_action
0       0       Neutral                No fit info provided           Negative           Material feels cheap
1       1      Negative             Lacks support for B cup           Negative         Itchy, flimsy material
2       2       Neutral                  No feedback on fit            Neutral        No feedback on material
3       3      Negative        Size too large, sleeves long            Neutral           No material feedback
4       4      Positive         Adjustable tie enhances fit            Neutral         Material not mentioned
5       5      Negative          Too large for petite frame            Neutral  No material feedback provided
6       6      Negative             Fit too loose, XS small            Neutral         Material not discussed
7       7      Negative               Too big, sleeves long            Neutral         No comment on material
8       8      Positive      Flattering fit after sizing up            Neutral         No mention of material
9       9      Positive  Fits perfectly, snug but not tight           Positive  Tulle longer than base fabric

Process finished with exit code 0
```
## Task 6: Chained Response Drafting

The structured aspect sentiment output from Task 5 was passed into a second
LLM prompt to generate a customer-facing response. The drafting prompt was
instructed to address the specific issues identified for each record rather
than producing a generic response
```
"C:\AI Capstone project\Part 3\.venv\Scripts\python.exe" "C:/AI Capstone project/Part 3/main.py"
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'fit_sizing': {'sentiment': 'Neutral', 'action': 'No comment on fit'}, 'material_quality': {'sentiment': 'Negative', 'action': 'Material feels cheap and low quality'}, 'template': 'Role', 'record': 0}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
DRAFTED REPLY:
I'm sorry to hear that the material feels cheap; we appreciate your feedback and will review our quality standards to ensure a better experience.
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'fit_sizing': {'sentiment': 'Negative', 'action': 'Lacks support for B cup'}, 'material_quality': {'sentiment': 'Negative', 'action': 'Itchy tags, uncomfortable'}, 'template': 'Role', 'record': 1}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
DRAFTED REPLY:
{"response":"I’m sorry to hear the bra doesn’t provide enough support for your B cup and that the tags are itchy; we appreciate your feedback and will work to improve both fit and comfort."}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
INFO:openai._base_client:Retrying request to /chat/completions in 22.000000 seconds
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'fit_sizing': {'sentiment': 'Neutral', 'action': 'No fit details provided'}, 'material_quality': {'sentiment': 'Neutral', 'action': 'No material details provided'}, 'template': 'Role', 'record': 2}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
DRAFTED REPLY:
{"reply":"Thank you for contacting us; we’re happy to help with any questions about fit, sizing, or material."}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
{'fit_sizing': {'sentiment': 'Negative', 'action': 'Size runs large, sleeves long'}, 'material_quality': {'sentiment': 'Neutral', 'action': 'No material comments'}, 'template': 'Role', 'record': 3}
INFO:httpx:HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK
DRAFTED REPLY:
I’m sorry to hear that the size runs large and the sleeves are longer than expected. We appreciate your feedback and will keep it in mind as we continue to improve our fit.

   Record Fit Sentiment                     Fit_Action Material sentiment                       Material Action                                                                                                                                                                                Manager Response
0       0       Neutral              No comment on fit           Negative  Material feels cheap and low quality                                               I'm sorry to hear that the material feels cheap; we appreciate your feedback and will review our quality standards to ensure a better experience.
1       1      Negative        Lacks support for B cup           Negative             Itchy tags, uncomfortable  {"response":"I’m sorry to hear the bra doesn’t provide enough support for your B cup and that the tags are itchy; we appreciate your feedback and will work to improve both fit and comfort."}
2       2       Neutral        No fit details provided            Neutral          No material details provided                                                                                 {"reply":"Thank you for contacting us; we’re happy to help with any questions about fit, sizing, or material."}
3       3      Negative  Size runs large, sleeves long            Neutral                  No material comments                    I’m sorry to hear that the size runs large and the sleeves are longer than expected. We appreciate your feedback and will keep it in mind as we continue to improve our fit.

Process finished with exit code 0

```
