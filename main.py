import json
import re
import logging
import pandas as pd
from llm import call_llm

from prompts import (
     ZERO_SHOT,
     FEW_SHOT,
    ROLE_PROMPT,
RESPONSE_PROMPT,
 )


logging.basicConfig(level=logging.INFO)

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

reviews = (
    df["Review Text"]
    .dropna()
    .head(4)
)

## Commented for task 5
templates = {
#    "Zero-shot": ZERO_SHOT,
#    "Few-shot": FEW_SHOT,
    "Role": ROLE_PROMPT,
}


results = []

for template_name, template in templates.items():

    for idx, review in enumerate(reviews):

        prompt = template.format(review=review)

        response = call_llm(
            prompt,
            temperature=0.1,
            max_tokens=1000,
        )

        if response is None:
            continue

        try:
            print("####Raw response")
            print(response)
            response = re.sub(r"^```json\s*", "", response.strip())
            response = re.sub(r"^```\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            parsed = json.loads(response)

            parsed["template"] = template_name
            parsed["record"] = idx

            print(parsed)
        except json.JSONDecodeError:

            logging.error(
                f"JSON parsing failed "
                f"(Template={template_name}, Record={idx})"
            )
            continue
            # -----------------------------
            # TASK 6: Response drafting
            # -----------------------------

        aspect_json = json.dumps(parsed, indent=2)

        response_prompt = RESPONSE_PROMPT.format(
                aspect_analysis=aspect_json
        )

        drafted_reply = call_llm(
                response_prompt,
                temperature=0.2,
                max_tokens=1000
        )

        if drafted_reply is None:
                logging.error(
                    f"Response drafting failed: Record={idx}"
                )
                continue

        print("\nDRAFTED REPLY:")
        print(drafted_reply)

        # -----------------------------
        # Save results
        # -----------------------------
        results.append({
            "Record": idx,
            "Fit Sentiment": parsed["fit_sizing"]["sentiment"],
            "Fit_Action": parsed["fit_sizing"]["action"],
            "Material sentiment": parsed["material_quality"]["sentiment"],
            "Material Action": parsed["material_quality"]["action"],
            "Manager Response": drafted_reply
        })




results_df = pd.DataFrame(results)

#results_df.to_csv(
#    "classification_results.csv",
#    index=False,
#)
# results_df.to_csv(
#     "aspect_sentiment_results.csv",
#     index=False
#)
results_df.to_csv(
    "aspect_sentiment_results.csv",
    index=False
)
print(results_df.to_string())

# ============================================================
# TASK 7: Multi-turn conversation
# ============================================================

print("\n" + "=" * 60)
print("TASK 7: MULTI-TURN CONTEXT")
print("=" * 60)

# Turn 1
turn1_user = """
Analyze the following customer review and identify the main issue,
including any relevant details about fit and the positive aspects
mentioned by the customer.

Customer review:
I took these out of the package and wanted them to fit so badly,
but i could tell before i put them on that they wouldn't. these
are for an hour-glass figure. i am more straight up and down.
the waist was way too small for my body shape and even if i sized up,
i could tell they would still be tight in the waist and too roomy
in the hips - for me.
that said, they are really nice. sturdy, linen-like fabric, pretty
color, well made. i hope they make someone very happy!
"""

turn1_prompt = turn1_user

turn1_response = call_llm(
    turn1_prompt,
    temperature=0.2,
    max_tokens=500
)

print("\nTURN 1")
print("User:")
print(turn1_user)

print("Assistant:")
print(turn1_response)


# ------------------------------------------------------------
# Conversation history
# ------------------------------------------------------------

conversation_history = [
    {
        "role": "user",
        "content": turn1_user.strip()
    },
    {
        "role": "assistant",
        "content": turn1_response.strip()
    }
]


# Turn 2
turn2_user = """
Write a short, professional and empathetic customer-service response
to this customer. Use the details from our previous conversation.
Acknowledge both their fit concern and the positive comments they made.
Do not ask them to repeat their issue.
"""

conversation_history.append({
    "role": "user",
    "content": turn2_user.strip()
})


# Create the prompt containing the conversation history
history_text = "\n\n".join(
    f"{message['role'].upper()}: {message['content']}"
    for message in conversation_history
)

turn2_response = call_llm(
    history_text,
    temperature=0.2,
    max_tokens=1000
)

conversation_history.append({
    "role": "assistant",
    "content": turn2_response.strip()
})

print("\nTURN 2")
print("User:")
print(turn2_user)

print("Assistant:")
print(turn2_response)


# ------------------------------------------------------------
# Show conversation history object
# ------------------------------------------------------------

print("\nCONVERSATION HISTORY OBJECT:")
print(json.dumps(conversation_history, indent=2))

