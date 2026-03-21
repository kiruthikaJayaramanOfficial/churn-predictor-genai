import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

def get_churn_playbook(churn_probability: float, shap_result: dict,
                       customer_info: dict, stream_callback=None) -> str:
    """
    Takes SHAP output → generates plain-English Churn Playbook via Groq
    """
    top_pos = shap_result['top_positive']
    top_neg = shap_result['top_negative']

    risk_reasons = []
    for item in top_pos:
        risk_reasons.append(f"- {item['feature']} = {item['feature_value']:.2f} (impact: +{item['shap_value']:.3f})")

    protective_factors = []
    for item in top_neg:
        protective_factors.append(f"- {item['feature']} = {item['feature_value']:.2f} (impact: {item['shap_value']:.3f})")

    risk_level = "HIGH" if churn_probability > 0.7 else "MEDIUM" if churn_probability > 0.4 else "LOW"

    prompt = f"""You are a senior customer retention analyst at a telecom company.

A customer has a {churn_probability*100:.1f}% churn probability. Risk level: {risk_level}

TOP REASONS THIS CUSTOMER WILL CHURN (SHAP values):
{chr(10).join(risk_reasons)}

PROTECTIVE FACTORS (reducing churn risk):
{chr(10).join(protective_factors) if protective_factors else "- None significant"}

CUSTOMER PROFILE:
- Tenure: {customer_info.get('tenure', 'Unknown')} months
- Monthly Charges: ${customer_info.get('MonthlyCharges', 'Unknown')}
- Senior Citizen: {'Yes' if customer_info.get('SeniorCitizen', 0) == 1 else 'No'}

Write a Churn Playbook with exactly these 3 sections:

## 🔍 Why This Customer Will Churn
Write 2-3 sentences explaining the key churn drivers in plain English (no jargon).

## 🎯 Retention Actions (do these TODAY)
List exactly 3 specific, actionable retention steps. Be concrete — name the exact offer, discount %, or action.

## ⚡ Urgency Level
State: {risk_level} PRIORITY
Give one sentence on timing (e.g., "Contact within 24 hours before next billing cycle").
"""

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=True
    )

    full_response = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        token = chunk.content
        full_response += token
        if stream_callback:
            stream_callback(token)

    return full_response


if __name__ == "__main__":
    # Quick test
    test_shap = {
        'top_positive': [
            {'feature': 'Contract_Month-to-month', 'shap_value': 0.45, 'feature_value': 1},
            {'feature': 'tenure', 'shap_value': 0.32, 'feature_value': 3},
            {'feature': 'MonthlyCharges', 'shap_value': 0.28, 'feature_value': 85.5},
        ],
        'top_negative': [
            {'feature': 'TechSupport', 'shap_value': -0.15, 'feature_value': 0},
        ]
    }
    test_customer = {'tenure': 3, 'MonthlyCharges': 85.5, 'SeniorCitizen': 0}

    print("Testing LangChain agent...")
    result = get_churn_playbook(0.73, test_shap, test_customer,
                                stream_callback=lambda x: print(x, end='', flush=True))
    print("\n\n✅ Agent test complete!")