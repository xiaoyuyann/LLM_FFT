import os
import json
import traceback
from pathlib import Path
from collections import Counter
import ast
import argparse

import pandas as pd
from dotenv import load_dotenv

# Model and API selection
def get_model_and_client(model_choice):
    load_dotenv()
    if model_choice == "gpt-4o":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found – set it in .env or env.")
        client = OpenAI(api_key=api_key)
        model_name = "gpt-4o-2024-08-06"
        return model_name, client, "openai"
    elif model_choice == "claude":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found – set it in .env or env.")
        client = anthropic.Anthropic(api_key=api_key)
        model_name = "claude-3-5-sonnet-20241022"
        return model_name, client, "anthropic"
    else:
        raise ValueError("Unknown model_choice. Use 'gpt-4o' or 'claude'.")

# Load policy simulation options
def load_policy_df(model_choice):
    if model_choice == "gpt-4o":
        policy_path = "./policy_simulation_Chicago.csv"
    else:
        policy_path = "./policy_simulation_Chicago.csv"
    return pd.read_csv(policy_path).reset_index()

def build_policy_list(policy_df):
    policy_list = []
    for idx, row in policy_df.iterrows():
        policy_list.append(
            f"{idx}. Tax: {row['tax_percentage']}%, Fare: ${row['fare']}/ride, Driving Fee: ${row['driver_fee']}/trip "
            f"▶ Car {row['drive_time_min']} min (${row['drive_cost']}) | Transit {row['bus_time_min']} min (${row['bus_cost']})"
        )
    return policy_list

SYSTEM_PROMPT = """
You are a representative from the City of Chicago. On behalf of the City of Chicago, you will be participating in a referendum in which you vote on a set of transportation policy proposals. You are allowed to choose up to five proposals and submit your vote as a ranked list of proposals. Remember you must act in the best interests of the City of Chicago.
Think step by step and submit the top five policy proposals in a descending order of preference. Return a JSON object:
{
  "community_area": "<name>",
  "thinking": {
    "disposable_income": "<summary of the disposable income of the community area>",
    "discretionary_consumption": "<summary of the discretionary consumption of the community area>",
    "accessibility": "<summary of the accessibility to resources and services by different modes of transportation in the community area>",
    "decision_rationale": "<rationale of the ranked voting decision, showing factores that influence the tradeoffs and ranking rationale, think in step by step>"
  "vote": [<rank1>, <rank2>, <rank3>,<rank4>, <rank5>]
}
* The vote list must contain 5 distinct integers from 0 to 26.
* No additional keys, no Markdown, no code fences.
"""

def build_user_prompt(policy_list):
    return """
In the referendum, you as the representative of an average person in the City of Chicago will vote on 27 transportation policy proposals.  
A policy proposal consists of three policies: (i) transit fare policy, which may set a per-trip fare for riding transit to either $0.75, $1.25, or $1.75, 
(ii) tax policy, which may set a dedicated sales tax rate to either 0.5%, 1%, or 1.5%; 
and (iii) a driver-fee policy, which may set a per-trip fee for driving to either  $0.00, $0.50, or $1.00.   
Since there are three options for each policy, you have 27 proposals to vote on.  
You must pick the top five proposals and rank them from 1 to 5 according to how they serve your interest.  Your interest should be defined by weighing the cost and benefits of each package.  
The Chicago Metropolitan Agency for Planning (CMAP) has estimated the travel times per trip by transit and driving corresponding to each policy, along with the corresponding fare and fees. 
So, you should use this information to gauge your interest.  You should also be aware that paying either fare for transit or a fee for driving will take a portion of your income away from other consumption (such as food, housing, cloth, vacation, tuition etc.).  
Hence, you should carefully assess how such a loss of income might affect your life.  In doing so, please bear in mind that you are acting as a representative resident in your community – so you should try to use as much data (census, employment, transit availability, car ownership, etc.) from your community to make your decision.

Here is some other information that might help your deliberations:
1.  The dedicated sales tax is meant to support transit services. The higher the tax, the more money the transit ANONYMOUS
2.  If you ride in transit, you will pay fare; if you drive, you will pay the driver fee.  All revenues from fare and driver fee will be used to support transit, which means higher fare and driver fee generally mean better transit services. However, higher fare and driver fees also mean you have less money to spend on other goods and services.  Also, remember the fare and fee are paid on the per trip basis.   
3.  Driver fees and better transit might persuade some drivers to switch from driving to transit. This could help reduce congestion, lowering driving time and reducing emissions.  

Current Transportation Policy and the travel time and cost:
the transit fare is $1.25/ride, the sales tax is 1%, and the driving fee is $0.00/trip. 
The travel time by car is 21 minutes, and the travel time by transit is 58.8 minutes, the cost per trip by car is $5.43, and the cost per trip by transit is $1.25.

Candidate Policy Options and the associated travel time and cost estimated by Chicago Metropolitan Agency for Planning:
{policy_options}

Evaluate these 27 policy combinations for the City of Chicago.

Consider the following factors based on the unique characteristics of the City of Chicago:
1. The disposable income of the City of Chicago
2. The discretionary consumption of the City of Chicago
3. The accessibility to resources and services by different modes of transportation in the City of Chicago
Then, consider the tradeoffs and ranking rationale of the ranked voting decision.

Return top 5 ranked policies (0-26) and reasoning as a JSON object.
""".replace("{policy_options}", "\n".join(policy_list))

def get_avg_agent_response(model_choice, model_name, client, user_prompt):
    try:
        if model_choice == "gpt-4o":
            # OpenAI API
            chat = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=120,
            )
            return json.loads(chat.choices[0].message.content)
        elif model_choice == "claude":
            # Anthropic API
            message = client.messages.create(
                model=model_name,
                max_tokens=2048,
                temperature=0,
                system=SYSTEM_PROMPT.strip(),
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt.strip()
                    }
                ]
            )
            return json.loads(message.content[0].text.strip())
        else:
            raise ValueError("Unknown model_choice. Use 'gpt-4o' or 'claude'.")
    except Exception as e:
        print(f"[ERROR] Average Chicagoan: {e}")
        traceback.print_exc()
        return None

def save_avg_response(data: dict, out_dir: str, run_idx: int) -> None:
    path = Path(out_dir) / f"avg_chicagoan_response_{run_idx+1}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def extract_result_for_csv(data: dict, run_idx: int) -> dict:
    # Flatten the relevant fields for CSV
    result = {
        "run": run_idx + 1,
        "community_area": data.get("community_area", ""),
        "disposable_income": data.get("thinking", {}).get("disposable_income", ""),
        "discretionary_consumption": data.get("thinking", {}).get("discretionary_consumption", ""),
        "accessibility": data.get("thinking", {}).get("accessibility", ""),
        "decision_rationale": data.get("thinking", {}).get("decision_rationale", ""),
    }
    vote = data.get("vote", [])
    for i in range(5):
        result[f"vote_{i+1}"] = vote[i] if i < len(vote) else None
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chicago policy agent with selectable model.")
    parser.add_argument("--model", choices=["gpt-4o", "claude"], default="claude", help="Which model to use: gpt-4o or claude")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to run")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    model_choice = args.model
    num_rounds = args.rounds
    output_dir = args.output_dir or (f"CA_CHI_{model_choice}_avg")

    model_name, client, _ = get_model_and_client(model_choice)
    POLICY_DF = load_policy_df(model_choice)
    POLICY_LIST = build_policy_list(POLICY_DF)
    USER_PROMPT_TEMPLATE = build_user_prompt(POLICY_LIST)

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    print(f"Requesting response from average Chicagoan agent for {num_rounds} rounds using {model_choice}...")
    for run_idx in range(num_rounds):
        print(f"--- Round {run_idx+1} ---")
        resp = get_avg_agent_response(model_choice, model_name, client, USER_PROMPT_TEMPLATE)
        if resp and {"community_area", "thinking", "vote"} <= resp.keys():
            save_avg_response(resp, output_dir, run_idx)
            print(f"✓ Response saved to {output_dir}/avg_chicagoan_response_{run_idx+1}.json")
            all_results.append(extract_result_for_csv(resp, run_idx))
        else:
            print(f"[WARN] Invalid or missing response from average Chicagoan agent in round {run_idx+1}.")
    # Save combined results to CSV
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_csv_path = Path(output_dir) / "combined_results.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"✓ Combined results saved to {combined_csv_path}")
    else:
        print("[WARN] No valid results to save to CSV.")
