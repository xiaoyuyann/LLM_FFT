import os
import json
import traceback
from pathlib import Path
from collections import Counter
import ast

import anthropic
import pandas as pd
from dotenv import load_dotenv

model_name = "claude-3-5-sonnet-20241022"
temp = 0

# Load .env API key
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY not found – set it in .env or env.")

# Load policy simulation options
POLICY_DF = pd.read_csv("./policy_simulation_results_27.csv").reset_index()
POLICY_LIST = []
for idx, row in POLICY_DF.iterrows():
    POLICY_LIST.append(
        f"{idx}. Tax: {row['tax_percentage']}%, Fare: ${row['fare']}/ride, Driving Fee: ${row['driver_fee']}/trip "
        f"▶ Car {row['drive_time_min']} min (${row['drive_cost']}) | Transit {row['bus_time_min']} min (${row['bus_cost']})"
    )

COMMUNITIES = ["Rogers Park", "West Ridge", "Uptown", "Lincoln Square", "North Center", "Lake View", "Lincoln Park", "Near North Side",
              "Edison Park", "Norwood Park", "Jefferson Park", "Forest Glen", "North Park", "Albany Park", "Portage Park", "Irving Park",
              "Dunning", "Montclare", "Belmont Cragin", "Hermosa", "Avondale", "Logan Square", "Humboldt Park", "West Town",
              "Austin", "West Garfield Park", "East Garfield Park", "Near West Side", "North Lawndale", "South Lawndale",
              "Lower West Side", "Loop", "Near South Side", "Armour Square", "Douglas", "Oakland", "Fuller Park", "Grand Boulevard",
              "Kenwood", "Washington Park", "Hyde Park", "Woodlawn", "South Shore", "Chatham", "Avalon Park", "South Chicago",
              "Burnside", "Calumet Heights", "Roseland", "Pullman", "South Deering", "East Side", "West Pullman", "Riverdale",
              "Hegewisch", "Garfield Ridge", "Archer Heights", "Brighton Park", "McKinley Park", "Bridgeport", "New City",
              "West Elsdon", "Gage Park", "Clearing", "West Lawn", "Chicago Lawn", "West Englewood", "Englewood",
              "Greater Grand Crossing", "Ashburn", "Auburn Gresham", "Beverly", "Washington Heights", "Mount Greenwood",
              "Morgan Park", "O'Hare", "Edgewater"]

SYSTEM_PROMPT = """
You are a representative from one of the seventy-seven communities in the City of Chicago. On behalf your community, you will be participating in a referendum in which representatives from all communities vote on a set of transportation policy proposals. You are allowed to choose up to five proposals and submit your vote as a ranked list of proposals. Remember you must act in the best interests of your community.
Think step by step and submit the top five policy proposals in a descending order of preference. Return a JSON object:
{
  "community_area": "<name>",
  "thinking": {
    "disposable_income": "<summary of the disposable income of the community area>",
    "discretionary_consumption": "<summary of the discretionary consumption of the community area>",
    "accessibility": "<summary of the accessibility to resources and services by different modes of transportation in the community area>",
    "decision_rationale": "<rationale of the ranked voting decision, showing factores that influence the tradeoffs and ranking rationale, think in step by step>"
  },
  "vote": [<rank1>, <rank2>, <rank3>,<rank4>, <rank5>]
}
* The vote list must contain 5 distinct integers from 0 to 26.
* No additional keys, no Markdown, no code fences.
"""

USER_PROMPT_TEMPLATE = """
In the referendum, residents from all seventy-seven communities will vote on 27 transportation policy proposals.  
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

Evaluate these 27 policy combinations for the {community} community area.

Consider the following factors based on the unique characteristics of the community area:
1. The disposable income of the community area
2. The discretionary consumption of the community area
3. The accessibility to resources and services by different modes of transportation in the community area
Then, consider the tradeoffs and ranking rationale of the ranked voting decision.

Return top 5 ranked policies (0-26) and reasoning as a JSON object.
""".replace("{policy_options}", "\n".join(POLICY_LIST))

def get_agent_response(community: str) -> dict | None:
    try:
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temp,
            system=SYSTEM_PROMPT.strip(),
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(community=community).strip()
                }
            ]
        )
        return json.loads(message.content[0].text.strip())
    except Exception as e:
        print(f"[ERROR] {community}: {e}")
        traceback.print_exc()
        return None

def save_individual_response(community: str, data: dict, run_dir: str) -> None:
    path = Path(run_dir) / "responses" / f"{community.lower().replace(' ', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def compile_responses(run_dir: str) -> str:
    rows = []
    for community in COMMUNITIES:
        path = Path(run_dir) / "responses" / f"{community.lower().replace(' ', '_')}.json"
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
        try:
            data = json.loads(path.read_text())
            rows.append({
                "community_area": data["community_area"],
                "disposable_income": data["thinking"]["disposable_income"],
                "discretionary_consumption": data["thinking"]["discretionary_consumption"],
                "accessibility": data["thinking"]["accessibility"],
                "decision_rationale": data["thinking"]["decision_rationale"],
                "vote": data["vote"],
                "rank1": data["vote"][0],
                "rank2": data["vote"][1],
                "rank3": data["vote"][2],
                "rank4": data["vote"][3],
                "rank5": data["vote"][4]
            })
        except Exception as e:
            print(f"[ERROR] Reading {path}: {e}")

    csv_path = Path(run_dir) / "combined_ranked_voting_results.csv"
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"✓ {csv_path.name} generated.")
    else:
        print("No data collected – nothing to compile.")
    return str(csv_path)


def run_irv_and_generate_summary(vote_data_path, output_json_path):
    df = pd.read_csv(vote_data_path)
    df['vote_list'] = df['vote'].apply(ast.literal_eval)
    votes = df['vote_list'].tolist()

    irv_history = []
    round_num = 1

    while True:
        first_choices = [vote[0] for vote in votes if vote]
        vote_counts = Counter(first_choices)
        total_votes = sum(vote_counts.values())

        irv_history.append({
            'round': round_num,
            'vote_counts': dict(vote_counts),
            'total_votes': total_votes,
        })

        for candidate, count in vote_counts.items():
            if count > total_votes / 2:
                result = {
                    "winning_policy": candidate,
                    "rounds": irv_history,
                    "summary": f"Policy {candidate} won with majority in round {round_num}."
                }
                with open(output_json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                return result

        min_votes = min(vote_counts.values())
        to_eliminate = [c for c, count in vote_counts.items() if count == min_votes]
        votes = [[v for v in vote if v not in to_eliminate] for vote in votes]
        round_num += 1

def batch_main(batch_version: str, runs: int = 10):
    base_dir = f"version_batch_{batch_version}"
    os.makedirs(base_dir, exist_ok=True)

    for i in range(1, runs + 1):
        i = i+8
        run_dir = os.path.join(base_dir, f"round_{i}")
        os.makedirs(os.path.join(run_dir, "responses"), exist_ok=True)

        print(f"\n🚀 Starting simulation round {i}...\n")

        for community in COMMUNITIES:
            print(f"Processing {community} …")
            resp = get_agent_response(community)
            if resp and {"community_area", "thinking", "vote"} <= resp.keys():
                save_individual_response(community, resp, run_dir)
            else:
                print(f"[WARN] Invalid or missing response for {community}")

        csv_path = compile_responses(run_dir)
        json_path = os.path.join(run_dir, "irv_summary.json")
        run_irv_and_generate_summary(csv_path, json_path)
    
    model_info_path = os.path.join(base_dir, "model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Temperature: {temp}\n")

    print("\n🎉 All simulations completed.")

if __name__ == "__main__":
    batch_main(batch_version="CA_CHI_35_claud", runs = 2) 