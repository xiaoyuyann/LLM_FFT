import os
import json
import traceback
from pathlib import Path
from collections import Counter
import ast
import re
import pandas as pd
from dotenv import load_dotenv

# Optional: import both OpenAI and anthropic, but only use one depending on model
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None

# --- Model selection and client setup ---
# Set model_name to either "gpt-4o-2024-08-06" or "claude-3-5-sonnet-20241022"
model_name = os.getenv("MODEL_NAME", 'gpt-4o-2024-08-06')# "claude-3-5-sonnet-20241022")# "gpt-4o-2024-08-06")  # override with env if desired
temp = 0


# Load .env API key(s)
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

if model_name.startswith("gpt-"):
    if not OpenAI:
        raise ImportError("openai package not installed.")
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not found – set it in .env or env.")
    client = OpenAI(api_key=OPENAI_KEY)
    model_family = "openai"
elif model_name.startswith("claude-"):
    if not anthropic:
        raise ImportError("anthropic package not installed.")
    if not ANTHROPIC_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not found – set it in .env or env.")
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    model_family = "anthropic"
else:
    raise ValueError(f"Unknown model_name: {model_name}")

# --- Policy simulation options ---
POLICY_DF = pd.read_csv("./policy/policy_simulation_Houston.csv").reset_index()
POLICY_LIST = []
for idx, row in POLICY_DF.iterrows():
    POLICY_LIST.append(
        f"{idx}. Tax: {row['tax_percentage']}%, Fare: ${row['fare']}/ride, Driving Fee: ${row['driver_fee']}/trip "
        f"▶ Car {row['drive_time_min']} min (${row['drive_cost']}) | Transit {row['bus_time_min']} min (${row['bus_cost']})"
    )

COMMUNITIES = [
    "Willowbrook",
    "Greater Greenspoint",
    "Carverdale",
    "Fairbanks / Northwest Crossing",
    "Greater Inwood",
    "Acres Home",
    "Hidden Valley",
    "Westbranch",
    "Addicks / Park Ten",
    "Spring Branch West",
    "Langwood",
    "Central Northwest (formerly Near Northwest)",
    "Independence Heights",
    "Lazybrook / Timbergrove",
    "Greater Heights",
    "Memorial",
    "Eldridge / West Oaks",
    "Briar Forest",
    "Westchase",
    "Mid‑West (formerly Woodlake/Briarmeadow)",
    "Greater Uptown",
    "Washington Avenue Coalition / Memorial Park",
    "Afton Oaks / River Oaks",
    "Neartown / Montrose",
    "Alief",
    "Sharpstown",
    "Gulfton",
    "West University Place",
    "Westwood",
    "Braeburn",
    "Meyerland",
    "Braeswood",
    "Medical Center",
    "Astrodome Area",
    "South Main",
    "Brays Oaks (formerly Greater Fondren S.W.)",
    "Westbury",
    "Willow Meadows / Willowbend",
    "Fondren Gardens",
    "Central Southwest",
    "Fort Bend / Houston",
    "IAH Airport",
    "Kingwood",
    "Lake Houston",
    "Northside / Northline",
    "Jensen",
    "East Little York / Homestead",
    "Trinity / Houston Gardens",
    "East Houston",
    "Settegast",
    "Northside Village",
    "Kashmere Gardens",
    "El Dorado / Oates Prairie",
    "Hunterwood",
    "Greater Fifth Ward",
    "Denver Harbor / Port Houston",
    "Pleasantville Area",
    "Northshore",
    "Clinton Park / Tri‑Community",
    "Fourth Ward",
    "Downtown",
    "Midtown",
    "Second Ward",
    "Greater Eastwood",
    "Harrisburg / Manchester",
    "Museum Park (formerly Binz)",
    "Greater Third Ward",
    "Greater OST / South Union",
    "Gulfgate Riverview / Pine Valley",
    "Pecan Park",
    "Sunnyside",
    "South Park",
    "Golfcrest / Bellfort / Reveille",
    "Park Place",
    "Meadowbrook / Allendale",
    "South Acres / Crestmont Park",
    "Minnetex",
    "Greater Hobby Area",
    "Edgebrook",
    "South Belt / Ellington",
    "Clear Lake",
    "Magnolia Park",
    "MacGregor",
    "Spring Branch North",
    "Spring Branch Central",
    "Spring Branch East",
    "Greenway / Upper Kirby",
    "Lawndale / Wayside"
]

SYSTEM_PROMPT = """
You are a representative from one of the eighty-eight super neighborhoods in Houston. On behalf your community, you will be participating in a referendum in which representatives from all neighborhoods vote on a set of transportation policy proposals. You are allowed to choose up to five proposals and submit your vote as a ranked list of proposals. Remember you must act in the best interests of your neighborhood.
Think step by step and submit the top five policy proposals in a descending order of preference. Return a JSON object:
{
  "community_area": "<name>",
  "thinking": {
    "disposable_income": "<one-sentence summary of the disposable income of the super neighborhood>",
    "discretionary_consumption": "<one-sentence summary of the discretionary consumption of the super neighborhood>",
    "accessibility": "<one-sentence summary of the accessibility to resources and services by different modes of transportation in the super neighborhood>",
    "decision_rationale": "<rationale of the ranked voting decision, showing factores that influence the tradeoffs and ranking rationale, think in step by step>"
  "vote": [<rank1>, <rank2>, <rank3>,<rank4>, <rank5>]
}
* The vote list must contain 5 distinct integers from 1 to 24.
* No additional keys, no Markdown, no code fences.
"""

USER_PROMPT_TEMPLATE = """
In the referendum, residents from all eighty-eight super neighborhoods will vote on 24 transportation policy proposals.  
A policy proposal consists of three policies: (i) transit fare policy, which may set a per-trip fare for riding transit to either $0.75, $1.25, or $1.75, 
(ii) tax policy, which may set a dedicated sales tax rate to either 0.5%, 1%, or 1.5%; 
and (iii) a driver-fee policy, which may set a per-trip fee for driving to either  $0.00, $0.50, or $1.00.   
Since there are three options for each policy, you have 24 proposals to vote on.  
You must pick the top five proposals and rank them from 1 to 5 according to how they serve your interest.  Your interest should be defined by weighing the cost and benefits of each package.  
The Houston city planning agency has estimated the travel times per trip by transit and driving corresponding to each policy, along with the corresponding fare and fees. 
So, you should use this information to gauge your interest.  You should also be aware that paying either fare for transit or a fee for driving will take a portion of your income away from other consumption (such as food, housing, cloth, vacation, tuition etc.).  
Hence, you should carefully assess how such a loss of income might affect your life.  In doing so, please bear in mind that you are acting as a representative resident in your community – so you should try to use as much data (census, employment, transit availability, car ownership, etc.) from your community to make your decision.

Here is some other information that might help your deliberations:
1.  The dedicated sales tax is meant to support transit services. The higher the tax, the more money the transit ANONYMOUS
2.  If you ride in transit, you will pay fare; if you drive, you will pay the driver fee.  All revenues from fare and driver fee will be used to support transit, which means higher fare and driver fee generally mean better transit services. However, higher fare and driver fees also mean you have less money to spend on other goods and services.  Also, remember the fare and fee are paid on the per trip basis.   
3.  Driver fees and better transit might persuade some drivers to switch from driving to transit. This could help reduce congestion, lowering driving time and reducing emissions.  

Current Transportation Policy and the travel time and cost:
the transit fare is $1.25/ride, the sales tax is 1%, and the driving fee is $0.00/trip. 
The travel time by car is 25 minutes, and the travel time by transit is 60.77 minutes, the cost per trip by car is $7.19, and the cost per trip by transit is $1.25.

Candidate Policy Options and the associated travel time and cost estimated by Houston city planning agency:
{policy_options}

Evaluate these 24 policy combinations for the {community} super neighborhood.

Consider the following factors based on the unique characteristics of the super neighborhood:
1. The disposable income of the super neighborhood
2. The discretionary consumption of the super neighborhood
3. The accessibility to resources and services by different modes of transportation in the super neighborhood
Then, consider the tradeoffs and ranking rationale of the ranked voting decision.

Return top 5 ranked policies (1-24) and reasoning as a JSON object.
""".replace("{policy_options}", "\n".join(POLICY_LIST))

def get_agent_response(community: str) -> dict | None:
    try:
        if model_family == "openai":
            chat = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                temperature=temp,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(community=community)}
                ],
                max_tokens=2048,
                timeout=120,
            )
            return json.loads(chat.choices[0].message.content)
        elif model_family == "anthropic":
            # Claude expects system prompt as system, user prompt as user
            msg = client.messages.create(
                model=model_name,
                max_tokens=2048,
                temperature=temp,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(community=community)}
                ],
            )
            # Claude 3.5 returns JSON as string in content
            return json.loads(msg.content[0].text if hasattr(msg.content[0], "text") else msg.content[0])
        else:
            raise ValueError("Unknown model family")
    except Exception as e:
        print(f"[ERROR] {community}: {e}")
        traceback.print_exc()
        return None
    
def sanitize_filename(name: str) -> str:
    """
    Replace slashes with '&', remove other non-alphanumeric characters except spaces and dashes,
    and convert to lowercase with underscores.
    """
    name = name.replace('/', '&')
    name = re.sub(r'[^\w\s&-]', '', name)  # remove all except word chars, spaces, dashes, and &
    return name.replace(' ', '_').lower()


def save_individual_response(community: str, data: dict, run_dir: str) -> None:
    safe_name = sanitize_filename(community)
    path = Path(run_dir) / "responses" / f"{safe_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def compile_responses(run_dir: str) -> str:
    rows = []
    for community in COMMUNITIES:
        safe_name = sanitize_filename(community)
        path = Path(run_dir) / "responses" / f"{safe_name}.json"
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

def summarize_winning_policies_across_rounds(base_dir: str, runs: int):
    """
    After all rounds, count the winning policies in each round and generate a summary.
    """
    winning_policies = []
    round_details = []
    for i in range(1, runs + 1):
        i = i+1
        run_dir = os.path.join(base_dir, f"round_{i}")
        irv_json_path = os.path.join(run_dir, "irv_summary.json")
        if not os.path.exists(irv_json_path):
            print(f"[WARN] Missing IRV summary for round {i}: {irv_json_path}")
            continue
        try:
            with open(irv_json_path, "r") as f:
                data = json.load(f)
                winning_policies.append(data["winning_policy"])
                round_details.append({
                    "round": i,
                    "winning_policy": data["winning_policy"],
                    "summary": data.get("summary", "")
                })
        except Exception as e:
            print(f"[ERROR] Reading {irv_json_path}: {e}")

    policy_counter = Counter(winning_policies)
    summary = {
        "total_rounds": runs,
        "winning_policy_counts": dict(policy_counter),
        "round_winners": round_details,
        "most_frequent_winner": policy_counter.most_common(1)[0][0] if policy_counter else None,
        "most_frequent_winner_count": policy_counter.most_common(1)[0][1] if policy_counter else 0
    }

    summary_path = os.path.join(base_dir, "winning_policy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n🏆 Winning policy summary saved to {summary_path}\n")
    return summary_path

def batch_main(batch_version: str, runs: int = 10):
    base_dir = f"version_batch_{batch_version}"
    os.makedirs(base_dir, exist_ok=True)

    for i in range(1, runs + 1):
        i = i+2
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

    # After all rounds, summarize the winning policies
    summarize_winning_policies_across_rounds(base_dir, runs)

    print("\n🎉 All simulations completed.")

if __name__ == "__main__":
    
    batch_main(batch_version=f"CA_HOU_{model_name.split('-')[0]}", runs=8) 