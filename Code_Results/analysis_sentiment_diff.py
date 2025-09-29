import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Step 0: Download VADER lexicon (first-time only)
# nltk.download('vader_lexicon')



#%%

############  calculate SCORE 
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Step 0: Download VADER lexicon (first-time only)
# nltk.download('vader_lexicon')

# Step 1: Load all community JSON responses from all rounds
def load_thinking_texts_across_rounds(base_folder, num_rounds=10):
    data = []
    for round_num in range(1, num_rounds + 1):
        round_folder = os.path.join(base_folder, f"round_{round_num}", "responses")
        if not os.path.exists(round_folder):
            continue
        for filename in os.listdir(round_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(round_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    community = record.get("community_area", filename.replace(".json", ""))
                    thinking = record.get("thinking", {})
                    text = " ".join([v for v in thinking.values() if isinstance(v, str)])
                    data.append({
                        "community_area": community,
                        "thinking_text": text.strip(),
                        "round": round_num
                    })
    return pd.DataFrame(data)

# Step 2: Apply VADER Sentiment Analysis
def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['vader_score'] = df['thinking_text'].apply(lambda text: sia.polarity_scores(text)['compound'])
    df['vader_sentiment'] = df['vader_score'].apply(lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral'))
    return df  # Do not sort here; sorting will be handled in plotting

# Step 3: Visualization of average sentiment per community_area
def plot_sentiment_avg(df_avg, figure_path):
    # Sort alphabetically by community_area
    df_sorted = df_avg.sort_values(by='community_area').reset_index(drop=True)
    
    # Assign colors based on sentiment
    colors = df_sorted['vader_sentiment'].map({'positive': 'skyblue', 'neutral': 'gray', 'negative': 'salmon'})

    # Plot
    plt.figure(figsize=(14, 7))
    bars = plt.bar(
        df_sorted['community_area'],
        df_sorted['vader_score'],
        color=colors
    )
    plt.ylabel("Average Compound Sentiment Score", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=90, fontsize=12)
    plt.ylim(-1, 1)  # Set y-axis scale from -1 to 1
    plt.tight_layout()
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    plt.show()

# === Run Script ===
if __name__ == "__main__":
    batch_name = 'CA_CHI_35_claud'
    base_folder = f"./version_batch_{batch_name}"
    figure_path = f"./results_vote/fig_sentiment/sentiment_{batch_name}_avg10.png"
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # Load all rounds' data
    df = load_thinking_texts_across_rounds(base_folder, num_rounds=10)
    df_result = analyze_sentiment(df)
    df_result.to_csv(f"./results_vote/sentiment_{batch_name}_allrounds.csv", index=False)

    # Compute average sentiment per community_area
    df_avg = df_result.groupby('community_area', as_index=False).agg({'vader_score': 'mean'})
    # Assign sentiment label based on average score
    df_avg['vader_sentiment'] = df_avg['vader_score'].apply(lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral'))

    df_avg.to_csv(f"./results_vote/sentiment_{batch_name}_avg10.csv", index=False)

    print(df_avg[['community_area', 'vader_score', 'vader_sentiment']].head(10))
    plot_sentiment_avg(df_avg, figure_path)



#%%
#########################  calculate SCORE DIFFERENCE 

# Step 1: Load all community JSON responses from all rounds
def load_thinking_texts_across_rounds(base_folder, num_rounds=10):
    data = []
    for round_num in range(1, num_rounds + 1):
        round_folder = os.path.join(base_folder, f"round_{round_num}", "responses")
        if not os.path.exists(round_folder):
            continue
        for filename in os.listdir(round_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(round_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    community = record.get("community_area", filename.replace(".json", ""))
                    thinking = record.get("thinking", {})
                    text = " ".join([v for v in thinking.values() if isinstance(v, str)])
                    data.append({
                        "community_area": community,
                        "thinking_text": text.strip(),
                        "round": round_num
                    })
    return pd.DataFrame(data)

# Step 2: Apply VADER Sentiment Analysis
def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['vader_score'] = df['thinking_text'].apply(lambda text: sia.polarity_scores(text)['compound'])
    df['vader_sentiment'] = df['vader_score'].apply(lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral'))
    return df  # Do not sort here; sorting will be handled in plotting

# Step 3: Visualization of sentiment difference between two batches
def plot_sentiment_difference(df_avg1, df_avg2, batch1, batch2, figure_path):
    # Merge on community_area
    df_merged = pd.merge(
        df_avg1[['community_area', 'vader_score']],
        df_avg2[['community_area', 'vader_score']],
        on='community_area',
        suffixes=(f'_{batch1}', f'_{batch2}')
    )
    df_merged['score_diff'] = df_merged[f'vader_score_{batch2}'] - df_merged[f'vader_score_{batch1}']

    # Sort alphabetically by community_area
    df_merged = df_merged.sort_values(by='community_area').reset_index(drop=True)

    # Color: positive diff = blue, negative diff = red, zero = gray
    def diff_color(x):
        if x > 0.05:
            return 'skyblue'
        elif x < -0.05:
            return 'salmon'
        else:
            return 'gray'
    colors = df_merged['score_diff'].apply(diff_color)

    plt.figure(figsize=(14, 7))
    bars = plt.bar(
        df_merged['community_area'],
        df_merged['score_diff'],
        color=colors
    )
    plt.ylabel(f"Sentiment Score Difference", fontsize=14)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=90, fontsize=13)
    plt.ylim(-1.2, 1.2)  # Set y-axis scale from -1 to 1
    # plt.title(f"Difference in Average Sentiment per Community Area", fontsize=14)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    plt.show()

# === Run Script ===
if __name__ == "__main__":
    # Compare these two batches
    batch1 = 'CA_CHI_4o'
    batch2 = 'CA_CHI_4o_info'
    # batch1 = 'CA_CHI_35_claud'
    # batch2 = 'CA_CHI_35_claud_info'
    base_folder1 = f"./version_batch_{batch1}"
    base_folder2 = f"./version_batch_{batch2}"
    figure_path = f"./results_vote/fig_sentiment/sentiment_diff_{batch2}_vs_{batch1}_avg10.png"
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # Load and process batch1
    df1 = load_thinking_texts_across_rounds(base_folder1, num_rounds=10)
    df_result1 = analyze_sentiment(df1)
    df_result1.to_csv(f"./results_vote/sentiment_{batch1}_allrounds.csv", index=False)
    df_avg1 = df_result1.groupby('community_area', as_index=False).agg({'vader_score': 'mean'})
    df_avg1['vader_sentiment'] = df_avg1['vader_score'].apply(lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral'))
    df_avg1.to_csv(f"./results_vote/sentiment_{batch1}_avg10.csv", index=False)

    # Load and process batch2
    df2 = load_thinking_texts_across_rounds(base_folder2, num_rounds=10)
    df_result2 = analyze_sentiment(df2)
    df_result2.to_csv(f"./results_vote/sentiment_{batch2}_allrounds.csv", index=False)
    df_avg2 = df_result2.groupby('community_area', as_index=False).agg({'vader_score': 'mean'})
    df_avg2['vader_sentiment'] = df_avg2['vader_score'].apply(lambda x: 'positive' if x >= 0.1 else ('negative' if x <= -0.1 else 'neutral'))
    df_avg2.to_csv(f"./results_vote/sentiment_{batch2}_avg10.csv", index=False)

    # Print head of both for inspection
    print("Batch 1 ({}):".format(batch1))
    print(df_avg1[['community_area', 'vader_score', 'vader_sentiment']].head(10))
    print("Batch 2 ({}):".format(batch2))
    print(df_avg2[['community_area', 'vader_score', 'vader_sentiment']].head(10))

    # Plot difference
    plot_sentiment_difference(df_avg1, df_avg2, batch1, batch2, figure_path)
