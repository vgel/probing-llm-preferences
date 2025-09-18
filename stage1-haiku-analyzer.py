import json
import os
from collections import Counter

from anthropic import Anthropic
import dotenv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

dotenv.load_dotenv()

IN_DIR = "results/stage0-topics"
OUT_DIR = "results/stage1-analysis"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for file in os.listdir(IN_DIR):
        if not file.endswith(".json"):
            continue
        analyze(os.path.join(IN_DIR, file), OUT_DIR)


def analyze(file_path: str, output_dir: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = [item["response"] for item in data if "response" in item]
    print(f"Found {len(responses)} responses. Extracting keywords...")

    full_analysis = []
    all_keywords: list[str] = []

    # Extract keywords from each response
    for i, response in enumerate(responses):
        if i % 10 == 0:
            print(f"Extracting keywords {i + 1}/{len(responses)}")

        # Get 5 salient keywords for this response
        keywords = extract_keywords(response)

        # Store individual analysis
        full_analysis.append(
            {
                "response_id": i + 1,
                "keywords": ", ".join(keywords),
                "response_preview": response[:100] + "..."
                if len(response) > 100
                else response,
            }
        )

        # Add to all keywords
        all_keywords.extend(keywords)

    print("Identifying top topics from all keywords...")
    top_topics = identify_top_topics(all_keywords)

    file_name = os.path.basename(file_path).replace(".json", "")
    display_and_save_results(
        top_topics=top_topics,
        all_keywords=all_keywords,
        total_responses=len(responses),
        full_analysis=full_analysis,
        file_name=file_name,
        output_dir=output_dir,
    )


def extract_keywords(response: str):
    """Extract 5 most salient keywords from a response."""
    prompt = f"""Analyze this text and identify the 5 most salient keywords that define the content:

"{response[:500]}..."

Give me exactly 5 keywords that best capture the essence and main concepts of this text.

Format: keyword1, keyword2, keyword3, keyword4, keyword5"""

    keywords_text = call_haiku(prompt, max_tokens=50).strip()

    # Parse keywords
    keywords = [k.strip() for k in keywords_text.split(",")]
    # Ensure we have exactly 5, pad with empty if needed
    while len(keywords) < 5:
        keywords.append("")

    return keywords[:5]


def identify_top_topics(all_keywords: list[str]) -> list[str]:
    """Use Haiku to identify 10 most relevant topics from all keywords."""

    # Get most frequent keywords (top 50) to feed to Haiku
    keyword_counts = Counter(all_keywords)
    top_keywords = [word for word, _ in keyword_counts.most_common(50) if word.strip()]

    keywords_text = ", ".join(top_keywords)

    prompt = f"""Based on these keywords extracted from text responses, identify the 10 most relevant and distinct topics they represent, ordered from MOST FREQUENT to LEAST FREQUENT:

Keywords: {keywords_text}

Analyze these keywords and group them into 10 distinct, meaningful topics. Consider how often keywords related to each topic appear when determining the order.

IMPORTANT: Order your topics from most frequent/common to least frequent/common based on the keyword patterns you see.

Format your response as:
1. Most Frequent Topic Name
2. Second Most Frequent Topic Name
3. Third Most Frequent Topic Name
(etc. up to 10, ordered by frequency)

Make each topic name 2-4 words and distinct from the others."""

    response_text = call_haiku(prompt, max_tokens=300).strip()

    topics = []
    for line in response_text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            topic = line.split(".", 1)[-1].strip()
            if topic:
                topics.append(topic)

    return topics[:10]


def display_and_save_results(
    top_topics: list[str],
    all_keywords: list[str],
    total_responses: int,
    full_analysis: list,
    file_name: str,
    output_dir: str,
):
    """Display results with plots and save everything."""

    # Get top 10 keywords by frequency
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(10)

    pdf_path = os.path.join(output_dir, f"{file_name}_analysis.pdf")

    with PdfPages(pdf_path) as pdf:
        # Page 1: Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Topics plot
        if top_topics:
            y_pos = range(len(top_topics))
            # For topics, we'll show them as ranked (no frequency count since they're thematically derived)
            topic_scores = list(
                range(len(top_topics), 0, -1)
            )  # Reverse ranking as scores
            ax1.barh(y_pos, topic_scores, color="#0d9488")
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(
                [t[:30] + "..." if len(t) > 30 else t for t in top_topics]
            )
            ax1.set_xlabel("Relevance Rank")
            ax1.set_title(f"Top 10 Topics - {file_name}")
            ax1.invert_yaxis()

        # Keywords plot
        if top_keywords:
            keywords, k_counts = zip(*top_keywords)
            y_pos = range(len(keywords))
            ax2.barh(y_pos, k_counts, color="#14b8a6")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(keywords)
            ax2.set_xlabel("Frequency")
            ax2.set_title(f"Top 10 Keywords - {file_name}")
            ax2.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.show()

        # Page 2: Summary Table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")

        # Create summary table
        summary_data = []
        summary_data.append(["ANALYSIS SUMMARY", ""])
        summary_data.append(["File", file_name])
        summary_data.append(["Total Responses", str(total_responses)])
        summary_data.append(["Total Keywords Extracted", str(len(all_keywords))])
        summary_data.append(["", ""])
        summary_data.append(["TOP 10 TOPICS (by relevance)", "RANK"])

        for i, topic in enumerate(top_topics, 1):
            summary_data.append([f"{i}. {topic}", str(i)])

        summary_data.append(["", ""])
        summary_data.append(["TOP 10 KEYWORDS (by frequency)", "COUNT"])

        for i, (keyword, count) in enumerate(top_keywords, 1):
            summary_data.append([f"{i}. {keyword}", str(count)])

        # Create table
        table = ax.table(cellText=summary_data, cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style table
        for i in range(len(summary_data)):
            if summary_data[i][0] in [
                "ANALYSIS SUMMARY",
                "TOP 10 TOPICS (by relevance)",
                "TOP 10 KEYWORDS (by frequency)",
            ]:
                table[(i, 0)].set_facecolor("#0d9488")
                table[(i, 0)].set_text_props(weight="bold", color="white")
                table[(i, 1)].set_facecolor("#0d9488")
                table[(i, 1)].set_text_props(weight="bold", color="white")

        plt.title(
            f"Analysis Summary - {file_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    # Save full analysis to CSV
    csv_path = os.path.join(output_dir, f"{file_name}_full_analysis.csv")
    df = pd.DataFrame(full_analysis)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save all keywords to separate CSV
    keywords_csv_path = os.path.join(output_dir, f"{file_name}_all_keywords.csv")
    keywords_df = pd.DataFrame(all_keywords, columns=["keyword"])
    keywords_df.to_csv(keywords_csv_path, index=False, encoding="utf-8")

    # Save summary to text file
    txt_path = os.path.join(output_dir, f"{file_name}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TOPIC ANALYSIS RESULTS - {file_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total responses analyzed: {total_responses}\n")
        f.write(f"Total keywords extracted: {len(all_keywords)}\n\n")

        f.write("TOP 10 TOPICS (by relevance):\n")
        f.write("-" * 30 + "\n")
        for i, topic in enumerate(top_topics, 1):
            f.write(f"{i}. {topic}\n")

        f.write("\nTOP 10 KEYWORDS (by frequency):\n")
        f.write("-" * 30 + "\n")
        for i, (keyword, count) in enumerate(top_keywords, 1):
            f.write(f"{i}. {keyword}: {count}\n")

    # Display results in interface
    print("TOP 10 TOPICS (identified by Haiku from keywords):")
    print("=" * 60)
    for i, topic in enumerate(top_topics, 1):
        print(f"{i}. {topic}")

    print("\nTOP 10 KEYWORDS (by frequency):")
    print("=" * 50)
    for i, (keyword, count) in enumerate(top_keywords, 1):
        print(f"{i}. {keyword}: {count}")

    print(f"\nTotal responses analyzed: {total_responses}")
    print(f"Total keywords extracted: {len(all_keywords)}")
    print("\n📁 FILES SAVED:")
    print(f"📊 PDF Report: {pdf_path}")
    print(f"📋 Full Analysis CSV: {csv_path}")
    print(f"🔑 All Keywords CSV: {keywords_csv_path}")
    print(f"📄 Summary TXT: {txt_path}")
    print("✅ Done!")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def call_haiku(prompt: str, max_tokens: int = 4000, temperature: float = 0.0) -> str:
    client = Anthropic()
    result = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return result.content[0].text  # type: ignore


if __name__ == "__main__":
    main()
