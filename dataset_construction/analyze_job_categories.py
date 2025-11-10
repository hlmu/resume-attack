import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_job_categories(csv_file_path):
    """
    Analyze the distribution of job categories in a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing job classifications
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get the distribution of categories
    category_distribution = df['category'].value_counts()
    
    # Calculate percentages
    category_percentages = df['category'].value_counts(normalize=True) * 100
    
    # Create summary text
    summary_text = "Job Category Distribution Analysis\n"
    summary_text += "==================================\n\n"
    summary_text += f"Total number of jobs: {len(df)}\n"
    summary_text += f"Number of unique categories: {df['category'].nunique()}\n\n"
    summary_text += "Category distribution:\n"
    for category, count in category_distribution.items():
        summary_text += f"{category:<30} {count}\n"
    summary_text += "\nCategory percentages:\n"
    for category, percentage in category_percentages.items():
        summary_text += f"{category}: {percentage:.2f}%\n"
    
    # Save summary to text file
    summary_path = "resume_category_analysis.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    
    # Print basic statistics
    print("Job Category Distribution Analysis")
    print("==================================")
    print(f"Total number of jobs: {len(df)}")
    print(f"Number of unique categories: {df['category'].nunique()}")
    print("\nCategory distribution:")
    print(category_distribution)
    print("\nCategory percentages:")
    for category, percentage in category_percentages.items():
        print(f"{category}: {percentage:.2f}%")
    print(f"\nAnalysis summary saved to {summary_path}")
    
    # Create a bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=category_distribution.values, y=category_distribution.index, hue=category_distribution.index, palette="viridis", legend=False)
    plt.title("Distribution of Job Categories")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Category")
    plt.tight_layout()
    
    # Save the plot
    plot_path = "resume_category_distribution.png"
    plt.savefig(plot_path)
    print(f"\nBar plot saved to {plot_path}")
    
    # Create a pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(category_distribution.values, labels=category_distribution.index.tolist(), autopct='%1.1f%%', startangle=90)
    plt.title("Job Category Distribution (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the pie chart
    pie_path = "resume_category_pie_chart.png"
    plt.savefig(pie_path)
    print(f"Pie chart saved to {pie_path}")
    
    return category_distribution, category_percentages

if __name__ == "__main__":
    csv_file_path = "results/resume_classifications.csv"
    distribution, percentages = analyze_job_categories(csv_file_path)
