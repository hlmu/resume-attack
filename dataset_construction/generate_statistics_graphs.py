import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Configure matplotlib to match LaTeX document font sizes
# LaTeX normal font size is 10bp, but figures get scaled down, so we use larger sizes
plt.rcParams.update({
    'font.size': 16,           # Base font size (increased from 10 to account for scaling)
    'axes.titlesize': 24,      # Subplot titles (increased to 24 for better visibility)
    'axes.labelsize': 20,      # Axis labels (increased to 20 for better visibility)
    'xtick.labelsize': 18,     # X-axis tick labels (increased to 18)
    'ytick.labelsize': 18,     # Y-axis tick labels (increased to 18)
    'legend.fontsize': 16,     # Legend text (increased to 16)
    'figure.titlesize': 26,    # Main figure title (increased to 26)
    'font.family': 'serif',    # Match LaTeX serif fonts
    'font.serif': ['Times', 'DejaVu Serif', 'serif'],
})

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data from the statistics files
jd_data = {
    'Technology & IT': 337,
    'Finance & Business Services': 171,
    'Engineering & Construction': 98,
    'Retail & Consumer Goods': 81,
    'Healthcare & Life Sciences': 73,
    'Media & Communications': 47,
    'Education & Non-profit': 33,
    'Other Services': 30,
    'Staffing & HR': 29,
    'Manufacturing': 23,
    'Legal': 23,
    'Transportation & Logistics': 20,
    'Government & Public Sector': 19,
    'Real Estate': 16
}

resume_data = {
    'Technology & IT': 242,
    'Media & Communications': 125,
    'Finance & Business Services': 115,
    'Education & Non-profit': 111,
    'Engineering & Construction': 98,
    'Healthcare & Life Sciences': 94,
    'Other Services': 56,
    'Retail & Consumer Goods': 33,
    'Staffing & HR': 33,
    'Legal': 23,
    'Government & Public Sector': 20,
    'Real Estate': 19,
    'Transportation & Logistics': 19,
    'Manufacturing': 12
}

def create_distribution_plots():
    """Create distribution plots for job descriptions and resumes"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Job Descriptions distribution
    categories_jd = list(jd_data.keys())
    counts_jd = list(jd_data.values())
    percentages_jd = [(count/sum(counts_jd))*100 for count in counts_jd]
    
    # Create horizontal bar plot for job descriptions
    bars1 = ax1.barh(range(len(categories_jd)), counts_jd, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(categories_jd)))
    ax1.set_yticklabels(categories_jd)
    ax1.set_xlabel('Number of Job Descriptions', fontsize=20)
    ax1.set_title('Job Description Distribution by Category\n(Total: 1,000)', fontsize=24, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars1, percentages_jd)):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontsize=16)
    
    # Resume distribution
    categories_resume = list(resume_data.keys())
    counts_resume = list(resume_data.values())
    percentages_resume = [(count/sum(counts_resume))*100 for count in counts_resume]
    
    # Create horizontal bar plot for resumes
    bars2 = ax2.barh(range(len(categories_resume)), counts_resume, color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(categories_resume)))
    ax2.set_yticklabels(categories_resume)
    ax2.set_xlabel('Number of Resume Profiles', fontsize=20)
    ax2.set_title('Resume Profile Distribution by Category\n(Total: 1,000)', fontsize=24, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars2, percentages_resume)):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('jmlc25_paper/figures/dataset_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('jmlc25_paper/figures/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_plot():
    """Create a comparison plot showing the difference between job and resume distributions"""
    
    # Align categories for comparison
    all_categories = sorted(set(jd_data.keys()) | set(resume_data.keys()))
    
    jd_counts = [jd_data.get(cat, 0) for cat in all_categories]
    resume_counts = [resume_data.get(cat, 0) for cat in all_categories]
    
    # Convert to percentages
    jd_percentages = [(count/sum(jd_counts))*100 for count in jd_counts]
    resume_percentages = [(count/sum(resume_counts))*100 for count in resume_counts]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(all_categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, jd_percentages, width, label='Job Descriptions', 
                   color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, resume_percentages, width, label='Resume Profiles', 
                   color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Professional Categories', fontsize=20)
    ax.set_ylabel('Percentage (%)', fontsize=20)
    ax.set_title('Comparison of Job Description vs Resume Profile Distributions', 
                 fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=45, ha='right')
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Add value labels on bars with consistent rotation to avoid overlap
    max_height = max(max(jd_percentages), max(resume_percentages))
    label_offset = 0.3
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # Job Descriptions (bars1) - above bar, rotated 45 degrees consistently
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + label_offset,
               f'{height1:.1f}%', ha='center', va='bottom', fontsize=13,
               color='black', weight='normal', rotation=45)
        
        # Resume Profiles (bars2) - above bar, rotated 45 degrees consistently  
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + label_offset,
               f'{height2:.1f}%', ha='center', va='bottom', fontsize=13,
               color='black', weight='normal', rotation=45)
    
    # Adjust y-axis limit to accommodate rotated labels (add extra space for 45-degree rotation)
    ax.set_ylim(0, max_height + 4)
    
    plt.tight_layout()
    plt.savefig('jmlc25_paper/figures/dataset_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('jmlc25_paper/figures/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_pie_charts():
    """Create pie charts for both distributions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Job Descriptions pie chart
    categories_jd = list(jd_data.keys())
    counts_jd = list(jd_data.values())
    
    # Group smaller categories for better visualization
    threshold = 30  # Group categories with less than 30 samples
    major_categories_jd = []
    major_counts_jd = []
    other_count_jd = 0
    
    for cat, count in zip(categories_jd, counts_jd):
        if count >= threshold:
            major_categories_jd.append(cat)
            major_counts_jd.append(count)
        else:
            other_count_jd += count
    
    if other_count_jd > 0:
        major_categories_jd.append('Others')
        major_counts_jd.append(other_count_jd)
    
    wedges1, texts1, autotexts1 = ax1.pie(major_counts_jd, labels=major_categories_jd, 
                                          autopct='%1.1f%%', startangle=90)
    ax1.set_title('Job Description Distribution\n(Total: 1,000)', fontsize=24, fontweight='bold')
    
    # Set font sizes for pie chart labels
    for text in texts1:
        text.set_fontsize(18)
    for autotext in autotexts1:
        autotext.set_fontsize(16)
    
    # Resume pie chart
    categories_resume = list(resume_data.keys())
    counts_resume = list(resume_data.values())
    
    major_categories_resume = []
    major_counts_resume = []
    other_count_resume = 0
    
    for cat, count in zip(categories_resume, counts_resume):
        if count >= threshold:
            major_categories_resume.append(cat)
            major_counts_resume.append(count)
        else:
            other_count_resume += count
    
    if other_count_resume > 0:
        major_categories_resume.append('Others')
        major_counts_resume.append(other_count_resume)
    
    wedges2, texts2, autotexts2 = ax2.pie(major_counts_resume, labels=major_categories_resume, 
                                          autopct='%1.1f%%', startangle=90)
    ax2.set_title('Resume Profile Distribution\n(Total: 1,000)', fontsize=24, fontweight='bold')
    
    # Set font sizes for pie chart labels
    for text in texts2:
        text.set_fontsize(18)
    for autotext in autotexts2:
        autotext.set_fontsize(16)
    
    plt.tight_layout()
    plt.savefig('jmlc25_paper/figures/dataset_pie_charts.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('jmlc25_paper/figures/dataset_pie_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_mismatch_analysis():
    """Create a plot showing the mismatch between supply (resumes) and demand (jobs)"""
    
    # Calculate mismatch ratios
    all_categories = sorted(set(jd_data.keys()) | set(resume_data.keys()))
    
    mismatch_data = []
    for cat in all_categories:
        jd_count = jd_data.get(cat, 0)
        resume_count = resume_data.get(cat, 0)
        
        # Calculate supply-demand ratio (resume/job)
        if jd_count > 0:
            ratio = resume_count / jd_count
        else:
            ratio = float('inf') if resume_count > 0 else 0
        
        mismatch_data.append({
            'category': cat,
            'jobs': jd_count,
            'resumes': resume_count,
            'ratio': ratio,
            'shortage': jd_count - resume_count  # positive means job shortage, negative means oversupply
        })
    
    df = pd.DataFrame(mismatch_data)
    
    # Create the mismatch plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Supply-demand ratio plot
    colors = ['red' if r < 1 else 'green' if r > 1 else 'gray' for r in df['ratio']]
    bars1 = ax1.bar(range(len(df)), df['ratio'], color=colors, alpha=0.7)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['category'], rotation=45, ha='right')
    ax1.set_ylabel('Supply/Demand Ratio\n(Resume Count / Job Count)', fontsize=20)
    ax1.set_title('Supply-Demand Analysis by Professional Category', fontsize=24, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars1, df['ratio'])):
        if ratio != float('inf'):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=16)
    
    # Shortage/oversupply plot
    # colors2 = ['red' if s > 0 else 'blue' for s in df['shortage']]
    # bars2 = ax2.bar(range(len(df)), df['shortage'], color=colors2, alpha=0.7)
    bars2 = ax2.bar(range(len(df)), df['shortage'], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['category'], rotation=45, ha='right')
    ax2.set_ylabel('Job Surplus (+) / Resume Surplus (-)', fontsize=20)
    ax2.set_title('Market Imbalance by Category\n(Positive = More Jobs than Resumes, Negative = More Resumes than Jobs)', 
                  fontsize=24, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Add shortage labels
    for i, (bar, shortage) in enumerate(zip(bars2, df['shortage'])):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (5 if shortage >= 0 else -15),
                f'{int(shortage)}', ha='center', va='bottom' if shortage >= 0 else 'top', 
                fontsize=16)
    
    plt.tight_layout()
    plt.savefig('jmlc25_paper/figures/supply_demand_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('jmlc25_paper/figures/supply_demand_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def main():
    """Generate all statistics visualizations"""
    
    # Create figures directory if it doesn't exist
    Path('jmlc25_paper/figures').mkdir(parents=True, exist_ok=True)
    
    print("Generating dataset distribution plots...")
    create_distribution_plots()
    
    print("Generating comparison plot...")
    create_comparison_plot()
    
    print("Generating pie charts...")
    create_pie_charts()
    
    print("Generating supply-demand analysis...")
    mismatch_df = create_mismatch_analysis()
    
    # Print summary statistics
    print("\n=== DATASET SUMMARY STATISTICS ===")
    print(f"Total Job Descriptions: {sum(jd_data.values())}")
    print(f"Total Resume Profiles: {sum(resume_data.values())}")
    print(f"Number of Categories: {len(set(jd_data.keys()) | set(resume_data.keys()))}")
    
    print("\n=== MARKET IMBALANCE ANALYSIS ===")
    print("Categories with highest job demand (shortage):")
    top_shortage = mismatch_df.nlargest(3, 'shortage')[['category', 'shortage']]
    for _, row in top_shortage.iterrows():
        print(f"  {row['category']}: {int(row['shortage'])} more jobs than resumes")
    
    print("\nCategories with highest resume supply (oversupply):")
    top_oversupply = mismatch_df.nsmallest(3, 'shortage')[['category', 'shortage']]
    for _, row in top_oversupply.iterrows():
        print(f"  {row['category']}: {abs(int(row['shortage']))} more resumes than jobs")
    
    print("\nAll figures saved to jmlc25_paper/figures/")

if __name__ == "__main__":
    main() 