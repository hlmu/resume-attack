import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

# Read the job-position averaged ASR data
with open('results/latex_tables/latex_table_overall_position_averaged_asr.tex', 'r') as f:
    content = f.read()

# Extract the tabular data
table_pattern = r'\\begin{tabular}{.*?}(.*?)\\end{tabular}'
table_match = re.search(table_pattern, content, re.DOTALL)

if table_match:
    table_data = table_match.group(1)
    
    # Split into lines and remove LaTeX commands
    lines = table_data.strip().split('\n')
    lines = [line for line in lines if not line.startswith('\\') and line.strip() != '']
    
    # Skip header lines
    data_lines = []
    in_data = False
    for line in lines:
        if in_data:
            data_lines.append(line)
        elif line.startswith('Qwen') or line.startswith('anthropic') or line.startswith('deepseek') or line.startswith('google') or line.startswith('gpt') or line.startswith('meta'):
            in_data = True
            data_lines.append(line)
    
    # Parse the data
    models = []
    instruction_asr = []
    invisible_experience_asr = []
    invisible_keywords_asr = []
    job_manipulation_asr = []
    
    for line in data_lines:
        if line.strip() == '':
            continue
        parts = line.split(' & ')
        if len(parts) >= 5:
            model = parts[0].strip()
            # Apply comprehensive renaming rules
            if model == 'Qwen-Qwen3-8B':
                model = 'Qwen3 8B Think'
            elif model == 'Qwen-Qwen3-8B-nonthink':
                model = 'Qwen3 8B Nonthink'
            elif model == 'anthropic-claude-3.5-haiku':
                model = 'Claude 3.5 Haiku'
            elif model == 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B':
                model = 'DeepSeek R1-Distill-Llama-8B'
            elif model == 'google-Gemini-2.5-Flash':
                model = 'Gemini 2.5 Flash'
            elif model == 'meta-llama-Llama-3.1-8B-Instruct':
                model = 'Llama 3.1 8B Instruct'
            elif model == 'openai-gpt-4o-mini':
                model = 'GPT-4o Mini'
            elif model == 'openai-gpt-5-mini-high':
                model = 'GPT-5 Mini High'
            elif model == 'openai-gpt-5-mini-minimal':
                model = 'GPT 5 Mini Minimal'
            elif model == 'openai-gpt-5-minimal':
                model = 'GPT 5 Minimal'
            elif model == 'openai-gpt-oss-120b-high':
                model = 'GPT OSS 120B High'
            elif model == 'openai-gpt-oss-120b-low':
                model = 'GPT OSS 120B Low'
            # Skip models that should be removed
            elif model in ['Qwen-Qwen3-8B-low', 'openai-gpt-oss-120b', 'gpt-oss-120b-openrouter-low']:
                continue
            models.append(model)
            
            # Extract first value (ASR) from each cell
            instruction_asr.append(float(parts[1].split('/')[0]))
            invisible_experience_asr.append(float(parts[2].split('/')[0]))
            invisible_keywords_asr.append(float(parts[3].split('/')[0]))
            job_manipulation_asr.append(float(parts[4].split('/')[0]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Instruction': instruction_asr,
        'Invisible Experience': invisible_experience_asr,
        'Invisible Keywords': invisible_keywords_asr,
        'Job Manipulation': job_manipulation_asr
    })

    attack_columns = ['Instruction', 'Invisible Experience', 'Invisible Keywords', 'Job Manipulation']
    df['Average ASR'] = df[attack_columns].mean(axis=1)
    df = df.sort_values('Average ASR', ascending=True).reset_index(drop=True)

    # Create a bar plot with models sorted by overall attack success
    df_melted = df[['Model'] + attack_columns].melt(id_vars=['Model'], var_name='Attack Method', value_name='ASR')

    # Plot
    plt.figure(figsize=(15, 10))
    sns.barplot(data=df_melted, x='Model', y='ASR', hue='Attack Method', order=df['Model'].tolist())
    plt.title('Attack Success Rate by Model and Attack Method (Averaged Across Positions)')
    plt.ylabel('Attack Success Rate (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Attack Method')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('jmlc25_paper/figures/attack_methods_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find most vulnerable models by attack method
    print("Most vulnerable models by attack method:")
    for attack_type in attack_columns:
        max_asr = df[attack_type].max()
        vulnerable_model = df[df[attack_type] == max_asr]['Model'].values[0]
        print(f"{attack_type}: {vulnerable_model} with ASR {max_asr}%")
    
    # Find overall most vulnerable model
    most_vulnerable = df.loc[df['Average ASR'].idxmax()]
    print(f"\nMost vulnerable model overall: {most_vulnerable['Model']} with average ASR {most_vulnerable['Average ASR']:.1f}%")
    
    # Print the DataFrame for verification
    print("\nData:")
    print(df)
else:
    print("Could not find tabular data in the file")
