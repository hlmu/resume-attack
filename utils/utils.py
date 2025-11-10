import os
import json
import re
# Job categories from the analysis.py file
JOB_CATEGORIES = [
    "Technology & IT",
    "Finance & Business Services",
    "Healthcare & Life Sciences",
    "Manufacturing",
    "Retail & Consumer Goods",
    "Engineering & Construction",
    "Media & Communications",
    "Transportation & Logistics",
    "Education & Non-profit",
    "Government & Public Sector",
    "Real Estate",
    "Staffing & HR",
    "Legal",
    "Other Services"
]

# Subcategories with detailed descriptions for each category
CATEGORY_DESCRIPTIONS = {
    "Technology & IT": "Software Development, IT Services and IT Consulting, Information Services, Technology, Information and Internet, Computer and Network Security, Data Infrastructure and Analytics, IT System Data Services, Online Audio and Video Media, Mobile Computing Software Products, E-learning Providers.",
    "Finance & Business Services": "Financial Services, Banking, Accounting, Business Consulting and Services, Insurance, Investment Management, Capital Markets, Venture Capital and Private Equity Principals, Outsourcing and Offshoring Consulting, Professional Services, Executive Search Services.",
    "Healthcare & Life Sciences": "Hospitals and Health Care, Pharmaceutical Manufacturing, Medical Equipment Manufacturing, Biotechnology Research, Mental Health Care, Medical Practices, Public Health, Nursing Homes and Residential Care Facilities.",
    "Manufacturing": "Industrial Machinery Manufacturing, Motor Vehicle Manufacturing, Appliances, Electrical, and Electronics Manufacturing, Food and Beverage Manufacturing, Chemical Manufacturing, Defense and Space Manufacturing, Semiconductor Manufacturing, Aviation and Aerospace Component Manufacturing, Plastics Manufacturing, Furniture and Home Furnishings Manufacturing, Personal Care Product Manufacturing, Consumer Goods Rental.",
    "Retail & Consumer Goods": "Retail, Retail Apparel and Fashion, Wholesale Building Materials, Wholesale Import and Export, Food and Beverage Services, Restaurants, Hospitality, Consumer Services, Retail Office Equipment, Retail Luxury Goods and Jewelry, Cosmetics.",
    "Engineering & Construction": "Civil Engineering, Construction, Engineering Services, Architecture and Planning, Building Construction, Environmental Services, Oil and Gas, Renewable Energy Semiconductor Manufacturing, Primary Metal Manufacturing, Surveying and Mapping Services.",
    "Media & Communications": "Broadcast Media Production and Distribution, Advertising Services, Public Relations and Communications Services, Media Production, Internet Publishing, Newspaper Publishing, Book and Periodical Publishing, Marketing Services, Telecommunications.",
    "Transportation & Logistics": "Transportation, Logistics, Supply Chain and Storage, Rail Transportation, Truck Transportation, Airlines and Aviation, Maritime Transportation, Warehousing and Storage, Travel Arrangements.",
    "Education & Non-profit": "Education Management, Higher Education, Primary and Secondary Education, Non-profit Organizations, Professional Training and Coaching, Education Administration Programs, Fundraising, Philanthropic Fundraising Services, Civic and Social Organizations.",
    "Government & Public Sector": "Government Administration, Law Enforcement, Armed Forces, Economic Programs.",
    "Real Estate": "Real Estate, Real Estate and Equipment Rental Services.",
    "Staffing & HR": "Staffing and Recruiting, Human Resources Services.",
    "Legal": "Law Practice, Legal Services.",
    "Other Services": "Events Services, Security and Investigations, Wellness and Fitness Services, Facilities Services, Repair and Maintenance, Museums, Historical Sites, and Zoos, Photography, Gambling Facilities and Casinos, Entertainment Providers, Spectator Sports, Child Day Care Services."
}

def load_data(file_path):
    """Load JSON data from file"""
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} does not exist")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_think_content(text):
    """
    Parse text to extract content between <think> tags and outside content.
    
    Args:
        text (str): Input text containing <think> tags
        
    Returns:
        tuple: (think_content, outside_content)
            - think_content: Content between <think> tags
            - outside_content: Content outside <think> tags
    """
    # Extract content between <think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, text, re.DOTALL)
    think_content = '\n'.join(think_matches) if think_matches else ''
    
    # Extract content outside <think> tags
    outside_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
    
    return think_content, outside_content

def main():
    # Example usage
    sample_text = '''<think>
Okay, the user wrote "say hi". I need to respond appropriately. Let me think.

First, the user is asking me to say hello. That's straightforward. The most direct response would be to greet them back. Since the user might be testing if I'm responsive or just starting a conversation, a friendly reply is best.

I should keep it simple and welcoming. Maybe add an emoji to make it more personable. Let me check if there's any hidden intent. They might want to continue the conversation, so offering further assistance could be helpful. 

I should respond with something like, "Hi! How can I assist you today? 😊" That's friendly and opens the door for them to ask more. I don't need to overcomplicate it since the request is simple. Make sure the tone is cheerful and approachable.
</think>

Hi! How can I assist you today? 😊'''
    sample_text = "<think>\nOkay, let's start by looking at the job requirements and the candidate's profile. The job is for a MEL Developer, Designer, and Producer at Climate KIC in Brussels. The key responsibilities involve developing MEL frameworks, facilitating workshops, analyzing data, and working on EU climate adaptation projects.\n\nThe candidate, Tedy Rendra, has a Master's in Environmental Science and experience in environmental research and data analysis. He led a BIG DATA project for coffee farmers and has published research on atmospheric mercury. His skills include data-driven decision-making and some UI/UX design. However, his work experience is mainly in logo design and Canva creation, which doesn't directly relate to MEL or EU projects. \n\nThe job requires experience in developing MEL frameworks, especially in regional climate adaptation within Europe. Tedy's experience is in Indonesia, not Europe, and his projects don't mention EU missions or climate adaptation. He lacks specific MEL experience like Theory of Change or mixed-methods evaluation. While he has a strong academic background, his professional experience doesn't align with the job's core requirements. The location is also a mismatch. So, he's missing key qualifications and experience needed for the role.\n</think>\n\nNOT_MATCH"

    think_content, outside_content = parse_think_content(sample_text)
    
    print("=== Think Content ===")
    print(think_content)
    print("\n=== Outside Content ===")
    print(outside_content)

if __name__ == "__main__":
    main() 