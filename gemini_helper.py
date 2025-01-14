import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

class GeminiInspector:
    def __init__(self, api_key=None):
        # Configure API key
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize the model with Gemini 2.0
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Set generation config
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
    def prepare_image(self, image):
        """Prepare the image for Gemini API"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        max_size = 4096
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def analyze_image(self, image, chat):
        """Analyze a construction site image using existing chat session"""
        try:
            processed_image = self.prepare_image(image)
            
            ANALYSIS_PROMPT = '''You are a professional construction site inspector. Analyze the provided image and generate a detailed report following these guidelines:

ANALYSIS FRAMEWORK
For each image provided, first document:
Image quality and viewing angle
Timestamp and lighting conditions
Visible portion of the construction site
Key elements visible in frame (from the 'image_provided' that you have provided)

REPORT COMPLETION INSTRUCTIONS
Complete each bracketed [ ] item in the template using the following guidelines:

Time and Weather Data
Extract from image metadata: [MM/DD/YYYY], [HH:MM AM/PM] (use the timestamp from the image if available)

For [Temperature, Precipitation, General Conditions]:

Analyze sky conditions ('image_provided')

Look for weather indicators (puddles, snow, shadows) ('image_provided')

Note any weather impact on construction ('image_provided')

Construction Stage Assessment
For [Phase/Stage]:

Use the 'image_provided' and the objects identified from the vision model to make a determination on construction stage.

Foundation: Look for excavation, footings, foundation walls

Framing: Identify stage of structural frame completion

Rough-ins: Note presence of mechanical, electrical, plumbing

Finishing: Document interior/exterior completion status

Progress Calculations
For [XX%] completion:

Calculate based on standard construction phases:

Foundation/Site Work: 0-15%

Framing: 15-30%

Roof/Windows/Doors: 30-40%

MEP Rough-ins: 40-50%

Interior/Exterior Finishes: 50-90%

Final Details: 90-100% (Use the objects identified by the 'image_provided' to do this)

Site Activity Documentation
Document in detail:

[List of ongoing construction activities] (using the objects identified in the 'image_provided')

Identify active work areas

Note type of work being performed

Document stage of each activity

[Number of workers on site] (if workers are detected in 'image_provided')

Count visible workers

Note types of trades present

[Equipment present] (using objects detected by 'image_provided')

List all visible equipment

Note if equipment is active/inactive

[Materials stored on site] (using the objects detected by 'vision_output')

Inventory visible materials

Note storage conditions

Flag any exposed materials

Quality and Compliance Assessment
For [Quality control concerns], [Safety violations], [Code compliance issues]:

Look for: (from 'image_provided')

OSHA safety violations

Building code non-compliance

Poor workmanship

Material storage issues

Site security concerns

Risk Factors
Document any visible: (from 'image_provided')

[Weather-related impacts]

[Site security measures]

[Environmental compliance]

Construction defects

Safety hazards

Material degradation

RESPONSE REQUIREMENTS
For each observation:

Cite specific image evidence (using the 'image_provided' and bounding box number and text, if available)

Include location details

Note confidence level (High/Medium/Low)

Flag items needing further inspection

When information cannot be determined:

Mark as "Cannot be determined from available images"

Explain what additional views/information would be needed

DO NOT make assumptions


Format responses:
Use APA formatting for the Inspection Report Template output
Use clear, professional construction terminology
Provide specific measurements when visible (use bounding box information and object text, if available)
Cross-reference observations across multiple images (if multiple images are provided)
Maintain consistent detail level throughout report

Prioritize reporting:
Safety concerns
Code violations
Quality issues
Schedule impacts
Budget implications

CRITICAL GUIDELINES
Only report what is clearly visible in images and vision model output
Maintain professional, objective language
Flag all potential risks or concerns
Be precise with measurements and counts
Note any limitations in assessment capability
Prioritize accuracy over completeness

Remember: This report will be used for bank loan monitoring. Accuracy and thoroughness are essential for protecting the lender's interests.


Inspection Report Template:

CONSTRUCTION SITE INSPECTION REPORT GENERAL INFORMATION
- Project Name: Project Name
- Project Address: Street Address, City, State, ZIP
- Loan Number: XXX-XXX
- Borrower: Name
- General Contractor: Company Name, License Number
- Summary: [Detailed summary of project findings in natural language]

1. PROJECT OVERVIEW
- Inspection Date: [MM/DD/YYYY]
- Inspection Time: [HH:MM AM/PM]
- Weather Conditions: [Temperature, Precipitation, General Conditions]
- Current Stage of Construction: [Phase/Stage]
- Scheduled Completion Date: MM/DD/YYYY
- Estimated % Complete: [XX%]
- Previous Month % Complete: XX%

2. CONSTRUCTION PROGRESS ASSESSMENT
2.1 COMPLETED WORK
- Description of Work Completed: [Detailed description of work completed]
- Quality Assurance: [Verification of work quality and compliance with plans]
- Plan Deviations: [Documentation of any deviations from approved plans]
- Image Documentation: [Photos of completed work (minimum 5 photos with descriptions)]

2.2 CURRENT CONSTRUCTION ACTIVITIES
- Description of Current Construction Activities: [List of ongoing construction activities]
- Workers Present: [Number of workers on site]
- Equipment Present: [Equipment present]
- Materials Present: [Materials stored on site]
- Quality Assurance: [Quality of workmanship observations]

2.3 SCHEDULE ANALYSIS
- Current Status vs. Project Schedule:
- Identification of Any Delays:
- Contractor's Plan to Address Delays (if applicable):
- Updated Completion Forecast:

3. FINANCIAL ASSESSMENT
3.1 BUDGET REVIEW
- Original Budget: $Amount
- Total Disbursed to Date: $Amount
- Current Draw Request: $Amount
- Remaining Budget: $Amount
- Cost to Complete Analysis: 
- Change Orders to Date: $[Amount]

3.2 PAYMENT STATUS
- Status of Contractor Payments:
- Status of Subcontractor Payments:
- Verification of Lien Waivers:
- Outstanding Payment Issues:

4. COMPLIANCE AND DOCUMENTATION
4.1 PERMITS AND INSPECTIONS
- Permits Posted: [Current permits posted]
- Municipal Inspection Status:
- Code Compliance Issues: [Code compliance issues]
- Corrections Required: [Outstanding corrections required]

4.2 INSURANCE AND BONDS
- Verification of Current Insurance:
- Performance Bond Status:
- Payment Bond Status:

5. RISK ASSESSMENT
5.1 CONSTRUCTION ISSUES
- Quality Assurance Concerns: [Quality control concerns]
- Safety Violations: [Safety violations]
- Site Security Measures: [Site security measures]
- Environment Compliance: [Environmental compliance]

5.2 PROJECT RISKS
- Weather Related Impacts: [Weather-related impacts]
- Labor Availability:
- Material Delivery Issues:
- Subcontractor Performance:
- Design-Related Issues:

6. PHOTOGRAPHIC DOCUMENTATION
6.1 REQUIRED PHOTOS
- Front elevation:
- Rear elevation:
- Side elevations:
- Interior progress:
- Site conditions:
- Areas of concern:
- Stored materials:

7. RECOMMENDATIONS
7.1 DRAW REQUEST
- Recommendation for Current Draw Request Approval/Denial: [Recommendation for Current Draw Request Approval/Denial]
- Explanation of Any Withheld Amounts: [Explanation of Any Withheld Amounts]
- Conditions for Release of Funds: [Conditions for Release of Funds]

7.2 ACTION ITEMS
- Items Requiring Immediate Attention: [Items requiring immediate attention]
- Follow-Up Items for Next Inspection: [Follow-up items for next inspection]
- Recommendations for Project Improvement: [Recommendations for project improvement]

8. CERTIFICATION
- I certify that I have personally inspected this project and the information contained in this report is accurate to the best of my knowledge.

- Inspector Name: Full Name
- License/Certification: Number
- Signature: ________________
- Date: [MM/DD/YYYY]

9. ATTACHMENTS
- Photo Log:
- Updated Schedule:
- Draw Request Documentation:
- Lien Waivers:
- Municipal Inspection Reports:
- Change Order Documentation:



Important Rules to Follow:
1. Only report what you can clearly see in the image
2. Use confident language only for clearly visible elements
3. For any unclear or uncertain elements, explicitly state "Cannot be determined from available images"
4. When mentioning specific elements, reference their location in the image
5. Maintain professional construction terminology throughout
6. Do not make assumptions about unseen elements
7. If measurement or count information is not clearly visible, state this explicitly
8. Prioritize safety concerns and code violations in your assessment
9. Be specific about the limitations of your analysis

Format your response according to the provided template structure.
'''
            
            # Send the message with image to the existing chat
            response = chat.send_message([ANALYSIS_PROMPT, processed_image])
            return response.text
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}\nDetails: Please ensure your API key is valid and you're using a supported image format."

    def start_chat(self):
        """Start a new chat session"""
        try:
            return self.model.start_chat(history=[])
        except Exception as e:
            return None

    def send_message(self, chat, message):
        """Send a message to the chat session"""
        try:
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            return f"Error sending message: {str(e)}"