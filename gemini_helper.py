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
            
            prompt = """You are a professional construction site inspector. Please analyze this construction site image and provide a detailed report with these sections:
            
            1. PROJECT OVERVIEW
            - Construction stage
            - Estimated completion percentage
            - Visible conditions
            
            2. CONSTRUCTION PROGRESS
            - Visible work status
            - Equipment present
            - Materials on site
            
            3. SAFETY AND QUALITY
            - Safety observations
            - Quality control points
            - Site security status
            
            Provide a thorough analysis that we can discuss further."""
            
            # Send the message with image to the existing chat
            response = chat.send_message([prompt, processed_image])
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