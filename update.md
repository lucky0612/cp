#!/usr/bin/env python3

import logging
import os
import sys
from datetime import datetime

# Google Gemini imports
from google import genai
from google.api_core.exceptions import GoogleAPICallError
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GeminiTest")

# Configuration (Environment Variables or Config File)
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

def test_gemini_connection(prompt="Tell me about Vertex AI and Gemini models."):
    """
    Test connection to Gemini model via Vertex AI.
    
    Args:
        prompt (str): The prompt to send to Gemini.
        
    Returns:
        str: The response from Gemini.
    """
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Location: {REGION}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Prompt: {prompt}")
    
    try:
        # Initialize the Gemini client with Vertex AI
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=REGION,
        )
        
        # Define safety settings - all turned off for testing
        safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=safety_settings
        )
        
        # Note: Removing the list_models call that caused the error
        logger.info(f"Using model: {MODEL_NAME}")
        
        # Generate response
        logger.info("Generating response...")
        
        # Option 1: Streaming response
        response_text = ""
        for chunk in client.generate_content_stream(  # Using client.generate_content_stream directly
            model=MODEL_NAME,
            contents=[prompt],
            generation_config=generate_content_config,  # Changed parameter name to match API
        ):
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                continue
            response_text += chunk.text
            print(".", end="", flush=True)  # Show progress
        
        print()  # New line after progress indicator
        
        logger.info(f"Response length: {len(response_text)} characters")
        return response_text
        
    except GoogleAPICallError as e:
        logger.error(f"[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}] Failed to run the program with exception: {e}")
        sys.exit(100)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def test_non_streaming():
    """Test non-streaming version of the API"""
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=REGION,
        )
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
        )
        
        # Generate response (non-streaming)
        response = client.generate_content(  # Using client.generate_content directly
            model=MODEL_NAME,
            contents="Write a short poem about AI in 4 lines.",
            generation_config=generate_content_config,  # Changed parameter name to match API
        )
        
        logger.info("Non-streaming response:")
        logger.info(response.text)
        
        return response.text
    except Exception as e:
        logger.error(f"Non-streaming test failed: {e}")
        return None

class GenAIChat:
    """Class for interacting with Gemini models via Vertex AI."""
    
    def __init__(self):
        pass
        
    def generate_response_from_prompt(self, instructions=None, prompt=None):
        """
        Uses Vertex AI to generate response from a prompt.
        
        Args:
            instructions (str): System instructions for the model.
            prompt (str): The prompt.
        Returns:
            str: The response.
        """
        logger.info(f"Project id: {PROJECT_ID}")
        logger.info(f"Location: {REGION}")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Prompt: {prompt}")
        
        try:
            client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=REGION,
            )
            
            # Create system instruction if provided
            system_instruction = None
            if instructions:
                system_instruction = types.Part.from_text(text=instructions)
            
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=2,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ), 
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ), 
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ), 
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ]
            )
            
            # Stream the response
            response_text = ""
            
            # Create request with or without system instruction
            if system_instruction:
                for chunk in client.generate_content_stream(
                    model=MODEL_NAME,
                    contents=[system_instruction, prompt],
                    generation_config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content.parts:
                        continue
                    response_text += chunk.text
            else:
                for chunk in client.generate_content_stream(
                    model=MODEL_NAME,
                    contents=[prompt],
                    generation_config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content.parts:
                        continue
                    response_text += chunk.text
                    
            return response_text
            
        except GoogleAPICallError as e:
            logger.error(f"[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}] Failed to run the program with exception: {e}")
            sys.exit(100)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    print("==== Testing Gemini Connection ====")
    print("\n1. Testing with simple prompt - streaming:")
    response = test_gemini_connection("What are the key features of Vertex AI?")
    print("\nResponse from Gemini:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    print("\n2. Testing non-streaming API:")
    non_streaming_response = test_non_streaming()
    
    print("\n3. Testing with GenAIChat class:")
    chat = GenAIChat()
    instructions = "You are a helpful AI assistant that provides concise and accurate information."
    complex_prompt = "Create a 3-day itinerary for visiting New York City for the first time."
    class_response = chat.generate_response_from_prompt(instructions=instructions, prompt=complex_prompt)
    print("\nResponse from GenAIChat class:")
    print("-" * 50)
    print(class_response)
    print("-" * 50)
    
    print("\nAll tests completed.")
