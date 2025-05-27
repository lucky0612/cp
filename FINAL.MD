#!/usr/bin/env python3
import os
import json
import logging
import subprocess
import tempfile
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional

# Web framework
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# AI/ML
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedOrchestrator")

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "prj-dv-cws-4363")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-001")

# Paths to your original scripts
CONFLUENCE_SCRIPT = "confluence_assistant.py"  # Your original Confluence script
JIRA_SCRIPT = "jira_chatbot.py"               # Your original Jira script

class QueryAnalyzer:
    """AI-powered query analyzer that determines how to split queries for individual scripts."""
    
    def __init__(self):
        try:
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.model = GenerativeModel(MODEL_NAME)
            logger.info(f"Query Analyzer initialized with {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            self.model = None
    
    def analyze_and_split_query(self, user_query):
        """Analyze user query and determine how to split it for Confluence and Jira."""
        if not self.model:
            # Fallback: use the same query for both
            return {
                "confluence_query": user_query,
                "jira_query": user_query,
                "analysis": "AI unavailable - using original query for both sources"
            }
        
        try:
            prompt = f"""You are a smart query analyzer. Analyze this user query and determine how to split it for two different systems:

USER QUERY: "{user_query}"

SYSTEMS:
1. Confluence: Contains documentation, procedures, how-to guides, setup instructions, troubleshooting guides
2. Jira: Contains tickets, issues, bugs, tasks, project tracking information

Your task: Create optimized queries for each system that will get the most relevant information.

ANALYSIS RULES:
1. If the query asks for BOTH documentation AND tickets (e.g., "rollout restart issues and how to solve them"):
   - Confluence query: Focus on procedures, solutions, documentation about the topic
   - Jira query: Focus on finding tickets, issues, problems related to the topic

2. If the query is primarily about documentation:
   - Confluence query: Use the full query
   - Jira query: Extract any ticket-related aspects or use related terms

3. If the query is primarily about tickets/issues:
   - Jira query: Use the full query  
   - Confluence query: Look for related documentation or procedures

4. For general topics:
   - Use similar but optimized queries for each system

EXAMPLES:
Query: "rollout restart issues and how to solve them"
→ Confluence: "rollout restart troubleshooting procedure solution"
→ Jira: "rollout restart issues problems"

Query: "BAMPS project setup"
→ Confluence: "BAMPS project setup documentation guide"
→ Jira: "BAMPS project tickets setup"

Query: "recent high priority bugs"
→ Jira: "recent high priority bugs"
→ Confluence: "bug resolution procedures troubleshooting"

Return your response in this JSON format:
{{
    "confluence_query": "optimized query for Confluence",
    "jira_query": "optimized query for Jira", 
    "analysis": "brief explanation of your reasoning"
}}"""

            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500
                )
            )
            
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].text
            
            # Try to parse JSON response
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    logger.info(f"Query analysis: {result['analysis']}")
                    return result
            except:
                pass
            
            # Fallback parsing
            lines = response_text.split('\n')
            confluence_query = user_query
            jira_query = user_query
            analysis = "Parsed from AI response"
            
            for line in lines:
                if 'confluence' in line.lower() and ':' in line:
                    confluence_query = line.split(':', 1)[1].strip().strip('"')
                elif 'jira' in line.lower() and ':' in line:
                    jira_query = line.split(':', 1)[1].strip().strip('"')
            
            return {
                "confluence_query": confluence_query,
                "jira_query": jira_query,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "confluence_query": user_query,
                "jira_query": user_query,
                "analysis": f"Analysis failed: {str(e)} - using original query"
            }

class ScriptOrchestrator:
    """Orchestrates your original scripts and combines their outputs."""
    
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        logger.info("Script Orchestrator initialized")
    
    def extract_confluence_final_answer(self, output):
        """Extract only the final answer from Confluence script output, skipping all errors and logs."""
        try:
            lines = output.split('\n')
            
            # Look for the "Answer (found in X seconds):" pattern
            answer_start_idx = -1
            for i, line in enumerate(lines):
                if re.search(r'Answer \(found in .* seconds\):', line):
                    answer_start_idx = i + 1
                    break
            
            if answer_start_idx == -1:
                # Fallback: look for "Successfully generated response" followed by actual content
                for i, line in enumerate(lines):
                    if "Successfully generated response" in line:
                        # Look for the actual answer after this line
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and not any(keyword in lines[j] for keyword in [
                                "REAssistant", "ERROR", "WARNING", "INFO", "Processing", "Selected", "Fetched"
                            ]):
                                answer_start_idx = j
                                break
                        break
            
            if answer_start_idx == -1:
                # Another fallback: look for lines that don't contain log keywords
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not any(keyword in line for keyword in [
                        "REAssistant", "ERROR", "WARNING", "INFO", "Processing", "Selected", 
                        "Fetched", "Connection", "Loading", "Found", "Executing", "urllib3",
                        "InsecureRequestWarning", "certificates", "verify=False", "packages"
                    ]):
                        clean_lines.append(line)
                
                if clean_lines:
                    # Join the clean lines and return
                    final_answer = '\n'.join(clean_lines)
                    logger.info("✅ Extracted Confluence answer using fallback method")
                    return final_answer
                else:
                    logger.warning("⚠️ Could not extract clean answer from Confluence output")
                    return "Confluence search completed, but could not extract clean answer from output."
            
            # Extract everything after "Answer (found in X seconds):"
            answer_lines = lines[answer_start_idx:]
            
            # Clean up the answer lines - remove any remaining log entries
            clean_answer_lines = []
            for line in answer_lines:
                line = line.strip()
                if line and not any(keyword in line for keyword in [
                    "REAssistant", "ERROR", "WARNING", "INFO", "Question:", "Processing"
                ]):
                    clean_answer_lines.append(line)
            
            if clean_answer_lines:
                final_answer = '\n'.join(clean_answer_lines)
                logger.info("✅ Successfully extracted Confluence final answer")
                return final_answer
            else:
                logger.warning("⚠️ Answer section found but appears to be empty")
                return "Confluence search completed, but the answer section appears to be empty."
                
        except Exception as e:
            logger.error(f"❌ Error extracting Confluence answer: {e}")
            return f"Error extracting Confluence answer: {str(e)}"
    
    def extract_jira_final_answer(self, output):
        """Extract the final answer from Jira script output, skipping logs."""
        try:
            lines = output.split('\n')
            
            # For Jira, look for the clean response after all the processing
            # Skip lines with "JiraGPT" logs and system messages
            clean_lines = []
            collecting_answer = False
            
            for line in lines:
                line = line.strip()
                
                # Skip log lines and system messages
                if any(keyword in line for keyword in [
                    "JiraGPT", "ERROR", "WARNING", "INFO", "Making request", "Loading", 
                    "Found", "Executing", "AI generated", "Query completed", "Processing",
                    "urllib3", "InsecureRequestWarning", "certificates"
                ]):
                    continue
                
                # If we find a line that looks like it's starting the actual response
                if line and not line.startswith('=') and not line.startswith('-'):
                    collecting_answer = True
                
                if collecting_answer and line:
                    clean_lines.append(line)
            
            if clean_lines:
                final_answer = '\n'.join(clean_lines)
                logger.info("✅ Successfully extracted Jira final answer")
                return final_answer
            else:
                logger.warning("⚠️ Could not extract clean answer from Jira output")
                return "Jira search completed, but could not extract clean answer from output."
                
        except Exception as e:
            logger.error(f"❌ Error extracting Jira answer: {e}")
            return f"Error extracting Jira answer: {str(e)}"
    
    def run_confluence_script(self, query):
        """Run your original Confluence script and extract only the final answer."""
        try:
            logger.info(f"🔍 Running Confluence script with query: '{query}'")
            
            # Create a temporary script that runs your original Confluence assistant
            temp_script = f'''
import sys
import os
sys.path.append('.')

# Set up environment to suppress warnings if needed
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    # Import your original Confluence script
    from confluence_assistant import main, REAssistant
    
    # Initialize your original assistant
    confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
    confluence_username = os.environ.get("CONFLUENCE_USERNAME", "lakshya.vijay@cmegroup.com")
    confluence_api_token = os.environ.get("CONFLUENCE_API_TOKEN", "")
    confluence_space = os.environ.get("CONFLUENCE_SPACE", "RE")
    
    assistant = REAssistant(confluence_url, confluence_username, confluence_api_token, confluence_space)
    
    if not assistant.initialize():
        print("CONFLUENCE_ERROR: Failed to initialize")
        sys.exit(1)
    
    # Run the query and capture the result
    result = assistant.answer_question("{query}")
    
    # Output markers for extraction
    print("CONFLUENCE_ANSWER_START")
    print(result)
    print("CONFLUENCE_ANSWER_END")
    
except Exception as e:
    print(f"CONFLUENCE_ERROR: {{str(e)}}")
    sys.exit(1)
'''
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(temp_script)
                temp_file = f.name
            
            try:
                # Run the script with suppressed stderr to avoid SSL warnings
                env = os.environ.copy()
                env['PYTHONWARNINGS'] = 'ignore'
                
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minute timeout
                    env=env
                )
                
                output = result.stdout
                error_output = result.stderr
                
                # Log the raw output for debugging
                logger.debug(f"Confluence raw stdout: {output[:500]}...")
                if error_output:
                    logger.debug(f"Confluence stderr: {error_output[:500]}...")
                
                # Check for our answer markers first
                if "CONFLUENCE_ANSWER_START" in output and "CONFLUENCE_ANSWER_END" in output:
                    start_idx = output.find("CONFLUENCE_ANSWER_START") + len("CONFLUENCE_ANSWER_START")
                    end_idx = output.find("CONFLUENCE_ANSWER_END")
                    confluence_result = output[start_idx:end_idx].strip()
                    logger.info("✅ Confluence script completed successfully using markers")
                    return confluence_result
                
                # Check for error markers
                if "CONFLUENCE_ERROR:" in output:
                    error_msg = output.split("CONFLUENCE_ERROR:")[1].split('\n')[0].strip()
                    logger.error(f"❌ Confluence script error: {error_msg}")
                    return f"Confluence search failed: {error_msg}"
                
                # If no markers, try to extract the final answer from all output
                if result.returncode == 0 and output.strip():
                    confluence_result = self.extract_confluence_final_answer(output)
                    return confluence_result
                else:
                    error_msg = error_output or output or "Unknown error"
                    logger.error(f"❌ Confluence script failed with return code {result.returncode}")
                    return f"Confluence search failed: {error_msg[:200]}"
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Confluence script timed out")
            return "Confluence search timed out. Please try again with a simpler query."
        except Exception as e:
            logger.error(f"❌ Error running Confluence script: {e}")
            return f"Error running Confluence search: {str(e)}"
    
    def run_jira_script(self, query):
        """Run your original Jira script and extract only the final answer."""
        try:
            logger.info(f"🎫 Running Jira script with query: '{query}'")
            
            # Create a temporary script that runs your original Jira chatbot
            temp_script = f'''
import sys
import os
sys.path.append('.')

# Set up environment to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    # Import your original Jira script
    from jira_chatbot import TrulyVersatileJiraChatbot
    
    # Initialize your original chatbot
    chatbot = TrulyVersatileJiraChatbot()
    
    # Run the query and capture the result
    result = chatbot.answer_any_question("{query}")
    
    # Output markers for extraction
    print("JIRA_ANSWER_START")
    print(result)
    print("JIRA_ANSWER_END")
    
except Exception as e:
    print(f"JIRA_ERROR: {{str(e)}}")
    sys.exit(1)
'''
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(temp_script)
                temp_file = f.name
            
            try:
                # Run the script with suppressed stderr
                env = os.environ.copy()
                env['PYTHONWARNINGS'] = 'ignore'
                
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minute timeout
                    env=env
                )
                
                output = result.stdout
                error_output = result.stderr
                
                # Log the raw output for debugging
                logger.debug(f"Jira raw stdout: {output[:500]}...")
                if error_output:
                    logger.debug(f"Jira stderr: {error_output[:500]}...")
                
                # Check for our answer markers first
                if "JIRA_ANSWER_START" in output and "JIRA_ANSWER_END" in output:
                    start_idx = output.find("JIRA_ANSWER_START") + len("JIRA_ANSWER_START")
                    end_idx = output.find("JIRA_ANSWER_END")
                    jira_result = output[start_idx:end_idx].strip()
                    logger.info("✅ Jira script completed successfully using markers")
                    return jira_result
                
                # Check for error markers
                if "JIRA_ERROR:" in output:
                    error_msg = output.split("JIRA_ERROR:")[1].split('\n')[0].strip()
                    logger.error(f"❌ Jira script error: {error_msg}")
                    return f"Jira search failed: {error_msg}"
                
                # If no markers, try to extract the final answer from all output
                if result.returncode == 0 and output.strip():
                    jira_result = self.extract_jira_final_answer(output)
                    return jira_result
                else:
                    error_msg = error_output or output or "Unknown error"
                    logger.error(f"❌ Jira script failed with return code {result.returncode}")
                    return f"Jira search failed: {error_msg[:200]}"
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Jira script timed out")
            return "Jira search timed out. Please try again with a simpler query."
        except Exception as e:
            logger.error(f"❌ Error running Jira script: {e}")
            return f"Error running Jira search: {str(e)}"

class ResponseCombiner:
    """Combines outputs from both scripts using AI."""
    
    def __init__(self):
        try:
            vertexai.init(project=PROJECT_ID, location=REGION)
            self.model = GenerativeModel(MODEL_NAME)
            logger.info("Response Combiner initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Response Combiner: {e}")
            self.model = None
    
    def combine_responses(self, user_query, confluence_result, jira_result, query_analysis):
        """Intelligently combine outputs from both scripts."""
        if not self.model:
            # Simple fallback combination
            combined = f"## Combined Results for: {user_query}\n\n"
            
            if confluence_result and not confluence_result.startswith("Error") and not confluence_result.startswith("Confluence search failed"):
                combined += f"### 📚 Documentation (Confluence)\n{confluence_result}\n\n"
            
            if jira_result and not jira_result.startswith("Error") and not jira_result.startswith("Jira search failed"):
                combined += f"### 🎫 Tickets (Jira)\n{jira_result}\n\n"
            
            return combined
        
        try:
            logger.info("🤖 Combining responses with AI...")
            
            prompt = f"""You are the CME Unified Knowledge Assistant. You have received results from two different systems for a user's query. Your job is to combine these results into one comprehensive, intelligent response.

USER ORIGINAL QUERY: "{user_query}"

QUERY ANALYSIS: {query_analysis.get('analysis', 'No analysis available')}

CONFLUENCE RESULTS (Documentation):
{confluence_result}

JIRA RESULTS (Tickets):
{jira_result}

YOUR TASK: Create a unified, intelligent response that:

1. **Starts with a direct answer** to the user's question
2. **Combines information logically** - show how documentation and tickets relate to each other
3. **Maintains the quality** of both individual responses
4. **Uses clear sections** to organize the information
5. **Preserves all links and formatting** from the original responses
6. **Is conversational and helpful** - like talking to a knowledgeable colleague
7. **Highlights key insights** that come from having both sources

FORMATTING GUIDELINES:
- Use markdown formatting (headers, bold, bullet points)
- Keep all ticket links from Jira results
- Keep all documentation links from Confluence results
- Use sections like "## Overview", "## Current Issues", "## Documentation", etc.
- Make it flow naturally - don't just append one after the other

EXAMPLE STRUCTURE (adapt based on content):
## [Topic] - Complete Overview

[Direct answer to the question]

### Current Situation (from Jira)
[Relevant tickets and their status]

### Procedures & Documentation (from Confluence)  
[Relevant documentation and how-to guides]

### Key Insights
[How the tickets and documentation relate, patterns, recommendations]

Now create your unified response:"""

            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            result = response.text if hasattr(response, 'text') else response.candidates[0].text
            logger.info("✅ AI response combination completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ AI combination failed: {e}")
            # Fallback to simple combination
            combined = f"## Combined Results for: {user_query}\n\n"
            
            if confluence_result and not confluence_result.startswith("Error"):
                combined += f"### 📚 Documentation (Confluence)\n{confluence_result}\n\n"
            
            if jira_result and not jira_result.startswith("Error"):
                combined += f"### 🎫 Tickets (Jira)\n{jira_result}\n\n"
            
            combined += f"*Note: AI combination failed ({str(e)}), showing individual results*"
            return combined

class UnifiedKnowledgeOrchestrator:
    """Main orchestrator that decides whether to use individual scripts or combine them."""
    
    def __init__(self):
        self.script_orchestrator = ScriptOrchestrator()
        self.response_combiner = ResponseCombiner()
        logger.info("🎭 Unified Knowledge Orchestrator initialized")
    
    def process_query(self, user_query, selected_sources):
        """Process user query based on selected sources."""
        start_time = datetime.now()
        logger.info(f"🚀 Processing query: '{user_query}' with sources: {selected_sources}")
        
        try:
            # Confluence only
            if selected_sources == ['confluence']:
                logger.info("📚 Confluence-only mode - using original script")
                result = self.script_orchestrator.run_confluence_script(user_query)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Confluence query completed in {elapsed:.2f} seconds")
                return result
            
            # Jira only  
            elif selected_sources == ['jira']:
                logger.info("🎫 Jira-only mode - using original script")
                result = self.script_orchestrator.run_jira_script(user_query)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Jira query completed in {elapsed:.2f} seconds")
                return result
            
            # Both sources - this is where the magic happens!
            elif 'confluence' in selected_sources and 'jira' in selected_sources:
                logger.info("🎭 Both sources mode - analyzing and orchestrating")
                
                # Step 1: Analyze the query to optimize for each system
                query_analysis = self.script_orchestrator.query_analyzer.analyze_and_split_query(user_query)
                logger.info(f"📊 Query analysis: {query_analysis['analysis']}")
                
                # Step 2: Run both scripts with optimized queries
                confluence_query = query_analysis['confluence_query']
                jira_query = query_analysis['jira_query']
                
                logger.info(f"📚 Confluence query: '{confluence_query}'")
                logger.info(f"🎫 Jira query: '{jira_query}'")
                
                # Run both scripts sequentially
                confluence_result = self.script_orchestrator.run_confluence_script(confluence_query)
                jira_result = self.script_orchestrator.run_jira_script(jira_query)
                
                # Step 3: Combine the results intelligently
                combined_result = self.response_combiner.combine_responses(
                    user_query, confluence_result, jira_result, query_analysis
                )
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Combined query completed in {elapsed:.2f} seconds")
                return combined_result
            
            else:
                return "Please select at least one data source (Confluence or Jira)."
                
        except Exception as e:
            logger.error(f"❌ Error in orchestrator: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try again."
    
    def get_status(self):
        """Get system status by testing the original scripts."""
        logger.info("📊 Checking system status...")
        
        confluence_status = False
        jira_status = False
        
        # Test Confluence by trying to import and initialize
        try:
            import confluence_assistant
            confluence_status = True
            logger.info("✅ Confluence script accessible")
        except Exception as e:
            logger.error(f"❌ Confluence script not accessible: {e}")
        
        # Test Jira by trying to import and initialize  
        try:
            import jira_chatbot
            jira_status = True
            logger.info("✅ Jira script accessible")
        except Exception as e:
            logger.error(f"❌ Jira script not accessible: {e}")
        
        status = {
            'confluence': confluence_status,
            'jira': jira_status,
            'confluence_pages': 0,  # Could be enhanced to get actual count
            'status': 'healthy' if confluence_status or jira_status else 'degraded'
        }
        
        logger.info(f"📊 System status: {status}")
        return status

# Flask Application
app = Flask(__name__)
CORS(app)

# Initialize the orchestrator
logger.info("🎭 Starting CME Unified Knowledge Orchestrator...")
orchestrator = UnifiedKnowledgeOrchestrator()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    logger.info("📨 Received chat request")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_query = data.get('message', '').strip()
        selected_sources = data.get('sources', [])
        
        logger.info(f"📝 Query: '{user_query}', Sources: {selected_sources}")
        
        if not user_query:
            return jsonify({'error': 'No message provided'}), 400
        
        if not selected_sources:
            return jsonify({'error': 'No sources selected'}), 400
        
        # Handle basic greetings
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['hello', 'hi', 'hey']):
            response = """👋 **Hello! I'm your CME Unified Knowledge Orchestrator!**

🎭 **How I work:**
• **Individual Sources**: Use your original working scripts directly
• **Combined Sources**: Intelligently orchestrate both scripts and merge results
• **Perfect Quality**: Guaranteed same quality as your individual scripts

🔍 **I can help you with:**
• **Confluence Only:** Documentation, procedures, how-to guides  
• **Jira Only:** Tickets, issues, bugs, project tracking
• **Both Together:** Complete insights combining documentation with current tickets

🚀 **Try asking:**
• "Show me rollout restart tickets and how to solve them" (Both)
• "BAMPS project documentation" (Confluence)  
• "Recent high priority issues" (Jira)

What would you like to explore? 🤔"""
            return jsonify({'response': response})
        
        if any(word in query_lower for word in ['bye', 'goodbye', 'thanks']):
            response = "👋 **Goodbye!** Your original scripts are always ready for perfect individual results, and I'm here when you need them combined! 🚀"
            return jsonify({'response': response})
        
        # Process the query using the orchestrator
        response = orchestrator.process_query(user_query, selected_sources)
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Check system status."""
    try:
        status_data = orchestrator.get_status()
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"❌ Error in status endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎭 Starting CME Unified Knowledge Orchestrator...")
    print("📚 Confluence: Using your original confluence_assistant.py")
    print("🎫 Jira: Using your original jira_chatbot.py") 
    print(f"🤖 AI Orchestrator: {MODEL_NAME}")
    print("=" * 60)
    print("🎯 INDIVIDUAL SOURCES: Perfect original script results")
    print("🎭 COMBINED SOURCES: AI-orchestrated intelligent combination")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
