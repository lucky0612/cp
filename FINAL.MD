#!/usr/bin/env python3
import os
import json
import logging
import subprocess
import tempfile
import sys
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
    
    def run_confluence_script(self, query):
        """Run your original Confluence script and capture output."""
        try:
            logger.info(f"🔍 Running Confluence script with query: '{query}'")
            
            # Create a temporary script that imports your original code and runs the query
            temp_script = f'''
import sys
sys.path.append('.')

# Import your original Confluence script
from confluence_assistant import REAssistant

# Initialize with environment variables
import os
confluence_url = os.environ.get("CONFLUENCE_URL", "https://cmegroup.atlassian.net")
confluence_username = os.environ.get("CONFLUENCE_USERNAME", "lalshya.vijay@cmegroup.com")
confluence_api_token = os.environ.get("CONFLUENCE_API_TOKEN", "")
confluence_space = os.environ.get("CONFLUENCE_SPACE", "RE")

# Initialize your original assistant
assistant = REAssistant(confluence_url, confluence_username, confluence_api_token, confluence_space)

if not assistant.initialize():
    print("ERROR: Failed to initialize Confluence assistant")
    sys.exit(1)

# Run the query
try:
    result = assistant.answer_question("{query}")
    print("CONFLUENCE_RESULT_START")
    print(result)
    print("CONFLUENCE_RESULT_END")
except Exception as e:
    print("ERROR: " + str(e))
'''
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(temp_script)
                temp_file = f.name
            
            try:
                # Run the script
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                if result.returncode == 0:
                    # Extract the result from stdout
                    output = result.stdout
                    if "CONFLUENCE_RESULT_START" in output and "CONFLUENCE_RESULT_END" in output:
                        start_idx = output.find("CONFLUENCE_RESULT_START") + len("CONFLUENCE_RESULT_START")
                        end_idx = output.find("CONFLUENCE_RESULT_END")
                        confluence_result = output[start_idx:end_idx].strip()
                        logger.info("✅ Confluence script completed successfully")
                        return confluence_result
                    else:
                        logger.warning("⚠️ Confluence script output format unexpected")
                        return output.strip()
                else:
                    error_msg = result.stderr or result.stdout
                    logger.error(f"❌ Confluence script failed: {error_msg}")
                    return f"Confluence search failed: {error_msg}"
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Error running Confluence script: {e}")
            return f"Error running Confluence search: {str(e)}"
    
    def run_jira_script(self, query):
        """Run your original Jira script and capture output."""
        try:
            logger.info(f"🎫 Running Jira script with query: '{query}'")
            
            # Create a temporary script that imports your original code and runs the query
            temp_script = f'''
import sys
sys.path.append('.')

# Import your original Jira script
from jira_chatbot import TrulyVersatileJiraChatbot

# Initialize your original chatbot
chatbot = TrulyVersatileJiraChatbot()

# Run the query
try:
    result = chatbot.answer_any_question("{query}")
    print("JIRA_RESULT_START")
    print(result)
    print("JIRA_RESULT_END")
except Exception as e:
    print("ERROR: " + str(e))
'''
            
            # Write temporary script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(temp_script)
                temp_file = f.name
            
            try:
                # Run the script
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                if result.returncode == 0:
                    # Extract the result from stdout
                    output = result.stdout
                    if "JIRA_RESULT_START" in output and "JIRA_RESULT_END" in output:
                        start_idx = output.find("JIRA_RESULT_START") + len("JIRA_RESULT_START")
                        end_idx = output.find("JIRA_RESULT_END")
                        jira_result = output[start_idx:end_idx].strip()
                        logger.info("✅ Jira script completed successfully")
                        return jira_result
                    else:
                        logger.warning("⚠️ Jira script output format unexpected")
                        return output.strip()
                else:
                    error_msg = result.stderr or result.stdout
                    logger.error(f"❌ Jira script failed: {error_msg}")
                    return f"Jira search failed: {error_msg}"
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
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
            combined += f"### Documentation (Confluence)\n{confluence_result}\n\n"
            combined += f"### Tickets (Jira)\n{jira_result}"
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
                
                # Run both scripts in parallel would be ideal, but for simplicity run sequentially
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
            from confluence_assistant import REAssistant
            confluence_status = True
            logger.info("✅ Confluence script accessible")
        except Exception as e:
            logger.error(f"❌ Confluence script not accessible: {e}")
        
        # Test Jira by trying to import and initialize  
        try:
            from jira_chatbot import TrulyVersatileJiraChatbot
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



















// CME Unified Knowledge Orchestrator - Frontend JavaScript
class UnifiedOrchestrator {
    constructor() {
        this.apiBase = '/api';
        this.isTyping = false;
        this.messageHistory = [];
        this.particleCount = 0;
        this.maxParticles = 20;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeParticles();
        this.checkSystemStatus();
        this.setupSourceTracking();
        this.initializeAnimations();
        this.setupAccessibility();
        
        console.log('🎭 CME Unified Knowledge Orchestrator initialized');
    }

    setupEventListeners() {
        // Message input handling
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const clearChat = document.getElementById('clearChat');

        if (messageInput) {
            messageInput.addEventListener('input', this.handleInputChange.bind(this));
            messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
            messageInput.addEventListener('paste', this.handlePaste.bind(this));
        }

        if (sendButton) {
            sendButton.addEventListener('click', this.sendMessage.bind(this));
        }

        if (clearChat) {
            clearChat.addEventListener('click', this.clearChatHistory.bind(this));
        }

        // Source selector handling
        const sourceCheckboxes = document.querySelectorAll('.source-option input[type="checkbox"]');
        sourceCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', this.updateSelectedSources.bind(this));
        });

        // Example query handling
        const exampleQueries = document.querySelectorAll('.example-query');
        exampleQueries.forEach(query => {
            query.addEventListener('click', this.handleExampleQuery.bind(this));
        });

        // Floating help
        const floatingHelp = document.getElementById('floatingHelp');
        if (floatingHelp) {
            floatingHelp.addEventListener('click', this.toggleHelp.bind(this));
        }

        // Window events
        window.addEventListener('resize', this.handleResize.bind(this));
        window.addEventListener('scroll', this.handleScroll.bind(this));
        
        // Parallax effect
        document.addEventListener('mousemove', this.handleMouseMove.bind(this));
    }

    handleInputChange(event) {
        const input = event.target;
        const charCount = document.getElementById('charCount');
        const sendButton = document.getElementById('sendButton');

        // Update character count
        if (charCount) {
            charCount.textContent = input.value.length;
        }

        // Auto-resize textarea
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';

        // Enable/disable send button
        if (sendButton) {
            sendButton.disabled = !input.value.trim();
        }

        // Update placeholder based on sources
        this.updateInputPlaceholder();
    }

    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!this.isTyping && event.target.value.trim()) {
                this.sendMessage();
            }
        }
    }

    handlePaste(event) {
        // Handle paste events
        setTimeout(() => {
            this.handleInputChange(event);
        }, 0);
    }

    async sendMessage() {
        console.log('🚀 Send message triggered');
        
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isTyping) {
            console.log('❌ Message empty or currently typing, aborting');
            return;
        }

        // Get selected sources
        const selectedSources = this.getSelectedSources();
        console.log('📊 Selected sources:', selectedSources);
        
        if (selectedSources.length === 0) {
            this.showToast('Please select at least one data source', 'warning');
            return;
        }

        console.log('📝 Sending message:', message);

        // Clear input and hide welcome
        messageInput.value = '';
        this.handleInputChange({ target: messageInput });
        this.hideWelcomeSection();

        // Add user message immediately
        this.addMessage(message, 'user');
        
        // Show enhanced typing indicator based on sources
        this.showEnhancedTypingIndicator(selectedSources);

        try {
            console.log('🌐 Making API request...');
            
            // Send to orchestrator backend
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    sources: selectedSources
                })
            });

            console.log('📡 API response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('❌ API error response:', errorText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('📦 API response data received');
            
            // Hide typing indicator
            this.hideTypingIndicator();

            // Add assistant response
            if (data.response) {
                this.addMessage(data.response, 'assistant');
                console.log('✅ Assistant message added');
                
                // Show success notification for combined queries
                if (selectedSources.length > 1) {
                    this.showToast('Combined results from both systems', 'success', 3000);
                }
            } else {
                console.error('❌ No response in API data:', data);
                this.addMessage('I received an empty response. Please try again.', 'assistant');
            }
            
            // Store in history with additional metadata
            this.messageHistory.push({
                user: message,
                assistant: data.response || 'Empty response',
                sources: selectedSources,
                timestamp: new Date().toISOString(),
                processingTime: data.processingTime || 'unknown'
            });

        } catch (error) {
            console.error('❌ Error sending message:', error);
            this.hideTypingIndicator();
            
            const errorMessage = `I apologize, but I'm having trouble connecting to the knowledge systems right now. Please try again in a moment.\n\n**Error Details:** ${error.message}`;
            this.addMessage(errorMessage, 'assistant');
            this.showToast('Connection error - please try again', 'error');
        }
    }

    addMessage(content, sender) {
        console.log(`📨 Adding ${sender} message:`, content.substring(0, 100) + '...');
        
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            console.error('❌ Chat messages container not found');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : '🎭';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        // Enhanced message formatting for orchestrator responses
        messageBubble.innerHTML = this.formatMessage(content);

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();

        messageContent.appendChild(messageBubble);
        messageContent.appendChild(messageTime);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        chatMessages.appendChild(messageDiv);

        // Force reflow before animation
        messageDiv.offsetHeight;

        // Animate in
        messageDiv.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';

        // Scroll to bottom
        this.scrollToBottom();
        
        console.log(`✅ ${sender} message added successfully`);
    }

    formatMessage(content) {
        // Enhanced message formatting with better support for orchestrator responses
        let formatted = content
            // Headers with proper hierarchy
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks with language detection
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                const language = lang ? ` data-language="${lang}"` : '';
                return `<pre><code${language}>${this.escapeHtml(code.trim())}</code></pre>`;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links with proper accessibility - enhanced for Jira tickets
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
                if (url.includes('browse/')) {
                    // Jira ticket link
                    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="jira-link" aria-label="${text} (opens in new tab)">🎫 ${text} <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>`;
                } else if (url.includes('confluence') || url.includes('pages/viewpage')) {
                    // Confluence page link
                    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="confluence-link" aria-label="${text} (opens in new tab)">📚 ${text} <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>`;
                } else {
                    // Generic link
                    return `<a href="${url}" target="_blank" rel="noopener noreferrer" aria-label="${text} (opens in new tab)">${text} <i class="fas fa-external-link-alt" aria-hidden="true"></i></a>`;
                }
            })
            // Blockquotes
            .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        // Wrap in paragraphs if not already wrapped
        if (!formatted.startsWith('<')) {
            formatted = '<p>' + formatted + '</p>';
        }

        // Handle lists with proper structure
        formatted = this.formatLists(formatted);

        // Handle tables
        formatted = this.formatTables(formatted);

        // Add special styling for orchestrator sections
        formatted = this.enhanceOrchestratorSections(formatted);

        return formatted;
    }

    enhanceOrchestratorSections(content) {
        // Add special styling for sections that indicate source
        content = content.replace(
            /(### .*(?:Confluence|Documentation).*)/gi,
            '<div class="confluence-section">$1</div>'
        );
        
        content = content.replace(
            /(### .*(?:Jira|Tickets|Issues).*)/gi,
            '<div class="jira-section">$1</div>'
        );

        // Enhance "Key Insights" or "Overview" sections
        content = content.replace(
            /(### .*(?:Key Insights|Overview|Summary).*)/gi,
            '<div class="insights-section">$1</div>'
        );

        return content;
    }

    showEnhancedTypingIndicator(sources) {
        this.isTyping = true;
        const indicator = document.getElementById('typingIndicator');
        const typingText = indicator?.querySelector('.typing-text');
        
        if (indicator) {
            indicator.classList.add('show');
        }
        
        if (typingText) {
            let message = 'AI is thinking...';
            
            if (sources.length === 1) {
                if (sources.includes('confluence')) {
                    message = 'Searching Confluence documentation...';
                } else if (sources.includes('jira')) {
                    message = 'Searching Jira tickets...';
                }
            } else if (sources.length > 1) {
                message = 'Orchestrating both systems and combining results...';
            }
            
            typingText.textContent = message;
        }
        
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.classList.remove('show');
        }
    }

    scrollToBottom() {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        }
    }

    hideWelcomeSection() {
        const welcomeSection = document.getElementById('welcomeSection');
        if (welcomeSection) {
            welcomeSection.style.display = 'none';
        }
    }

    clearChatHistory() {
        const chatMessages = document.getElementById('chatMessages');
        const welcomeSection = document.getElementById('welcomeSection');
        
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        if (welcomeSection) {
            welcomeSection.style.display = 'block';
        }
        
        this.messageHistory = [];
        this.showToast('Chat history cleared', 'success');
    }

    getSelectedSources() {
        const sources = [];
        const confluenceCheck = document.getElementById('confluence');
        const jiraCheck = document.getElementById('jira');

        if (confluenceCheck && confluenceCheck.checked) {
            sources.push('confluence');
        }
        if (jiraCheck && jiraCheck.checked) {
            sources.push('jira');
        }

        return sources;
    }

    updateSelectedSources() {
        const sourcesList = document.getElementById('sourcesList');
        if (!sourcesList) return;

        const sources = this.getSelectedSources();
        let displayText = '';

        if (sources.includes('confluence') && sources.includes('jira')) {
            displayText = 'Confluence + Jira (Orchestrated)';
        } else if (sources.includes('confluence')) {
            displayText = 'Confluence Only (Original Script)';
        } else if (sources.includes('jira')) {
            displayText = 'Jira Only (Original Script)';
        } else {
            displayText = 'No Sources Selected';
        }

        sourcesList.textContent = displayText;
        this.updateInputPlaceholder();
    }

    updateInputPlaceholder() {
        const messageInput = document.getElementById('messageInput');
        if (!messageInput) return;

        const sources = this.getSelectedSources();
        let placeholder = '';

        if (sources.includes('confluence') && sources.includes('jira')) {
            placeholder = 'Ask me anything - I\'ll orchestrate both systems and combine results...';
        } else if (sources.includes('confluence')) {
            placeholder = 'Ask me about CME\'s documentation and procedures...';
        } else if (sources.includes('jira')) {
            placeholder = 'Ask me about tickets, issues, and project tracking...';
        } else {
            placeholder = 'Please select at least one data source first...';
        }

        messageInput.placeholder = placeholder;
    }

    handleExampleQuery(event) {
        const query = event.currentTarget.getAttribute('data-query');
        if (query) {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.value = query;
                this.handleInputChange({ target: messageInput });
                messageInput.focus();
                
                // Auto-select appropriate sources based on query
                this.smartSourceSelection(query);
            }
        }
    }

    smartSourceSelection(query) {
        const queryLower = query.toLowerCase();
        const confluenceCheck = document.getElementById('confluence');
        const jiraCheck = document.getElementById('jira');
        
        // Smart source selection based on query content
        if (queryLower.includes('and how to solve') || queryLower.includes('documentation')) {
            // Queries that likely need both
            if (confluenceCheck) confluenceCheck.checked = true;
            if (jiraCheck) jiraCheck.checked = true;
        } else if (queryLower.includes('documentation') || queryLower.includes('guide') || queryLower.includes('setup')) {
            // Confluence-focused queries
            if (confluenceCheck) confluenceCheck.checked = true;
            if (jiraCheck) jiraCheck.checked = false;
        } else if (queryLower.includes('tickets') || queryLower.includes('issues') || queryLower.includes('bugs')) {
            // Jira-focused queries
            if (confluenceCheck) confluenceCheck.checked = false;
            if (jiraCheck) jiraCheck.checked = true;
        }
        
        this.updateSelectedSources();
    }

    async checkSystemStatus() {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = statusIndicator?.querySelector('.status-text');

        try {
            const response = await fetch(`${this.apiBase}/status`);
            const data = await response.json();

            let status = 'error';
            let text = 'System Error';

            if (response.ok) {
                if (data.confluence && data.jira) {
                    status = 'connected';
                    text = 'Both Scripts Ready';
                } else if (data.confluence || data.jira) {
                    status = 'degraded';
                    text = 'Partial Connection';
                } else {
                    status = 'error';
                    text = 'Scripts Not Found';
                }
            }

            if (statusIndicator) {
                statusIndicator.className = `status-indicator ${status}`;
            }
            if (statusText) {
                statusText.textContent = text;
            }

        } catch (error) {
            console.error('Status check failed:', error);
            if (statusIndicator) {
                statusIndicator.className = 'status-indicator error';
            }
            if (statusText) {
                statusText.textContent = 'Connection Error';
            }
        }

        // Check status every 30 seconds
        setTimeout(() => this.checkSystemStatus(), 30000);
    }

    formatLists(content) {
        // Unordered lists
        content = content.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

        // Ordered lists
        content = content.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        
        return content;
    }

    formatTables(content) {
        // Simple table detection and formatting
        const lines = content.split('\n');
        let inTable = false;
        let tableRows = [];
        let result = [];

        lines.forEach(line => {
            if (line.includes('|') && line.split('|').length > 2) {
                if (!inTable) {
                    inTable = true;
                    tableRows = [];
                }
                tableRows.push(line);
            } else {
                if (inTable) {
                    result.push(this.buildTable(tableRows));
                    inTable = false;
                    tableRows = [];
                }
                result.push(line);
            }
        });

        if (inTable && tableRows.length > 0) {
            result.push(this.buildTable(tableRows));
        }

        return result.join('\n');
    }

    buildTable(rows) {
        if (rows.length === 0) return '';

        let table = '<table role="table"><thead><tr>';
        
        // Header row
        const headerCells = rows[0].split('|').map(cell => cell.trim()).filter(cell => cell);
        headerCells.forEach(cell => {
            table += `<th scope="col">${cell}</th>`;
        });
        table += '</tr></thead><tbody>';

        // Data rows (skip header and separator if exists)
        const dataRows = rows.slice(rows[1] && rows[1].includes('---') ? 2 : 1);
        dataRows.forEach(row => {
            const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell);
            if (cells.length > 0) {
                table += '<tr>';
                cells.forEach(cell => {
                    table += `<td>${cell}</td>`;
                });
                table += '</tr>';
            }
        });

        table += '</tbody></table>';
        return table;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initialize remaining methods (particles, animations, etc.)
    initializeParticles() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        // Create initial particles
        for (let i = 0; i < this.maxParticles; i++) {
            setTimeout(() => this.createParticle(), i * 200);
        }

        // Continue creating particles
        setInterval(() => {
            if (this.particleCount < this.maxParticles) {
                this.createParticle();
            }
        }, 1000);
    }

    createParticle() {
        const particlesContainer = document.getElementById('particles');
        if (!particlesContainer) return;

        const particle = document.createElement('div');
        const type = Math.floor(Math.random() * 3) + 1;
        
        particle.className = `particle particle-${type}`;
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (8 + Math.random() * 8) + 's';
        particle.style.animationDelay = Math.random() * 2 + 's';

        particlesContainer.appendChild(particle);
        this.particleCount++;

        // Remove particle after animation
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
                this.particleCount--;
            }
        }, 16000);
    }

    initializeAnimations() {
        // Add stagger animation to example queries
        const exampleQueries = document.querySelectorAll('.example-query');
        exampleQueries.forEach((query, index) => {
            query.classList.add('stagger-item');
            query.style.animationDelay = `${0.1 + index * 0.1}s`;
        });

        // Add morphing shapes
        this.createMorphingShapes();
        
        // Setup intersection observer for scroll animations
        this.setupScrollAnimations();
    }

    createMorphingShapes() {
        const backgroundPattern = document.querySelector('.background-pattern');
        if (!backgroundPattern) return;

        const shapesContainer = document.createElement('div');
        shapesContainer.className = 'morphing-shapes';

        // Create 3 morphing shapes
        for (let i = 1; i <= 3; i++) {
            const shape = document.createElement('div');
            shape.className = 'morphing-shape';
            shapesContainer.appendChild(shape);
        }

        backgroundPattern.appendChild(shapesContainer);
    }

    setupScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);

        // Observe elements for scroll animations
        document.querySelectorAll('.fade-in-viewport').forEach(el => {
            observer.observe(el);
        });
    }

    handleMouseMove(event) {
        // Parallax effect for floating elements
        const floatingElements = document.querySelectorAll('.floating-element');
        const { clientX, clientY } = event;
        const { innerWidth, innerHeight } = window;

        const xPercent = (clientX / innerWidth) * 100;
        const yPercent = (clientY / innerHeight) * 100;

        floatingElements.forEach((element, index) => {
            const speed = 0.05 + (index * 0.02);
            const x = (xPercent - 50) * speed;
            const y = (yPercent - 50) * speed;
            
            element.style.transform = `translate(${x}px, ${y}px)`;
        });

        // Update morphing shapes based on mouse position
        const shapes = document.querySelectorAll('.morphing-shape');
        shapes.forEach((shape, index) => {
            const speed = 0.02 + (index * 0.01);
            const x = (xPercent - 50) * speed;
            const y = (yPercent - 50) * speed;
            
            const currentTransform = shape.style.transform || '';
            shape.style.transform = `${currentTransform} translate(${x}px, ${y}px)`;
        });
    }

    handleResize() {
        // Adjust particle count based on screen size
        const width = window.innerWidth;
        
        if (width < 768) {
            this.maxParticles = 10;
        } else if (width < 1024) {
            this.maxParticles = 15;
        } else {
            this.maxParticles = 20;
        }

        // Update input placeholder
        this.updateInputPlaceholder();
    }

    handleScroll() {
        // Parallax scrolling effect
        const scrolled = window.pageYOffset;
        const parallaxElements = document.querySelectorAll('.parallax-element');
        
        parallaxElements.forEach((element, index) => {
            const speed = 0.1 + (index * 0.05);
            const yPos = -(scrolled * speed);
            element.style.transform = `translateY(${yPos}px)`;
        });
    }

    showToast(message, type = 'info', duration = 4000) {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = this.getToastIcon(type);
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="${icon}"></i>
                <span>${message}</span>
            </div>
        `;

        toastContainer.appendChild(toast);

        // Auto remove
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);

        // Manual close on click
        toast.addEventListener('click', () => {
            toast.style.animation = 'slideOutRight 0.3s ease-in forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        });
    }

    getToastIcon(type) {
        switch (type) {
            case 'success': return 'fas fa-check-circle';
            case 'error': return 'fas fa-exclamation-circle';
            case 'warning': return 'fas fa-exclamation-triangle';
            default: return 'fas fa-info-circle';
        }
    }

    toggleHelp() {
        const tooltip = document.querySelector('.help-tooltip');
        if (tooltip) {
            tooltip.style.opacity = tooltip.style.opacity === '1' ? '0' : '1';
            tooltip.style.visibility = tooltip.style.visibility === 'visible' ? 'hidden' : 'visible';
        }
    }

    setupSourceTracking() {
        // Track source changes for analytics
        const sourceCheckboxes = document.querySelectorAll('.source-option input[type="checkbox"]');
        sourceCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const sources = this.getSelectedSources();
                console.log('🔄 Sources updated:', sources);
                
                // Update UI feedback
                this.updateSourceFeedback(sources);
            });
        });

        // Initialize with default state
        this.updateSelectedSources();
    }

    updateSourceFeedback(sources) {
        // Visual feedback for source selection
        const sourceOptions = document.querySelectorAll('.source-option');
        
        sourceOptions.forEach(option => {
            const checkbox = option.querySelector('input[type="checkbox"]');
            const label = option.querySelector('label');
            
            if (checkbox && label) {
                if (checkbox.checked) {
                    label.classList.add('interactive-element');
                    option.classList.add('breathing');
                } else {
                    label.classList.remove('interactive-element');
                    option.classList.remove('breathing');
                }
            }
        });

        // Show enhanced feedback for orchestration mode
        if (sources.length > 1) {
            this.showToast('Orchestration mode activated - both systems will be combined!', 'info', 2000);
        }

        // Show warning if no sources selected
        if (sources.length === 0) {
            this.showToast('Please select at least one data source', 'warning', 3000);
        }
    }

    setupAccessibility() {
        // Enhanced keyboard navigation
        const focusableElements = document.querySelectorAll(
            'button, input, textarea, [tabindex]:not([tabindex="-1"])'
        );

        focusableElements.forEach(element => {
            element.classList.add('focus-ring');
            
            // Add ripple effect to buttons
            if (element.tagName === 'BUTTON') {
                element.classList.add('ripple');
            }
        });

        // Screen reader announcements
        this.setupScreenReaderSupport();
        
        // High contrast mode detection
        this.detectHighContrast();
    }

    setupScreenReaderSupport() {
        // Create announcer for screen readers
        const announcer = document.createElement('div');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.className = 'sr-only';
        announcer.style.cssText = `
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        `;
        document.body.appendChild(announcer);

        this.announcer = announcer;

        // Announce status changes
        const originalCheckSystemStatus = this.checkSystemStatus;
        this.checkSystemStatus = async function() {
            await originalCheckSystemStatus.call(this);
            const statusText = document.querySelector('.status-text')?.textContent;
            if (statusText && this.announcer) {
                this.announcer.textContent = `System status: ${statusText}`;
            }
        };
    }

    detectHighContrast() {
        // Detect high contrast mode
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            document.body.classList.add('high-contrast-mode');
            console.log('High contrast mode detected');
        }

        // Listen for changes
        window.matchMedia('(prefers-contrast: high)').addEventListener('change', (e) => {
            if (e.matches) {
                document.body.classList.add('high-contrast-mode');
            } else {
                document.body.classList.remove('high-contrast-mode');
            }
        });
    }

    // Performance monitoring
    measurePerformance(name, fn) {
        const start = performance.now();
        const result = fn();
        const end = performance.now();
        console.log(`${name} took ${end - start} milliseconds`);
        return result;
    }

    // Error handling and recovery
    handleError(error, context = '') {
        console.error(`Error in ${context}:`, error);
        
        // Log to analytics if available
        if (window.gtag) {
            window.gtag('event', 'exception', {
                description: `${context}: ${error.message}`,
                fatal: false
            });
        }

        // Show user-friendly error
        this.showToast(
            'Something went wrong. Please try again or refresh the page.',
            'error',
            5000
        );
    }

    // Cleanup and destroy
    destroy() {
        // Remove event listeners
        window.removeEventListener('resize', this.handleResize);
        window.removeEventListener('scroll', this.handleScroll);
        document.removeEventListener('mousemove', this.handleMouseMove);

        // Clear particles
        const particlesContainer = document.getElementById('particles');
        if (particlesContainer) {
            particlesContainer.innerHTML = '';
        }

        // Clear any running intervals/timeouts
        this.particleCount = 0;

        console.log('🧹 UnifiedOrchestrator cleaned up');
    }
}

// Enhanced slideOutRight animation keyframes
const slideOutRightKeyframes = `
@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(400px);
        opacity: 0;
    }
}
`;

// Add the keyframes to the document
const style = document.createElement('style');
style.textContent = slideOutRightKeyframes;
document.head.appendChild(style);

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check for modern browser features
    if (!window.fetch || !window.Promise) {
        alert('This application requires a modern browser. Please update your browser to continue.');
        return;
    }

    // Initialize the orchestrator
    window.orchestrator = new UnifiedOrchestrator();

    // Service worker registration for offline support (optional)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(err => {
            console.log('Service worker registration failed:', err);
        });
    }

    // Add loading class to body initially
    document.body.classList.add('loading');
    
    // Remove loading class after initialization
    setTimeout(() => {
        document.body.classList.remove('loading');
        document.body.classList.add('loaded');
    }, 1000);

    // Global error handler
    window.addEventListener('error', (event) => {
        if (window.orchestrator) {
            window.orchestrator.handleError(event.error, 'Global Error');
        }
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
        if (window.orchestrator) {
            window.orchestrator.handleError(event.reason, 'Unhandled Promise Rejection');
        }
    });

    console.log('🎭 CME Unified Knowledge Orchestrator ready!');
});

// Expose orchestrator to global scope for debugging
window.CMEOrchestrator = UnifiedOrchestrator;





















/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* CME Blue Gradient Colors */
    --primary-blue: #1e3a8a;
    --secondary-blue: #3b82f6;
    --light-blue: #60a5fa;
    --accent-blue: #93c5fd;
    --soft-blue: #dbeafe;
    --white: #ffffff;
    --off-white: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-300: #cbd5e1;
    --gray-400: #94a3b8;
    --gray-500: #64748b;
    --gray-600: #475569;
    --gray-700: #334155;
    --gray-800: #1e293b;
    --gray-900: #0f172a;
    
    /* Gradients */
    --primary-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    --secondary-gradient: linear-gradient(135deg, #60a5fa 0%, #93c5fd 50%, #dbeafe 100%);
    --background-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 25%, #dbeafe 50%, #ffffff 100%);
    --header-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    --card-gradient: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-blue: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    
    /* Transitions */
    --transition-fast: 0.15s ease-out;
    --transition-normal: 0.3s ease-out;
    --transition-slow: 0.5s ease-out;
    
    /* Fonts */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body {
    font-family: var(--font-family);
    background: var(--background-gradient);
    min-height: 100vh;
    overflow-x: hidden;
    color: var(--gray-800);
    line-height: 1.6;
}

/* Background Pattern */
.background-pattern {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -2;
    opacity: 0.03;
}

.cme-logo-pattern {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 80%, var(--primary-blue) 2px, transparent 2px),
        radial-gradient(circle at 80% 20%, var(--secondary-blue) 2px, transparent 2px),
        radial-gradient(circle at 40% 40%, var(--light-blue) 1px, transparent 1px);
    background-size: 100px 100px, 80px 80px, 60px 60px;
    background-repeat: repeat;
}

.gradient-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--background-gradient);
    z-index: -1;
}

/* Status Indicator */
.status-indicator {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 8px 16px;
    border-radius: 25px;
    box-shadow: var(--shadow-md);
    display: flex;
    align-items: center;
    gap: 8px;
    z-index: 1000;
    font-size: 14px;
    font-weight: 500;
    transition: var(--transition-normal);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gray-400);
    transition: var(--transition-normal);
}

.status-indicator.connected .status-dot {
    background: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}

.status-indicator.degraded .status-dot {
    background: #f59e0b;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
}

.status-indicator.error .status-dot {
    background: #ef4444;
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
}

/* Main Container */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.header {
    background: var(--header-gradient);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    color: white;
    box-shadow: var(--shadow-blue);
    position: relative;
    overflow: hidden;
}

.header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
    opacity: 0.5;
}

.header-content {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

.logo-section {
    display: flex;
    align-items: center;
    gap: 20px;
}

.cme-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 24px;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.2);
    padding: 15px 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.cme-logo i {
    font-size: 28px;
}

.title-section h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 5px;
    background: linear-gradient(45deg, #ffffff, #dbeafe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 400;
}

/* Source Selector */
.source-selector {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.selector-label {
    font-size: 14px;
    font-weight: 600;
    opacity: 0.9;
}

.source-options {
    display: flex;
    gap: 15px;
}

.source-option {
    position: relative;
}

.source-option input[type="checkbox"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}

.source-option label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    cursor: pointer;
    transition: var(--transition-normal);
    font-weight: 500;
    user-select: none;
    backdrop-filter: blur(10px);
}

.source-option input[type="checkbox"]:checked + label {
    background: rgba(255, 255, 255, 0.3);
    border-color: rgba(255, 255, 255, 0.8);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.source-option label:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-1px);
}

/* Chat Container */
.chat-container {
    flex: 1;
    background: var(--card-gradient);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    margin-bottom: 20px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.chat-header {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px 30px;
    border-bottom: 1px solid var(--gray-200);
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(10px);
}

.chat-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--gray-800);
}

.chat-title i {
    color: var(--secondary-blue);
    font-size: 1.3rem;
}

.clear-chat {
    background: none;
    border: 2px solid var(--gray-300);
    color: var(--gray-600);
    padding: 8px 16px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    transition: var(--transition-normal);
}

.clear-chat:hover {
    border-color: var(--secondary-blue);
    color: var(--secondary-blue);
    transform: translateY(-1px);
}

/* Welcome Section */
.welcome-section {
    padding: 60px 30px;
    text-align: center;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%);
}

.welcome-content {
    max-width: 600px;
    margin: 0 auto;
}

.welcome-icon {
    width: 80px;
    height: 80px;
    background: var(--primary-gradient);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 30px;
    box-shadow: var(--shadow-blue);
}

.welcome-icon i {
    font-size: 36px;
    color: white;
}

.welcome-content h2 {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gray-800);
    margin-bottom: 15px;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.welcome-content p {
    font-size: 1.1rem;
    color: var(--gray-600);
    margin-bottom: 40px;
}

.example-queries h3 {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 20px;
}

.query-examples {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.example-query {
    background: rgba(255, 255, 255, 0.7);
    border: 2px solid var(--gray-200);
    border-radius: 15px;
    padding: 20px;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: 12px;
    text-align: left;
    font-weight: 500;
    color: var(--gray-700);
    backdrop-filter: blur(10px);
}

.example-query:hover {
    border-color: var(--secondary-blue);
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

.example-query i {
    color: var(--secondary-blue);
    font-size: 18px;
    flex-shrink: 0;
}

/* Chat Messages */
.chat-messages {
    flex: 1;
    padding: 20px 30px;
    overflow-y: auto;
    max-height: 500px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.message {
    display: flex;
    gap: 15px;
    margin-bottom: 20px;
    opacity: 1;
    transform: translateY(0);
    transition: opacity 0.5s ease, transform 0.5s ease;
}

.message.user {
    flex-direction: row-reverse;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 18px;
    color: white;
    font-weight: 600;
}

.message.user .message-avatar {
    background: var(--primary-gradient);
}

.message.assistant .message-avatar {
    background: linear-gradient(135deg, #10b981, #34d399);
}

.message-content {
    flex: 1;
    max-width: 70%;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.message.user .message-content {
    text-align: right;
}

.message-bubble {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--gray-200);
    border-radius: 18px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow-sm);
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.6;
}

.message.user .message-bubble {
    background: var(--secondary-gradient);
    color: white;
    border-color: transparent;
}

.message-time {
    font-size: 12px;
    color: var(--gray-500);
    margin-top: 5px;
}

.message.user .message-time {
    color: rgba(255, 255, 255, 0.8);
}

/* Typing Indicator */
.typing-indicator {
    display: none;
    align-items: center;
    gap: 15px;
    padding: 15px 30px;
    background: rgba(59, 130, 246, 0.05);
    border-top: 1px solid var(--gray-200);
}

.typing-indicator.show {
    display: flex;
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background: var(--secondary-blue);
    border-radius: 50%;
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

.typing-text {
    color: var(--gray-600);
    font-style: italic;
}

/* Input Section */
.input-section {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.input-container {
    padding: 25px 30px;
}

.input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 15px;
    background: rgba(255, 255, 255, 0.8);
    border: 2px solid var(--gray-200);
    border-radius: 15px;
    padding: 15px 20px;
    transition: var(--transition-normal);
    backdrop-filter: blur(10px);
}

.input-wrapper:focus-within {
    border-color: var(--secondary-blue);
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

#messageInput {
    flex: 1;
    border: none;
    outline: none;
    background: none;
    font-size: 16px;
    font-family: var(--font-family);
    color: var(--gray-800);
    resize: none;
    max-height: 120px;
    min-height: 24px;
    line-height: 1.5;
}

#messageInput::placeholder {
    color: var(--gray-500);
}

.input-actions {
    display: flex;
    align-items: center;
    gap: 10px;
}

.character-count {
    font-size: 12px;
    color: var(--gray-500);
    font-weight: 500;
}

.send-button {
    width: 44px;
    height: 44px;
    background: var(--primary-gradient);
    border: none;
    border-radius: 12px;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    transition: var(--transition-normal);
    box-shadow: var(--shadow-md);
}

.send-button:disabled {
    background: var(--gray-300);
    cursor: not-allowed;
    box-shadow: none;
}

.send-button:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-blue);
}

.input-hints {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 15px;
    font-size: 14px;
    color: var(--gray-600);
}

.hint-item, .selected-sources {
    display: flex;
    align-items: center;
    gap: 8px;
}

.hint-item i, .selected-sources i {
    color: var(--secondary-blue);
}

#sourcesList {
    font-weight: 600;
    color: var(--secondary-blue);
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.loading-overlay.show {
    display: flex;
}

.loading-content {
    text-align: center;
    background: rgba(255, 255, 255, 0.9);
    padding: 40px;
    border-radius: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid var(--gray-200);
    border-top: 4px solid var(--secondary-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

.loading-text h3 {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 10px;
}

.loading-text p {
    color: var(--gray-600);
}

/* Toast Notifications */
.toast-container {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.toast {
    background: rgba(255, 255, 255, 0.95);
    border-left: 4px solid var(--secondary-blue);
    border-radius: 10px;
    padding: 15px 20px;
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(10px);
    transform: translateX(400px);
    animation: slideInRight 0.3s ease-out forwards;
    max-width: 350px;
}

.toast.error {
    border-left-color: #ef4444;
}

.toast.success {
    border-left-color: #10b981;
}

.toast.warning {
    border-left-color: #f59e0b;
}

/* Floating Help */
.floating-help {
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 50px;
    height: 50px;
    background: var(--primary-gradient);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    cursor: pointer;
    box-shadow: var(--shadow-lg);
    transition: var(--transition-normal);
    z-index: 1000;
}

.floating-help:hover {
    transform: scale(1.1);
    box-shadow: var(--shadow-blue);
}

.help-tooltip {
    position: absolute;
    bottom: 60px;
    left: 0;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: var(--shadow-xl);
    backdrop-filter: blur(20px);
    border: 1px solid var(--gray-200);
    width: 300px;
    opacity: 0;
    visibility: hidden;
    transform: translateY(10px);
    transition: var(--transition-normal);
}

.floating-help:hover .help-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.help-content h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 15px;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.help-content ul {
    list-style: none;
}

.help-content li {
    padding: 8px 0;
    color: var(--gray-700);
    font-size: 14px;
    line-height: 1.5;
}

.help-content li strong {
    color: var(--secondary-blue);
    font-weight: 600;
}

/* Particles Background */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    background: var(--secondary-blue);
    border-radius: 50%;
    opacity: 0.1;
    animation: float 6s ease-in-out infinite;
}

/* Message Content Styling */
.message-content h1, .message-content h2, .message-content h3 {
    color: var(--gray-800);
    margin: 15px 0 10px 0;
    font-weight: 600;
}

.message-content h1 { font-size: 1.5rem; }
.message-content h2 { font-size: 1.3rem; }
.message-content h3 { font-size: 1.1rem; }

.message-content p {
    margin: 10px 0;
    line-height: 1.6;
}

.message-content ul, .message-content ol {
    margin: 10px 0;
    padding-left: 20px;
}

.message-content li {
    margin: 5px 0;
    line-height: 1.5;
}

.message-content code {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 6px;
    padding: 2px 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    color: var(--primary-blue);
}

.message-content pre {
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 15px;
    overflow-x: auto;
    margin: 15px 0;
}

.message-content pre code {
    background: none;
    border: none;
    padding: 0;
    color: var(--gray-800);
}

.message-content a {
    color: var(--secondary-blue);
    text-decoration: none;
    font-weight: 500;
    border-bottom: 1px solid transparent;
    transition: var(--transition-fast);
}

.message-content a:hover {
    border-bottom-color: var(--secondary-blue);
}

.message-content blockquote {
    border-left: 4px solid var(--secondary-blue);
    padding-left: 15px;
    margin: 15px 0;
    color: var(--gray-600);
    font-style: italic;
    background: rgba(59, 130, 246, 0.05);
    padding: 15px;
    border-radius: 0 10px 10px 0;
}

.message-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.message-content th,
.message-content td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid var(--gray-200);
}

.message-content th {
    background: var(--secondary-gradient);
    color: white;
    font-weight: 600;
}

.message-content tr:hover {
    background: rgba(59, 130, 246, 0.05);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-container {
        padding: 10px;
    }
    
    .header {
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .header-content {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
    
    .logo-section {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
    
    .title-section h1 {
        font-size: 1.8rem;
    }
    
    .subtitle {
        font-size: 1rem;
    }
    
    .source-options {
        flex-direction: column;
        gap: 10px;
    }
    
    .chat-header {
        padding: 15px 20px;
        flex-direction: column;
        gap: 15px;
        align-items: flex-start;
    }
    
    .chat-messages {
        padding: 15px 20px;
    }
    
    .message-content {
        max-width: 85%;
    }
    
    .welcome-section {
        padding: 40px 20px;
    }
    
    .welcome-content h2 {
        font-size: 1.8rem;
    }
    
    .query-examples {
        grid-template-columns: 1fr;
    }
    
    .input-container {
        padding: 20px;
    }
    
    .input-hints {
        flex-direction: column;
        gap: 10px;
        align-items: flex-start;
    }
    
    .floating-help {
        bottom: 20px;
        left: 20px;
        width: 45px;
        height: 45px;
    }
    
    .help-tooltip {
        width: 280px;
    }
    
    .status-indicator {
        top: 10px;
        right: 10px;
        font-size: 12px;
        padding: 6px 12px;
    }
}

@media (max-width: 480px) {
    .header {
        border-radius: 15px;
        padding: 15px;
    }
    
    .title-section h1 {
        font-size: 1.5rem;
    }
    
    .chat-container {
        border-radius: 15px;
    }
    
    .input-section {
        border-radius: 15px;
    }
    
    .input-wrapper {
        padding: 12px 15px;
        flex-direction: column;
        gap: 10px;
    }
    
    .input-actions {
        align-self: flex-end;
    }
    
    .message-content {
        max-width: 90%;
    }
    
    .help-tooltip {
        width: 250px;
    }
}

/* Animations */
@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes typingBounce {
    0%, 80%, 100% {
        transform: scale(0);
    }
    40% {
        transform: scale(1);
    }
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

@keyframes slideInRight {
    to {
        transform: translateX(0);
    }
}

@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-20px);
    }
}

/* Utility Classes */
.hidden {
    display: none !important;
}

.fade-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

.blur-background {
    backdrop-filter: blur(10px);
}

/* Scrollbar Styling */
.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: var(--gray-100);
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: var(--gray-300);
    border-radius: 3px;
    transition: var(--transition-normal);
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: var(--gray-400);
}

/* Focus Styles */
button:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--secondary-blue);
    outline-offset: 2px;
}

/* Selection Styles */
::selection {
    background: rgba(59, 130, 246, 0.2);
    color: var(--gray-800);
}

/* Enhanced styles for orchestrator responses */
.confluence-section {
    border-left: 4px solid #60a5fa;
    background: linear-gradient(135deg, rgba(96, 165, 250, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%);
    padding: 15px;
    margin: 15px 0;
    border-radius: 0 10px 10px 0;
}

.jira-section {
    border-left: 4px solid #10b981;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(52, 211, 153, 0.05) 100%);
    padding: 15px;
    margin: 15px 0;
    border-radius: 0 10px 10px 0;
}

.insights-section {
    border-left: 4px solid #f59e0b;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(251, 191, 36, 0.05) 100%);
    padding: 15px;
    margin: 15px 0;
    border-radius: 0 10px 10px 0;
}

/* Enhanced link styles for different sources */
.jira-link {
    color: #10b981;
    font-weight: 600;
    text-decoration: none;
    border-bottom: 2px solid transparent;
    transition: var(--transition-normal);
    padding: 2px 4px;
    border-radius: 4px;
}

.jira-link:hover {
    background: rgba(16, 185, 129, 0.1);
    border-bottom-color: #10b981;
    transform: translateY(-1px);
}

.confluence-link {
    color: #3b82f6;
    font-weight: 600;
    text-decoration: none;
    border-bottom: 2px solid transparent;
    transition: var(--transition-normal);
    padding: 2px 4px;
    border-radius: 4px;
}

.confluence-link:hover {
    background: rgba(59, 130, 246, 0.1);
    border-bottom-color: #3b82f6;
    transform: translateY(-1px);
}
    .background-pattern,
    .status-indicator,
    .floating-help,
    .loading-overlay,
    .toast-container,
    .particles {
        display: none !important;
    }
    
    .main-container {
        max-width: none;
        padding: 0;
    }
    
    .header,
    .chat-container,
    .input-section {
        box-shadow: none;
        border: 1px solid var(--gray-300);
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    :root {
        --gray-100: #f0f0f0;
        --gray-200: #e0e0e0;
        --gray-300: #c0c0c0;
        --gray-400: #a0a0a0;
        --gray-500: #808080;
        --gray-600: #606060;
        --gray-700: #404040;
        --gray-800: #202020;
        --gray-900: #000000;
    }
    
    .message-bubble {
        border-width: 2px;
    }
    
    .source-option label {
        border-width: 3px;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    
    .particles {
        display: none;
    }
}

/* Dark Mode Support (for future enhancement) */
@media (prefers-color-scheme: dark) {
    /* Dark mode styles can be added here in the future */
}
