# Import asyncio for asynchronous programming, allowing concurrent operations
import asyncio
# Import logging to enable logging of information, warnings, and errors
import logging
# Import os to access environment variables and interact with the operating system
import os
# Import ABC and abstractmethod to define abstract base classes and enforce method implementation in subclasses
from abc import ABC, abstractmethod
# Import dataclass to easily create classes for storing data with less boilerplate
from dataclasses import dataclass
#from dataclassses import dataclass
# Import Dict type hint for type annotations of dictionaries
#from typing import Dict ,Optional ,Any
from typing import Dict, Optional, Any

# Import load_dotenv to load environment variables from a .env file
from dotenv import load_dotenv # for envorment variables from .env file
# Import GenerativeModel from Google Generative AI for Gemini LLM integration
from google.generativeai import GenerativeModel
# Import END, START, StateGraph from langgraph.graph to build and manage workflow graphs
from langgraph.graph import END, START, StateGraph
# Import AsyncOpenAI for asynchronous OpenAI API calls
from openai import AsyncOpenAI
# Import TavilyClient to interact with the Tavily search/enrichment API
from tavily import TavilyClient

# Configure logging to show INFO level logs and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#logging.basicCongfig(level=logging.INFO)
#logger = logging.getLogger(_name_)
# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# Abstract base class for LLM providers, enforcing the generate method
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        # Abstract method to be implemented by subclasses for generating text from a prompt
        pass

#class LLMPROVIDER(ABC):
#@abstractmethod
#async def generate(self,prompt:str)->str:
#pass

# Implementation of LLMProvider for OpenAI's GPT models
class OpenAIProvider(LLMProvider):
    def __init__(self, client, model="gpt-4.1-2025-04-14"):
        self.client = client  # AsyncOpenAI client instance
        self.model = model    # Model name (default: GPT-4.1)

    async def generate(self, prompt: str) -> str:
        # Asynchronously call OpenAI's chat completion API with the prompt
        response = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        # Return the generated message content, stripped of whitespace
        return response.choices[0].message.content.strip()

        #class openairesonse(LLMprovider):
        #def __init__(self,model):
        #self.model=model
        
  


# Implementation of LLMProvider for Google's Gemini model
class GeminiProvider(LLMProvider):
    def __init__(self, model):
        self.model = model  # GenerativeModel instance

    async def generate(self, prompt: str) -> str:
        # Run Gemini's generate_content in a thread to avoid blocking
        response = await asyncio.to_thread(lambda: self.model.generate_content(prompt))
        # Return the generated text, stripped of whitespace
        return response.text.strip()


        #class Geminiresponse(LLMProvider):
        #def __init__(self,model):
        #self.model=model

        #async def generate(self,prompt:str)->str:
       # response = await asyncio.to_thread(lambda:self.model.generate_content(prompt))
# Dataclass to hold all relevant information for a single enrichment operation
@dataclass
class EnrichmentContext:
    column_name: str                # e.g., "CEO" (the header/question for this column)
    target_value: str               # e.g., "Amazon" (the main entity for this row)
    context_values: Dict[str, str]  # Additional context (e.g., {"Industry": "E-commerce"})
    
    # NEW FIELDS for agent chaining and advanced planning
    input_source_type: Optional[str] = None   # 'ENTITY', 'URL', or 'TEXT_FROM_COLUMN' (where the input comes from)
    input_data: Optional[str] = None          # The actual data to use (e.g., "plaid.com" or text from another cell)
    plan: Optional[dict] = None               # Output of the Planner node: structured plan for what to do next
    custom_prompt: Optional[str] = None       # Custom prompt/instruction for AI agent
    
    # Existing/legacy fields for enrichment
    search_result: Optional[dict] = None      # Results from Tavily search or other sources
    answer: Optional[str] = None              # Final extracted answer

# Pipeline class to orchestrate the enrichment process using Tavily and LLM
class EnrichmentPipeline:
    def __init__(self, tavily_client, llm_provider: LLMProvider):
        self.tavily = tavily_client      # TavilyClient instance
        self.llm = llm_provider          # LLMProvider instance (OpenAI or Gemini)
#async def search_tavily(self,state:EnrichmentContext):
    async def search_tavily(self, state: EnrichmentContext):
        """Run Tavily search in a separate thread"""
        try:
            # DEBUG: Log all the state information
            logger.info(f"DEBUG search_tavily - column_name: {state.column_name}")
            logger.info(f"DEBUG search_tavily - target_value: {state.target_value}")
            logger.info(f"DEBUG search_tavily - input_source_type: {state.input_source_type}")
            logger.info(f"DEBUG search_tavily - input_data: {state.input_data}")
            logger.info(f"DEBUG search_tavily - custom_prompt: {state.custom_prompt}")
            logger.info(f"DEBUG search_tavily - plan: {state.plan}")
            
            # Use a custom search_query from the plan if available, otherwise fall back to improved query construction
            query = None
            if state.plan and isinstance(state.plan, dict):
                query = state.plan.get("search_query")
            if not query:
                # Get current year for time-sensitive searches
                import datetime
                current_year = datetime.datetime.now().year
                
                # Normalize company names for better search results
                def normalize_company_name(company_name):
                    """Normalize company names for better search results"""
                    if not company_name:
                        return company_name
                    
                    # Convert to lowercase for comparison
                    name_lower = company_name.lower().strip()
                    
                    # Handle common variations
                    if name_lower in ['linked in', 'linkedin inc', 'linkedin corp']:
                        return 'LinkedIn'
                    elif name_lower in ['amazon inc', 'amazon.com', 'amazon com']:
                        return 'Amazon'
                    elif name_lower in ['microsoft corp', 'microsoft inc', 'msft']:
                        return 'Microsoft'
                    elif name_lower in ['google inc', 'alphabet inc', 'alphabet']:
                        return 'Google'
                    elif name_lower in ['apple inc', 'apple computer']:
                        return 'Apple'
                    elif name_lower in ['meta', 'facebook inc', 'meta platforms']:
                        return 'Meta'
                    
                    # Return original with proper capitalization
                    return company_name.strip()
                
                # Improved fallback query construction using custom prompt and input data
                if state.custom_prompt and state.input_data:
                    # Use custom prompt with input data for AI agent queries
                    normalized_company = normalize_company_name(state.input_data)
                    query = f"{state.custom_prompt} {normalized_company} current {current_year}"
                elif state.custom_prompt:
                    # Use custom prompt with target value
                    normalized_company = normalize_company_name(state.target_value)
                    query = f"{state.custom_prompt} {normalized_company} current {current_year}"
                else:
                    # Default fallback with current year for time-sensitive information
                    normalized_company = normalize_company_name(state.target_value)
                    query = f"current {state.column_name} of {normalized_company} {current_year}"
            logger.info(f"FINAL SEARCH QUERY: {query}")
            print(f"🔍 FINAL SEARCH QUERY: {query}")
            print(f"🎯 Query components - custom_prompt: {state.custom_prompt}, input_data: {state.input_data}, target_value: {state.target_value}")
            # Run the Tavily search in a thread to avoid blocking
            result = await asyncio.to_thread(
                lambda: self.tavily.search(
                    query=query, 
                    auto_parameters=True, 
                    search_depth="advanced", 
                    max_results=5, 
                    include_raw_content=True,
                    days=365  # Search within the last year for more recent information
                )
            )
            print(result["auto_parameters"])
            logger.info(f"Tavily search result: {result}")
            # Return the search result to be added to the context
            return {"search_result": result}
            l
        except Exception as e:
            logger.error(f"Error in search_tavily: {str(e)}")
            raise

    async def extract_minimal_answer( self, state: EnrichmentContext ) -> dict:
        """Use LLM to extract a minimal answer from Tavily's or scraped results."""
        content = ""
        # Safely get results from search_result
        results = []
        if state.search_result and isinstance(state.search_result, dict):
            results = state.search_result.get("results", [])
        # Collect all non-None raw_content fields or use string results
        result_contents = []
        for result in results:
            if isinstance(result, dict) and result.get("raw_content") is not None:
                result_contents.append(result["raw_content"])
            elif isinstance(result, str):
                result_contents.append(result)
        # Join all result contents with separators
        content = "\n\n---\n\n".join(result_contents)
        print(f"Content: {content}")
        try:
            # Get current year for time-sensitive extraction
            import datetime
            current_year = datetime.datetime.now().year
            
            # Build the prompt for the LLM, instructing it to extract only the direct answer
            extraction_instructions = ""
            if state.plan and isinstance(state.plan, dict):
                extraction_instructions = state.plan.get("extraction_instructions", "")
            prompt = f"""
                    Extract the CURRENT {state.column_name} of {state.target_value} from this search result:

                    {content}

                    Rules:
                    1. PRIORITIZE the most recent information (look for dates from {current_year} or late 2023)
                    2. If there are multiple entries, choose the most recent one
                    3. For leadership positions (CEO, President, etc.), always look for "current" or most recent information
                    4. Provide ONLY the direct answer - no explanations
                    5. Be concise
                    6. If not found, respond \"Information not found\"
                    7. No citations or references
                    
                    Example:
                    Search result: "Amazon's CEO is Andy Jassy as of 2021. Previously, Jeff Bezos was CEO until July 2021."
                    Direct Answer: Andy Jassy
                    
                    {extraction_instructions}
                    
                    Current {state.column_name} of {state.target_value}:
                    """
            logger.info(f"Extracting answer for {state.target_value}")

            # Use the LLM to generate the answer
            answer = await self.llm.generate(prompt)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Extracted answer: {answer}")
            # Return the answer to be added to the context
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error in extract_minimal_answer: {str(e)}")
            return {"answer": "Information not found"}

    async def generate_plan(self, state: EnrichmentContext) -> dict:
        """
        Planner node: Uses the LLM to analyze the user's request and input data, and produces a structured plan.
        The plan determines the next action (e.g., analyze a URL, search the web, analyze text, etc.).
        """
        # Get custom prompt and context for better planning
        custom_prompt = state.custom_prompt
        context_info = ""
        if state.context_values:
            context_info = f"Additional context: {state.context_values}"
        
        # Get current year for time-sensitive searches
        import datetime
        current_year = datetime.datetime.now().year
        
        # Compose a prompt for the LLM to analyze the user's intent and input
        prompt = f"""
        You are an AI research agent planner. Your job is to analyze the user's enrichment request and input data, and return a JSON plan for the next step.

        User's column/question: {state.column_name}
        Input source type: {state.input_source_type}
        Input data: {state.input_data}
        Custom prompt/instruction: {custom_prompt or "None"}
        {context_info}

        Instructions:
        - If the input source type is 'URL', and the question is about analyzing or extracting from that URL, set action to 'analyze_url' and include 'source_url' and 'extraction_instructions'.
        - If the input source type is 'TEXT_FROM_COLUMN', and the question is about analyzing or extracting from that text, set action to 'analyze_text' and include 'source_text' and 'extraction_instructions'.
        - If the input source type is 'ENTITY' or the question is general, set action to 'search_web' and include 'search_query' and 'extraction_instructions'.
        - When creating search_query, ALWAYS include "current {current_year}" or "latest" for time-sensitive information like CEO, leadership, current status, etc.
        - For leadership/CEO queries, include both "current" and "{current_year}" in the search query.
        - If custom prompt is provided, incorporate it into the search query and extraction instructions.
        - Always include a concise 'extraction_instructions' field that tells the next node what to extract or analyze.
        - Respond ONLY with a valid JSON object, no explanation.

        Example outputs:
        {{
            "action": "analyze_url",
            "source_url": "https://plaid.com",
            "extraction_instructions": "Analyze the text of the website to determine its target audience."
        }}
        or
        {{
            "action": "search_web",
            "search_query": "current CEO of Amazon {current_year} latest",
            "extraction_instructions": "Extract the name of the current CEO as of {current_year}."
        }}
        or
        {{
            "action": "analyze_text",
            "source_text": "Developers, financial institutions, and fintech startups...",
            "extraction_instructions": "List three potential marketing slogans for this audience."
        }}

        Now, generate the plan for this request:
        """
        try:
            # Use the LLM to generate the plan as a JSON string
            #plan_json = await self.llm.generate(prompt)
            plan_json = await self.llm.generate(prompt)
            import json
            plan = json.loads(plan_json)
            # Attach the plan to the state
            return {"plan": plan}
        except Exception as e:
            logger.error(f"Error in generate_plan: {str(e)}")
            # Fallback: default to web search with current year
            fallback_query = f"current {state.column_name} of {state.target_value} {current_year}"
            return {"plan": {"action": "search_web", "search_query": fallback_query, "extraction_instructions": f"Extract the current answer to: {state.column_name}"}}

    async def scrape_website(self, state: EnrichmentContext):
        """
        Scrapes the website at the given URL using Tavily and returns the raw content.
        """
        # Get the URL from the plan or input_data
        url = None
        if state.plan and isinstance(state.plan, dict):
            url = state.plan.get("source_url")
        if not url and state.input_data:
            url = state.input_data

        if not url:
            logger.error("No URL provided for website scraping.")
            return {"search_result": {"results": ["[ERROR: No URL provided]"]}}

        try:
            logger.info(f"Scraping website: {url}")
            # Tavily can take a URL as a query and return raw content
            result = await asyncio.to_thread(
                lambda: self.tavily.search(
                    query=url,
                    auto_parameters=True,
                    search_depth="advanced",
                    max_results=1,
                    include_raw_content=True
                )
            )
            logger.info(f"Tavily scrape result: {result}")
            return {"search_result": result}
        except Exception as e:
            logger.error(f"Error in scrape_website: {str(e)}")
            return {"search_result": {"results": [f"[ERROR: {str(e)}]"]}}

    async def analyze_text_node(self, state: EnrichmentContext):
        """
        Analyzes provided text using the LLM according to extraction instructions in the plan.
        """
        # Get the text to analyze
        text = None
        if state.plan and isinstance(state.plan, dict):
            text = state.plan.get("source_text")
        if not text and state.input_data:
            text = state.input_data
        if not text:
            logger.error("No text provided for analyze_text_node.")
            return {"search_result": {"results": ["[ERROR: No text provided]"]}}
        # Get extraction instructions
        extraction_instructions = ""
        if state.plan and isinstance(state.plan, dict):
            extraction_instructions = state.plan.get("extraction_instructions", "")
        # Compose the prompt for the LLM
        prompt = f"""
        Analyze the following text according to these instructions:
        {extraction_instructions}

        Text:
        {text}
        """
        try:
            logger.info(f"Analyzing text with instructions: {extraction_instructions}")
            answer = await self.llm.generate(prompt)
            logger.info(f"LLM analyze_text_node result: {answer}")
            return {"search_result": {"results": [answer]}}
        except Exception as e:
            logger.error(f"Error in analyze_text_node: {str(e)}")
            return {"search_result": {"results": [f"[ERROR: {str(e)}]"]}}

    def build_graph(self):
        """Build and compile the conditional agent graph with planner, router, and new nodes."""
        graph = StateGraph(EnrichmentContext)
        # Add the planner node
        graph.add_node("planner", self.generate_plan)
        # Add the main action nodes
        graph.add_node("search_tavily_node", self.search_tavily)
        graph.add_node("scrape_website_node", self.scrape_website if hasattr(self, 'scrape_website') else lambda state: {"search_result": {"results": ["[SCRAPE PLACEHOLDER]"]}})
        graph.add_node("analyze_text_node", self.analyze_text_node)
        graph.add_node("extract", self.extract_minimal_answer)
        # Edges: Start → planner
        graph.add_edge(START, "planner")
        # Conditional routing after planner
        graph.add_conditional_edges("planner", self.route_after_planning, {
            "scrape_website_node": "scrape_website_node",
            "search_tavily_node": "search_tavily_node",
            "analyze_text_node": "analyze_text_node"
        })
        # All action nodes → extract
        graph.add_edge("scrape_website_node", "extract")
        graph.add_edge("search_tavily_node", "extract")
        graph.add_edge("analyze_text_node", "extract")
        # Extract → End
        graph.add_edge("extract", END)
        compiled_graph = graph.compile()
        return compiled_graph

    def route_after_planning(self, state: EnrichmentContext):
        """
        Router function: Decides the next node after planning, based on the plan's action.
        Safely handles missing or malformed plan.
        """
        plan = state.plan
        if not isinstance(plan, dict):
            return "search_tavily_node"  # Default fallback
        action = plan.get("action", "search_web")
        if action == "analyze_url":
            return "scrape_website_node"
        elif action == "analyze_text":
            return "analyze_text_node"
        else:
            return "search_tavily_node"  # Default to web search

# Asynchronous helper function to run the enrichment pipeline for a single cell
async def enrich_cell_with_graph(
    column_name: str,
    target_value: str,
    context_values: Dict[str, str],
    tavily_client,
    llm_provider: LLMProvider,
    input_source_type: Optional[str] = "ENTITY",
    input_data: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> Dict:
    """Helper function to enrich a single cell using langgraph."""
    try:
        logger.info(f"Starting enrich_cell_with_graph for {target_value}")
        # Create the pipeline instance
        pipeline = EnrichmentPipeline(tavily_client, llm_provider)
        # Build the initial context for this cell
        initial_context = EnrichmentContext(
            column_name=column_name,
            target_value=target_value,
            context_values=context_values,
            input_source_type=input_source_type,
            input_data=input_data or target_value,
            custom_prompt=custom_prompt,
            search_result=None,
            answer=None
        )
        # Build and run the graph asynchronously
        graph = pipeline.build_graph()
        result = await graph.ainvoke(initial_context)
        logger.info(f"Completed enrich_cell_with_graph for {target_value}")
        return result #, result['urls']
    except Exception as e:
        logger.error(f"Error in enrich_cell_with_graph: {str(e)}")
        return {"answer": "Error during enrichment", "search_result": None}

# Example usage block for running the enrichment pipeline directly
if __name__ == "__main__":
    # Example context for enrichment (e.g., finding CEO of Amazon)
    context = EnrichmentContext(
        column_name="CEO",
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994",
            "Location": "Seattle, WA",
        },
    )

    # Initialize Tavily client with API key from environment
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # Initialize OpenAI client with API key from environment
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Initialize Gemini model (Google Generative AI)
    gemini_model = GenerativeModel(model_name="gemini-1.5-flash")
    #gemini_model = GenerativeModel(api_key=os.getenv("GEMINI_API_KEY"))
    # Example: Create OpenAI provider and pipeline
    openai_provider = OpenAIProvider(openai_client)
    pipeline_openai = EnrichmentPipeline(tavily_client, openai_provider)

    # Example: Create Gemini provider and pipeline
    gemini_provider = GeminiProvider(gemini_model)
    pipeline_gemini = EnrichmentPipeline(tavily_client, gemini_provider)

    # Build and run the graph using Gemini provider
    graph = pipeline_gemini.build_graph()
    initial_context = EnrichmentContext(
        column_name="CEO", 
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994", 
            "Location": "Seattle, WA",
        },
        search_result=None,
        answer=None
    )
    result = asyncio.run(graph.ainvoke(initial_context))

    # Or using the helper function for a single cell
    result_helper = asyncio.run(enrich_cell_with_graph(
        column_name="CEO",
        target_value="Amazon",
        context_values={
            "Industry": "E-commerce",
            "Founded": "1994",
            "Location": "Seattle, WA",
        },
        tavily_client=tavily_client,
        llm_provider=gemini_provider
    ))
