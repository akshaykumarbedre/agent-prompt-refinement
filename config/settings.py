"""Configuration settings for the prompt refinement workflow."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.TEST_CASES_DIR = self.DATA_DIR / "test_cases"
        self.PROMPTS_DIR = self.DATA_DIR / "prompts" / "versions"
        self.RESULTS_DIR = self.DATA_DIR / "results"
        
        # Create directories if they don't exist
        self.TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)
        self.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # OpenAI settings for GPT-4 Judge
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")  # Use gpt-4o-mini for JSON mode support
        
        # Agent API settings
        self.AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000/api/agent/invoke")
        self.AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
        
        # Prompt Management API settings
        self.PROMPT_API_GET_URL = os.getenv("PROMPT_API_GET_URL", "http://localhost:8000/api/prompt/get")
        self.PROMPT_API_UPDATE_URL = os.getenv("PROMPT_API_UPDATE_URL", "http://localhost:8000/api/prompt/update")
        self.PROMPT_API_KEY = os.getenv("PROMPT_API_KEY", "")
        
        # Workflow settings
        self.MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
        self.CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.95"))
        self.IMPROVEMENT_MIN = float(os.getenv("IMPROVEMENT_MIN", "0.02"))
        self.PATIENCE = int(os.getenv("PATIENCE", "3"))
        self.PASS_SCORE_THRESHOLD = float(os.getenv("PASS_SCORE_THRESHOLD", "7"))
    
    def validate(self) -> list[str]:
        """Validate required settings and return list of missing ones."""
        missing = []
        if not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not self.AGENT_API_URL:
            missing.append("AGENT_API_URL")
        if not self.PROMPT_API_GET_URL:
            missing.append("PROMPT_API_GET_URL")
        if not self.PROMPT_API_UPDATE_URL:
            missing.append("PROMPT_API_UPDATE_URL")
        return missing


# Global settings instance
settings = Settings()
