"""Test case manager for managing question/expected-output pairs."""
import json
from pathlib import Path
from typing import Optional
from rich.console import Console

from .models import TestCase, TestDataset
from config import settings

console = Console()


class TestCaseManager:
    """Manages test cases with JSONL storage."""
    
    def __init__(self):
        self.test_cases_dir = settings.TEST_CASES_DIR
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        self.default_dataset_path = self.test_cases_dir / "dataset.jsonl"
    
    def _get_dataset_path(self, dataset_name: Optional[str] = None) -> Path:
        """Get path for a dataset."""
        if dataset_name:
            return self.test_cases_dir / f"{dataset_name}.jsonl"
        return self.default_dataset_path
    
    def add_test_case(
        self,
        input_text: str,
        expected_output: str,
        tags: list[str] = None,
        difficulty: str = "medium",
        dataset_name: Optional[str] = None,
    ) -> TestCase:
        """Add a new test case to the dataset."""
        test_case = TestCase(
            input_text=input_text,
            expected_output=expected_output,
            tags=tags or [],
            difficulty=difficulty,
        )
        
        path = self._get_dataset_path(dataset_name)
        with open(path, "a") as f:
            f.write(json.dumps({"item": test_case.model_dump()}) + "\n")
        
        return test_case
    
    def load_test_cases(self, dataset_name: Optional[str] = None) -> list[TestCase]:
        """Load all test cases from a dataset."""
        path = self._get_dataset_path(dataset_name)
        if not path.exists():
            return []
        
        cases = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cases.append(TestCase(**data.get("item", data)))
        return cases
    
    def get_test_case(self, test_id: str, dataset_name: Optional[str] = None) -> Optional[TestCase]:
        """Get a specific test case by ID."""
        cases = self.load_test_cases(dataset_name)
        for case in cases:
            if case.id == test_id:
                return case
        return None
    
    def delete_test_case(self, test_id: str, dataset_name: Optional[str] = None) -> bool:
        """Delete a test case by ID."""
        path = self._get_dataset_path(dataset_name)
        cases = self.load_test_cases(dataset_name)
        original_count = len(cases)
        cases = [c for c in cases if c.id != test_id]
        
        if len(cases) == original_count:
            return False
        
        # Rewrite the file
        with open(path, "w") as f:
            for case in cases:
                f.write(json.dumps({"item": case.model_dump()}) + "\n")
        
        return True
    
    def import_jsonl(self, file_path: str, dataset_name: Optional[str] = None) -> int:
        """Import test cases from a JSONL file."""
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        imported = 0
        with open(source_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    item = data.get("item", data)
                    self.add_test_case(
                        input_text=item.get("input_text", item.get("question", "")),
                        expected_output=item.get("expected_output", item.get("answer", "")),
                        tags=item.get("tags", []),
                        difficulty=item.get("difficulty", "medium"),
                        dataset_name=dataset_name,
                    )
                    imported += 1
        
        return imported
    
    def export_jsonl(self, output_path: str, dataset_name: Optional[str] = None) -> int:
        """Export test cases to a JSONL file."""
        cases = self.load_test_cases(dataset_name)
        with open(output_path, "w") as f:
            for case in cases:
                f.write(json.dumps({"item": case.model_dump()}) + "\n")
        return len(cases)
    
    def get_dataset_stats(self, dataset_name: Optional[str] = None) -> dict:
        """Get statistics about the dataset."""
        cases = self.load_test_cases(dataset_name)
        
        difficulty_counts = {}
        tag_counts = {}
        
        for case in cases:
            # Count difficulties
            difficulty_counts[case.difficulty] = difficulty_counts.get(case.difficulty, 0) + 1
            
            # Count tags
            for tag in case.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_cases": len(cases),
            "by_difficulty": difficulty_counts,
            "by_tag": tag_counts,
        }
    
    def filter_cases(
        self,
        dataset_name: Optional[str] = None,
        tags: list[str] = None,
        difficulty: str = None,
    ) -> list[TestCase]:
        """Filter test cases by tags and/or difficulty."""
        cases = self.load_test_cases(dataset_name)
        
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]
        
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]
        
        return cases
