"""
Edison Scientific literature search provider (optional).
Uses PaperQA3-backed agents to retrieve cited, high-accuracy scientific answers.
Requires: pip install edison-client
"""

try:
    from edison_client import EdisonClient, JobNames
    _EDISON_AVAILABLE = True
except ImportError:
    _EDISON_AVAILABLE = False

_JOB_MAP = {
    "literature": "LITERATURE",
    "literature_high": "LITERATURE_HIGH",
    "precedent": "PRECEDENT",
}


class EdisonLiteratureProvider:
    """
    Queries Edison Scientific's literature search agents.
    Returns a (answer, formatted_answer, has_successful_answer) tuple.
    formatted_answer includes real cited references from PaperQA3.
    """

    def __init__(self, api_key: str, job_type: str = "literature"):
        if not _EDISON_AVAILABLE:
            raise ImportError(
                "edison-client not installed. Run: pip install 'alchemist-nrel[llm]'"
            )
        self.client = EdisonClient(api_key=api_key)
        attr = _JOB_MAP.get(job_type, "LITERATURE")
        self.job_name = getattr(JobNames, attr, JobNames.LITERATURE)

    async def search_literature(self, query: str) -> tuple[str, str, bool]:
        """
        Submit a literature search task and await the result.

        Returns:
            (answer, formatted_answer, has_successful_answer)
            formatted_answer contains cited references suitable for the sources field.
        """
        task_data = {"name": self.job_name, "query": query}
        result = await self.client.arun_tasks_until_done(task_data)
        return result.answer, result.formatted_answer, result.has_successful_answer


def is_available() -> bool:
    return _EDISON_AVAILABLE
