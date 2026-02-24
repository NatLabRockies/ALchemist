"""
Edison Scientific literature search provider (optional).
Uses PaperQA3-backed agents to retrieve cited, high-accuracy scientific answers.
Requires: pip install edison-client

Async pattern used:
  1. acreate_task()  → task_id  (returns immediately; trajectory URL can be shown to user)
  2. aget_task(id)   → status / result  (poll every N seconds until status == 'success')
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

TRAJECTORY_BASE = "https://platform.edisonscientific.com/trajectories"


class EdisonLiteratureProvider:
    """
    Queries Edison Scientific's literature search agents.

    Preferred async usage (non-blocking with keepalive support):
        task_id = await provider.start_search(query)
        url = provider.get_trajectory_url(task_id)
        while True:
            result = await provider.poll_task(task_id)
            if result.status == "success":
                break

    Legacy blocking usage (kept for convenience):
        answer, formatted_answer, has_successful_answer = await provider.search_literature(query)
    """

    def __init__(self, api_key: str | None = None, job_type: str = "literature"):
        if not _EDISON_AVAILABLE:
            raise ImportError(
                "edison-client not installed. Run: pip install 'alchemist-nrel[llm]'"
            )
        # api_key=None → SDK uses EDISON_API_KEY env var natively
        self.client = EdisonClient(api_key=api_key)
        attr = _JOB_MAP.get(job_type, "LITERATURE")
        self.job_name = getattr(JobNames, attr, JobNames.LITERATURE)

    # ------------------------------------------------------------------
    # Preferred poll-based interface
    # ------------------------------------------------------------------

    async def start_search(self, query: str) -> str:
        """
        Submit a literature search task and return the task_id immediately.

        The caller can show the trajectory URL right away and then poll for
        completion via poll_task().
        """
        task_data = {"name": self.job_name, "query": query}
        task_id = await self.client.acreate_task(task_data)
        return str(task_id)

    async def poll_task(self, task_id: str):
        """
        Single poll of task status.

        Returns a PQATaskResponse (or TaskResponse).  When complete:
            result.status == "success"
            result.answer           — full answer text
            result.formatted_answer — answer with cited references
            result.has_successful_answer — bool
        """
        return await self.client.aget_task(task_id)

    def get_trajectory_url(self, task_id: str) -> str:
        """Return the Edison platform trajectory URL for the given task_id."""
        return f"{TRAJECTORY_BASE}/{task_id}"

    # ------------------------------------------------------------------
    # Legacy blocking interface (kept for convenience / backwards compat)
    # ------------------------------------------------------------------

    async def search_literature(self, query: str) -> tuple[str, str, bool]:
        """
        Submit a literature search task and block until done.

        Returns:
            (answer, formatted_answer, has_successful_answer)
            formatted_answer contains cited references suitable for the sources field.

        Note: arun_tasks_until_done returns a list even for a single task;
        we defensively index [0] to get the PQATaskResponse.
        """
        task_data = {"name": self.job_name, "query": query}
        results = await self.client.arun_tasks_until_done(task_data)
        # arun_tasks_until_done returns a list for async calls
        r = results[0] if isinstance(results, list) else results
        return r.answer, r.formatted_answer, r.has_successful_answer


def is_available() -> bool:
    return _EDISON_AVAILABLE
