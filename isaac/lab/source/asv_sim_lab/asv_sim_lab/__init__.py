"""Public Isaac Lab tasks for the ASV simulation platform."""

from .tasks.t1_s2 import TASK_ID, register_task

register_task()

__all__ = ["TASK_ID", "register_task"]
