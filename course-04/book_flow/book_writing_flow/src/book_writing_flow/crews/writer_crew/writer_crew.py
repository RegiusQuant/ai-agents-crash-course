from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from pydantic import BaseModel


class Chapter(BaseModel):
    """Chapter of the book"""
    title: str
    content: str


@CrewBase
class ChapterWriterCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_researcher"],
            tools=[SerperDevTool()]
        )

    @task
    def research_topic(self) -> Task:
        return Task(config=self.tasks_config["research_topic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"])

    @task
    def write_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["write_chapter"],
            output_pydantic=Chapter
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )