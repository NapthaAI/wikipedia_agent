import asyncio
import logging
import os
from dotenv import load_dotenv
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.client.node import Node
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from wikipedia_agent.schemas import InputSchema, SystemPromptSchema

load_dotenv()

logger = logging.getLogger(__name__)

class WikipediaAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.kb = KnowledgeBase(kb_deployment=self.deployment.kb_deployments[0])
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.node = Node(self.deployment.node)

    async def answer_question_from_content(self, tool_input_data):
        page = await self.kb.get_kb(column_name='title', column_value=tool_input_data["query"])
        if not page:
            return {"error": "Page not found"}
        logger.info(f"Page content: {page}")

        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": f"The user asked: {tool_input_data['question']}. The wikipedia page content is: {page}\n\nAnswer the question based on the page content."}
        ]
        logger.info(f"Messages: {messages}")

        llm_response = await self.node.run_inference({"model": self.deployment.config.llm_config.model,
                                                      "messages": messages})
        return llm_response
    

async def run(module_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {module_run.inputs.tool_input_data}")
    wikipedia_agent = WikipediaAgent(module_run.deployment)
    method = getattr(wikipedia_agent, module_run.inputs.tool_name, None)
    answer = await method(module_run.inputs.tool_input_data)
    return answer


if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = setup_module_deployment("agent", "wikipedia_agent/configs/deployment.json", node_url = os.getenv("NODE_URL"))

    query = "Elon Musk"
    question = "What is Elon Musk's net worth?"

    input_params = InputSchema(
        tool_name="answer_question_from_content",
        tool_input_data={"query": query, "question": question},
    )

    module_run = AgentRunInput(
        inputs=input_params,
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    response = asyncio.run(run(module_run))

    print(response)
