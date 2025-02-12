import logging
import os
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.inference import InferenceClient
from naptha_sdk.schemas import AgentDeployment, AgentRunInput, KBRunInput
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from wikipedia_agent.schemas import InputSchema, SystemPromptSchema

load_dotenv()

logger = logging.getLogger(__name__)

class WikipediaAgent:
    async def create(self, deployment: AgentDeployment, *args, **kwargs):
        self.deployment = deployment
        self.wikipedia_kb = KnowledgeBase()
        kb_deployment = await self.wikipedia_kb.create(deployment=self.deployment.kb_deployments[0])
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.inference_client = InferenceClient(self.deployment.node)

    async def run(self, module_run: AgentRunInput, *args, **kwargs):
        logger.info("Checking if knowledge base exists")
        kb_run_input = KBRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "init", "func_input_data": None},
            deployment=self.deployment.kb_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY_FULL_PATH")))
        )
        result = await self.wikipedia_kb.run(kb_run_input)
        logger.info(f"KB run result: {result}")

        # Now run the query
        kb_run_input = KBRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "run_query", "func_input_data": {"query": module_run.inputs.query}},
            deployment=self.deployment.kb_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY_FULL_PATH")))
        )

        page = await self.wikipedia_kb.run(kb_run_input)
        
        if not page:
            return {"error": "Page not found"}
        logger.info(f"Page content: {page}")

        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": f"The user asked: {module_run.inputs.question}. The wikipedia page content is: {page}\n\nAnswer the question based on the page content."}
        ]
        logger.info(f"Messages: {messages}")

        llm_response = await self.inference_client.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens})
        return llm_response.choices[0].message.content
    

async def run(module_run: Dict, *args, **kwargs):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    wikipedia_agent = WikipediaAgent()
    await wikipedia_agent.create(module_run.deployment)
    agent_response = await wikipedia_agent.run(module_run)
    return agent_response


if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("agent", "wikipedia_agent/configs/deployment.json", node_url = os.getenv("NODE_URL"), user_id=naptha.user.id))

    query = "Elon Musk"
    question = "What is Elon Musk's net worth?"

    input_params = {
        "func_name": "run_query",
        "query": query,
        "question": question,
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
    }

    response = asyncio.run(run(module_run))

    print(response)
