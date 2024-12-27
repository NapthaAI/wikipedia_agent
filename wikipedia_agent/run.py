import logging
from naptha_sdk.kb import KnowledgeBase
from naptha_sdk.client.node import Node
from naptha_sdk.schemas import AgentRunInput

logger = logging.getLogger(__name__)


async def run(module_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {module_run.inputs}")

    kb_deployment = module_run.deployment.kb_deployments[0]
    kb = KnowledgeBase(kb_deployment)
    llm_node = Node(module_run.deployment.node)

    query = module_run.inputs.query
    question = module_run.inputs.question

    # Retrieve the wikipedia page
    page = await kb.get_kb(column_name='title', column_value=query)

    if not page:
        return {"error": "Page not found"}
    
    logger.info(f"Page content: {page}")

    # Create a prompt for the LLM

    messages = [
        {"role": "system", "content": module_run.deployment.config.system_prompt['role']},
        {"role": "user", "content": f"The user asked: {question}. The wikipedia page content is: {page['text']}\n\nAnswer the question based on the page content."}
    ]
    logger.info(f"Messages: {messages}")

    # Call the LLM
    input_ = {
        "messages": messages,
        "model": module_run.deployment.config.llm_config.model
    }
    llm_response = await llm_node.run_inference(input_)
    logger.info(f"LLM response: {llm_response}")
    return {"answer": llm_response}


