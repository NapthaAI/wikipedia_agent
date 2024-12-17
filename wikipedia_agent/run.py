import logging
from naptha_sdk.client.node import Node
from naptha_sdk.schemas import AgentRunInput

logger = logging.getLogger(__name__)


async def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {agent_run.inputs}")

    kb_deployment = agent_run.kb_deployment
    table_name = kb_deployment.kb_config["table_name"]
    table_schema = kb_deployment.kb_config["schema"]

    kb_node = Node(kb_deployment.kb_node_url)
    llm_node = Node(agent_run.agent_deployment.worker_node_url)

    query = agent_run.inputs.query
    question = agent_run.inputs.question

    # Retrieve the wikipedia page
    page = await kb_node.query_table(table_name=table_name, condition={'title': query})

    if not page:
        return {"error": "Page not found"}
    
    page = page.json()['rows'][0]['text']
    logger.info(f"Page content: {page}")

    # Create a prompt for the LLM
    messages = [
        {"role": "system", "content": agent_run.agent_deployment.agent_config.system_prompt.role},
        {"role": "user", "content": f"The user asked: {question}. The wikipedia page content is: {page}\n\nAnswer the question based on the page content."}
    ]
    logger.info(f"Messages: {messages}")

    # Call the LLM
    llm_response = await llm_node.chat(
        messages=messages,
        model=agent_run.agent_deployment.agent_config.llm_config.model
    )
    logger.info(f"LLM response: {llm_response}")
    return {"answer": llm_response}


