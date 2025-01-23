from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning

def business_info_collection_agent(state: AgentState):
    """Ensures that the user has provided the necessary information to begin the campaign."""
    
    # Create the message
    message = HumanMessage(
        content="I want to launch a campaign for Saverio's Pizza located in Massapequa, NY.",
        name="business_info_collection",
    )
    
    print("business_info_collection_agent")
    print(message)
    print("")

    return {"messages": state["messages"] + [message]}