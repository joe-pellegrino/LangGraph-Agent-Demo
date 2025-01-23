from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def copywrite_agent(state: AgentState):
    """Provides copy for posts and blogs. Will provide ideas as well."""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    business_research_agent = next(msg for msg in state["messages"] if msg.name == "business_research")
    campaignplanner_agent = next(msg for msg in state["messages"] if msg.name == "campaign_planner")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a copywriter that provides blog ideas vital for the success of a marketing campaign.
                Do exactly as the campaign planner has directed you to do.
                
                """
            ),
            (
                "human",
                """
                Based on the information provided to you below, and with the direction of the campaign planner, return copy to use in posts.
                
                *** Business Info ***
                {business_info_collection_content}
                
                *** Industry Research ***
                {business_research_content}
                
                *** Campaign Planner ***
                {campaign_planner_content}
                """
            )
        ]
    )
    
    prompt = template.invoke(
        {
            "business_info_collection_content": business_info_collection_agent.content,
            "business_research_content": business_research_agent.content,
            "campaign_planner_content": campaignplanner_agent.content
        }
    )
    
    # Invoke the LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        api_key=""
    )
    result = llm.invoke(prompt)

    # Create the message
    message = HumanMessage(
        content=result.content,
        name="copywriter",
    )
    
    print("copywrite_agent")
    print(result.pretty_print())
    print("")

    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}