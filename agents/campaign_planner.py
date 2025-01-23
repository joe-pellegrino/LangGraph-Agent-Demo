from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def campaign_planner_agent(state: AgentState):
    """Plans out a marketing campaign overview for the rest of the team"""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    business_research_agent = next(msg for msg in state["messages"] if msg.name == "business_research")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a marketing campaign planner that provides a high-level overview of the marketing campaign with direction for the rest of the teams involved. 
                The other teams involved are 
                - Copywriter
                - Media (photography & videography)
                - Social Media Manager
                
                """
            ),
            (
                "human",
                """
                Based on the information provided to you below, return a marketing plan that is vital to a cuccessful marketing campaign outcome.
                
                *** Business Info ***
                {business_info_collection_content}
                
                *** Industry Research ***
                {business_research_content}
                """
            )
        ]
    )
    
    prompt = template.invoke(
        {
            "business_info_collection_content": business_info_collection_agent.content,
            "business_research_content": business_research_agent.content
        }
    )
    
    # Invoke the LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        api_key=""
    )
    
    print("Prompt")
    print(prompt)
    print("")
    
    result = llm.invoke(prompt)

    # Create the message
    message = HumanMessage(
        content=result.content,
        name="campaign_planner",
    )

    print("campaign_planner_agent")
    print(result.pretty_print())
    print("")
    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}