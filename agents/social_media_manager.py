from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def social_media_management_agent(state: AgentState):
    """Creates posts with hashtags, as well as image prompts to generate images with AI."""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    business_research_agent = next(msg for msg in state["messages"] if msg.name == "business_research")
    copywrite_agent = next(msg for msg in state["messages"] if msg.name == "copywriter")
    campaign_planner_agent = next(msg for msg in state["messages"] if msg.name == "campaign_planner")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You create posts for Instagram, TikTok, Facebook, and LinkedIn.
                Make sure you use the same copy that the copywriter has provided you below. 
                All you are doing is ensuring the right number of words are used for each platform and use hashtags if needed. 
                """
            ),
            (
                "human",
                """
                You will follow the direction of the campaign planner and use the copywiters content to create posts.
                Add relevant hashtags and image prompts to the posts.
                Based on the information provided to you below, create at least 1 week worth of posts.
                
                *** Business Info ***
                {business_info_collection_content}
                
                *** Industry Research ***
                {business_research_content}
                
                *** Copywrite Research ***
                {copywrite_content}
                
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
            "copywrite_content": copywrite_agent.content,
            "campaign_planner_content": campaign_planner_agent.content
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
        name="social_media_manager",
    )

    print("social_media_manager_agent")
    print(result.pretty_print())
    print("")
    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}