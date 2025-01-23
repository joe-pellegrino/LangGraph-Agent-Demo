from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def media_agent(state: AgentState):
    """Creates posts with hashtags, as well as image prompts to generate images with AI."""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    business_research_agent = next(msg for msg in state["messages"] if msg.name == "business_research")
    copywrite_agent = next(msg for msg in state["messages"] if msg.name == "copywriter")
    campaign_planner_agent = next(msg for msg in state["messages"] if msg.name == "campaign_planner")
    social_media_manager_agent = next(msg for msg in state["messages"] if msg.name == "social_media_manager")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a seasoned content creator in the art of photography and videos. 
                You will provide the shoot style, length, and script for video content. 
                Make sure your content schedule aligns with the campaign planner and the posts that the social media manager has lined up. 
                Provide instructions for each shoot as to how to shoot it and the camera or phone to use for the shoot. 
                """
            ),
            (
                "human",
                """
                You will follow the direction of the campaign planner and use the copywiters content to recommend video and photo shoots.
                Based on the information provided to you below, create at least 1 week worth of posts.
                
                *** Business Info ***
                {business_info_collection_content}
                
                *** Industry Research ***
                {business_research_content}
                
                *** Copywrite Research ***
                {copywrite_content}
                
                *** Campaign Planner ***
                {campaign_planner_content}
                
                 *** Social Media Manager ***
                {social_media_manager_content}
                
                """
            )
        ]
    )
    
    prompt = template.invoke(
        {
            "business_info_collection_content": business_info_collection_agent.content,
            "business_research_content": business_research_agent.content,
            "copywrite_content": copywrite_agent.content,
            "campaign_planner_content": campaign_planner_agent.content,
            "social_media_manager_content": social_media_manager_agent.content
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
        name="media",
    )
    
    print("media_agent")
    print(result.pretty_print())
    print("")

    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}