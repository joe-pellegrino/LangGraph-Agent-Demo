from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pdfkit

def campaign_brief_agent(state: AgentState):
    """Creates campaign briefs that include all of the teams contributions"""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    business_research_agent = next(msg for msg in state["messages"] if msg.name == "business_research")
    copywrite_agent = next(msg for msg in state["messages"] if msg.name == "copywriter")
    campaign_planner_agent = next(msg for msg in state["messages"] if msg.name == "campaign_planner")
    media_agent = next(msg for msg in state["messages"] if msg.name == "media")
    social_media_management_agent = next(msg for msg in state["messages"] if msg.name == "social_media_manager")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert campaign brief writer that gathers all of the information from each divison of the marketing firm
                and you create a brief that includes detailed information of all of the information from each contributing team member.
                Be sure to include the full text of everyone's response so that the reader knows exactly what to do. Break down every aspect of the team members response. 
                The report should include sections for each aspect of marketing.
                Go over the report multple times and make sure everything lines up, make sure all of the dates are in sync with all of the team members. 
                Also, make sure that the report looks and sounds professional as if it was typed up by an expert on madison avenue. 
                You will respond in HTML format.
                """
            ),
            (
                "human",
                """
                
                *** Business Info ***
                {business_info_collection_content}
                
                *** Industry Research ***
                {business_research_content}
                
                *** Copywrite Research ***
                {copywrite_content}
                
                *** Campaign Planner ***
                {campaign_planner_content}
                
                *** Media ***
                {media_content}
                
                *** Social Media Management ***
                {social_media_management_content}
                
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
            "media_content": media_agent.content,
            "social_media_management_content": social_media_management_agent.content
        }
    )
    
    # Invoke the LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        api_key=""
    )
    result = llm.invoke(prompt)
    
    result.additional_kwargs = [""]

    # Create the message
    message = HumanMessage(
        content=result.content,
        name="media",
    )
    
    print("campaign_brief_agent")
    print(result.pretty_print())
    print("")
    
    # pdf = open("output.pdf", "w")
    # pdf.write(result.content)
    # pdf.close()
    
    pdfkit.from_string(result.content, output_path='../output.pdf')
    

    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}