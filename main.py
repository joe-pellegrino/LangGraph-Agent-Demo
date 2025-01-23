from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
from state import AgentState

# import the agents
from agents.business_research import business_research_agent
from agents.business_info_collection import business_info_collection_agent
from agents.copywrite import copywrite_agent
from agents.social_media_manager import social_media_management_agent
from agents.campaign_planner import campaign_planner_agent
from agents.media import media_agent
from agents.campaign_brief import campaign_brief_agent


llm = ChatOpenAI(
        model="gpt-4o", 
        api_key="",
        streaming=True
        )

# State
class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]
    
    
    
workflow = StateGraph(AgentState)

workflow.add_node("business_info_collection_agent", business_info_collection_agent)
workflow.add_node("business_research_agent", business_research_agent)
workflow.add_node("campaign_planner_agent", campaign_planner_agent)
workflow.add_node("copywrite_agent", copywrite_agent)
workflow.add_node("social_media_manager_agent", social_media_management_agent)
workflow.add_node("media_agent", media_agent)
workflow.add_node("campaign_brief_agent", campaign_brief_agent)


workflow.set_entry_point("business_info_collection_agent")
workflow.add_edge("business_info_collection_agent", "business_research_agent")
workflow.add_edge("business_research_agent", "campaign_planner_agent")
workflow.add_edge("campaign_planner_agent", "copywrite_agent")  
workflow.add_edge("copywrite_agent", "social_media_manager_agent")
workflow.add_edge("social_media_manager_agent", "media_agent")
workflow.add_edge("media_agent", "campaign_brief_agent")
workflow.add_edge("campaign_brief_agent", END)


app = workflow.compile()




final_state = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="Plan my marketing camaign for me.",
            )
        ],
         "metadata": {
                "show_reasoning": "",
            }
        
    },
    stream_mode="messages-tuple"
)["messages"][-1].content

#display(Image(app.get_graph().draw_mermaid_png()))



#llm_with_tools = llm.bind_tools([get_campaign_brief, get_copy])

#tool_call = llm_with_tools.invoke("I need to create a marketing plan for a new pizza product. My place is located on long island.")

#tool_call.additional_kwargs['tool_calls']

#print(tool_call)