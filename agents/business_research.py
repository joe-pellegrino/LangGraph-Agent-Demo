from langchain_core.messages import HumanMessage
from state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def business_research_agent(state: AgentState):
    """Researches business niches to deteremine audience, competition."""
    show_reasoning = state["metadata"]['show_reasoning']
    
    #Get the business name and location
    business_info_collection_agent = next(msg for msg in state["messages"] if msg.name == "business_info_collection")
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a market research analyst returning fundamental research data to help guide a marketing campaign. 
                
                """
            ),
            (
                "human",
                """
                Based on the information provided to you below, return information that is vital to a successful marketing campaign outcome.
                
                {business_info_collection_agent}
                """
            )
        ]
    )
    
    prompt = template.invoke(
        {
            "business_info_collection_agent": business_info_collection_agent.content
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
        name="business_research",
    )
    
    print("business_research_agent")
    print(result.pretty_print())
    print("")

    # Print the decision if the flag is set
    # if show_reasoning:
    #     show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}