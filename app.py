import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Text to Math Problem Solver And Data Search Assistant",
    page_icon="🧮"
)

st.title("Text to Math Problem Solver using Groq (Gemma 2)")

# ---------------- API KEY ----------------
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your GROQ API key to continue")
    st.stop()

# ---------------- LLM (UNCHANGED) ----------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# ---------------- TOOLS ----------------

# Wikipedia tool (UNCHANGED)
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for factual information"
)

# ✅ SAFE Calculator (FIX)
def safe_calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "Invalid mathematical expression"

calculator = Tool(
    name="Calculator",
    func=safe_calculator,
    description=(
        "Use ONLY for pure mathematical expressions like "
        "'5 + 3 * 2'. Do NOT use for word problems."
    )
)

# Reasoning tool (UNCHANGED)
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an intelligent agent.
Solve the following problem step-by-step with clear explanation.

Question: {question}

Answer:
"""
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt)

reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.run,
    description="Solve logical and reasoning-based questions"
)

# ---------------- AGENT (UNCHANGED) ----------------
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I can solve math problems and search information for you!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- USER INPUT ----------------
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes."
)

if st.button("Find my answer"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            response = assistant_agent.invoke(
                {"input": question},
                callbacks=[st_cb]
            )

        final_answer = response["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )

        st.chat_message("assistant").write(final_answer)

    else:
        st.warning("Please enter a question")
