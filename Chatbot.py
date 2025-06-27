from together import Together

with open("together_api_key.txt", "r") as f:
    TOGETHER_API_KEY = f.read().strip()


client = Together(api_key=TOGETHER_API_KEY)



import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
st.set_page_config(page_title="🧠 Multi-Agent AI Assistant", layout="centered")


config_list = [
    {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "api_key": TOGETHER_API_KEY,
        "base_url": "https://api.together.xyz/v1",
        "max_tokens": 512 
    }
]


@st.cache_resource
def init_agents():
    analyzer = AssistantAgent(
        name="QuestionAnalyzerAgent",
        system_message="You analyze the user's query and decide what it's about.",
        llm_config={"config_list": config_list}
    )
    knowledge = AssistantAgent(
        name="KnowledgeAgent",
        system_message="You fetch and explain information based on the topic.",
        llm_config={"config_list": config_list}
    )
    formatter = AssistantAgent(
        name="AnswerFormatterAgent",
        system_message="You convert the explanation into a friendly chatbot reply.",
        llm_config={"config_list": config_list}
    )
    code_writer = AssistantAgent(
        name="CodeWriterAgent",
        system_message="You write Python code to solve problems described by the user or other agents.",
        llm_config={"config_list": config_list}
    )
    image_generator = AssistantAgent(
        name="ImageGeneratorAgent",
        system_message=(
            "You create text descriptions of images based on the user's request. "
            "Describe the scene in vivid detail as a prompt for an image generation model."
        ),
        llm_config={"config_list": config_list}
    )
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        llm_config={"config_list": config_list}
    )
    groupchat = GroupChat(
        agents=[user_proxy, analyzer, knowledge, formatter, code_writer, image_generator],
        messages=[],
        max_round=10,
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list}
    )
    return user_proxy, manager


user_proxy, manager = init_agents()





st.title("🤖 Multi-Agent Chatbot")
st.markdown("Ask me anything!")

# Initializing chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Showing previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_prompt := st.chat_input("Type your message here..."):
    if len(user_prompt.split()) > 300:
        warning = "⚠️ Your message is too long. Please limit it to 300 words for best performance."
        st.session_state.chat_history.append({"role": "assistant", "content": warning})
        with st.chat_message("assistant"):
            st.warning(warning)

    else:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        manager.groupchat.messages = manager.groupchat.messages[-6:]

        # AutoGen response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    user_proxy.initiate_chat(manager, message=user_prompt)

                    # Grab last agent reply
                    reply = None
                    for msg in reversed(manager.groupchat.messages):
                        if msg["name"] != "User":
                            reply = f"**{msg['name']}**: {msg['content']}"
                            break

                    if reply:
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    else:
                        fallback = "I'm not sure how to respond right now."
                        st.markdown(fallback)
                        st.session_state.chat_history.append({"role": "assistant", "content": fallback})

                except Exception as e:
                    err_msg = f"⚠️ Error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
