import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container  # or use chat container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
    # def on_agent_action(self, action, **kwargs):
    #     self.step_count += 1
    #     st.write(action)
    #     with self.container.expander(f"🧠 Step {self.step_count}: Agent Action", expanded=True):
    #         st.markdown(f"**🧠 Thought**: {action.log}")
    #         st.markdown(f"**🔧 Action**: `{action.tool}`")
    #         st.markdown(f"**📥 Input**: `{action.tool_input[:30]}`")

    # def on_tool_start(self, serialized, input_str, **kwargs):
    #     self.step_count += 1
    #     st.write(serialized)
    #     with self.container.expander(f"🧠 Step {self.step_count}: {serialized['name']}", expanded=self.expand_all):
    #         st.markdown(f"**🔧 Tool**: `{serialized['name']}`")
    #         st.markdown(f"**📥 Input**: `{input_str}`")

    def on_agent_finish(self, result, **kwargs):
        if self.text=="":
            self.container.markdown('Oops! Something went wrong. Please give it another try!')
