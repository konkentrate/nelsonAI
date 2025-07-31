from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["long_term", "short_term", "query"],
    template=(
        "You are a Discord chat bot. You are given some context before the user's querry. \n"
        "Relevant long-term messages:\n{long_term}\n\n"
        "Recent short-term conversation:\n{short_term}\n\n"
        "Current user query:\n{query}\n"
    )
)

